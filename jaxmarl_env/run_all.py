"""
Generic IPPO driver that trains on ONE JaxMARL environment and logs the
learning dynamics (mean reward per step vs. training update).

Designed to be launched once per environment by a SLURM array job. It
introspects the env and runs a shared-policy feed-forward IPPO when the env is
compatible (discrete actions, flat-or-flattenable observations), otherwise it
writes a `skipped` status with the reason. This lets a single sweep attempt
every registered env and produce curves for the ones that fit, while clearly
reporting what needs a different policy/encoder head (continuous -> Gaussian,
image -> CNN).

Usage:
    python run_all.py <ENV_NAME> [--timesteps N] [--num_envs N] [--seed S]

Outputs (under results/):
    <ENV_NAME>.npz     dynamics array + metadata        (on success)
    <ENV_NAME>.png     learning-curve plot              (on success)
    <ENV_NAME>.status  one line: ok / skipped:<reason> / failed:<reason>
"""
import argparse, os, traceback
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax

import jaxmarl
from jaxmarl.environments.spaces import Discrete

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
MAX_OBS_DIM = 6000   # above this we assume an image obs that wants a CNN -> skip


class SkipEnv(Exception):
    pass


# --------------------------------------------------------------------------- net
class Categorical:
    def __init__(self, logits, mask=None):
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        self.logits = logits
        self.log_p = jax.nn.log_softmax(logits)

    def sample(self, seed):
        return jax.random.categorical(seed, self.logits)

    def log_prob(self, a):
        return jnp.take_along_axis(self.log_p, a[..., None], axis=-1)[..., 0]

    def entropy(self):
        return -jnp.sum(jnp.exp(self.log_p) * jnp.where(jnp.isinf(self.log_p), 0.0,
                                                        self.log_p), axis=-1)


class ActorCritic(nn.Module):
    action_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(a))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(a)
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(v))
        v = nn.Dense(1, kernel_init=orthogonal(1.0))(v)
        return logits, jnp.squeeze(v, -1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray


# ----------------------------------------------------------------- introspection
def introspect(env):
    agents = env.agents
    obs_dims, act_dims = [], []
    for a in agents:
        ospace = env.observation_space(a) if callable(getattr(env, "observation_space", None)) \
            else env.observation_spaces[a]
        aspace = env.action_space(a) if callable(getattr(env, "action_space", None)) \
            else env.action_spaces[a]
        if not isinstance(aspace, Discrete):
            raise SkipEnv("continuous action space (needs Gaussian policy)")
        obs_dims.append(int(np.prod(ospace.shape)))
        act_dims.append(int(aspace.n))
    obs_dim, act_dim = max(obs_dims), max(act_dims)
    if obs_dim > MAX_OBS_DIM:
        raise SkipEnv(f"obs dim {obs_dim} too large (image obs -> needs CNN)")
    act_mask = np.zeros((len(agents), act_dim), bool)
    for i, n in enumerate(act_dims):
        act_mask[i, :n] = True
    return agents, obs_dim, act_dim, jnp.array(act_mask), obs_dims


# ------------------------------------------------------------------------- train
def make_train(env, config, meta):
    agents, obs_dim, act_dim, act_mask, obs_dims = meta
    nA = len(agents)
    n_actors = nA * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (
        config["NUM_STEPS"] * config["NUM_ENVS"])
    full_mask = jnp.repeat(act_mask, config["NUM_ENVS"], axis=0)  # (n_actors, act_dim)

    def flat_pad(x):  # (num_envs, *obs_shape) -> (num_envs, obs_dim)
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] < obs_dim:
            x = jnp.pad(x, ((0, 0), (0, obs_dim - x.shape[1])))
        return x

    def batchify(d):
        return jnp.concatenate([flat_pad(d[a]) for a in agents], axis=0)  # (nA*envs,obs)

    def batch_rew(d):
        return jnp.concatenate([d[a].reshape(-1) for a in agents], axis=0)

    def unbatchify(x):
        x = x.reshape(nA, config["NUM_ENVS"])
        return {a: x[i] for i, a in enumerate(agents)}

    def train(rng):
        net = ActorCritic(act_dim, config["HIDDEN"])
        rng, _r = jax.random.split(rng)
        params = net.init(_r, jnp.zeros((1, obs_dim)))
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                         optax.adam(config["LR"], eps=1e-5))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        rng, _r = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(_r, config["NUM_ENVS"]))

        def update_step(runner, _):
            def env_step(runner, _):
                ts, env_state, last_obs, rng = runner
                obs_b = batchify(last_obs)
                logits, value = net.apply(ts.params, obs_b)
                pi = Categorical(logits, mask=full_mask)
                rng, _r = jax.random.split(rng)
                action = pi.sample(_r)
                lp = pi.log_prob(action)
                env_act = unbatchify(action)
                rng, _r = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(_r, config["NUM_ENVS"]), env_state, env_act)
                t = Transition(batch_rew(done) if isinstance(done, dict) else None,
                               action, value, batch_rew(reward), lp, obs_b, full_mask)
                # done dict has per-agent + __all__; build per-actor done
                d = jnp.concatenate([done[a].reshape(-1) for a in agents], axis=0)
                t = t._replace(done=d)
                return (ts, env_state, obsv, rng), (t, batch_rew(reward).mean())

            runner, (traj, rew_mean) = jax.lax.scan(
                env_step, runner, None, config["NUM_STEPS"])
            ts, env_state, last_obs, rng = runner
            _, last_val = net.apply(ts.params, batchify(last_obs))

            def gae(traj, last_val):
                def step(carry, t):
                    g, nv = carry
                    delta = t.reward + config["GAMMA"] * nv * (1 - t.done) - t.value
                    g = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * g
                    return (g, t.value), g
                _, adv = jax.lax.scan(step, (jnp.zeros_like(last_val), last_val),
                                      traj, reverse=True, unroll=16)
                return adv, adv + traj.value

            adv, tgt = gae(traj, last_val)

            def epoch(state, _):
                def mb(ts, batch):
                    traj, adv, tgt = batch

                    def loss_fn(p):
                        logits, value = net.apply(p, traj.obs)
                        pi = Categorical(logits, mask=traj.mask)
                        lp = pi.log_prob(traj.action)
                        vcl = traj.value + (value - traj.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        vloss = 0.5 * jnp.maximum((value - tgt) ** 2,
                                                  (vcl - tgt) ** 2).mean()
                        ratio = jnp.exp(lp - traj.log_prob)
                        an = (adv - adv.mean()) / (adv.std() + 1e-8)
                        aloss = -jnp.minimum(ratio * an,
                                             jnp.clip(ratio, 1 - config["CLIP_EPS"],
                                                      1 + config["CLIP_EPS"]) * an).mean()
                        return aloss + config["VF_COEF"] * vloss \
                            - config["ENT_COEF"] * pi.entropy().mean()

                    grads = jax.grad(loss_fn)(ts.params)
                    return ts.apply_gradients(grads=grads), None

                ts, traj, adv, tgt, rng = state
                rng, _r = jax.random.split(rng)
                B = config["NUM_STEPS"] * n_actors
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((B,) + x.shape[2:]), (traj, adv, tgt))
                perm = jax.random.permutation(_r, B)
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                mbs = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                    batch)
                ts, _ = jax.lax.scan(mb, ts, mbs)
                return (ts, traj, adv, tgt, rng), None

            state = (ts, traj, adv, tgt, rng)
            state, _ = jax.lax.scan(epoch, state, None, config["UPDATE_EPOCHS"])
            ts, rng = state[0], state[-1]
            return (ts, env_state, last_obs, rng), rew_mean.mean()

        rng, _r = jax.random.split(rng)
        runner = (ts, env_state, obsv, _r)
        runner, dynamics = jax.lax.scan(update_step, runner, None,
                                        config["NUM_UPDATES"])
        return dynamics   # (NUM_UPDATES,) mean reward per update

    return train


DEFAULT = dict(LR=2.5e-4, NUM_ENVS=64, NUM_STEPS=128, TOTAL_TIMESTEPS=3_000_000,
               UPDATE_EPOCHS=4, NUM_MINIBATCHES=4, GAMMA=0.99, GAE_LAMBDA=0.95,
               CLIP_EPS=0.2, ENT_COEF=0.01, VF_COEF=0.5, MAX_GRAD_NORM=0.5, HIDDEN=64)


def run(env_name, config, seed):
    status = os.path.join(RESULTS, f"{env_name}.status")
    try:
        env = jaxmarl.make(env_name)
        meta = introspect(env)
        train = jax.jit(make_train(env, config, meta))
        dynamics = np.asarray(train(jax.random.PRNGKey(seed)))
        x = (np.arange(len(dynamics)) + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]
        np.savez(os.path.join(RESULTS, f"{env_name}.npz"),
                 dynamics=dynamics, timesteps=x, num_agents=env.num_agents)
        _plot(env_name, x, dynamics, env.num_agents)
        open(status, "w").write("ok\n")
        print(f"[ok] {env_name}: reward {dynamics[0]:.3f} -> {dynamics[-1]:.3f}")
    except SkipEnv as e:
        open(status, "w").write(f"skipped:{e}\n")
        print(f"[skip] {env_name}: {e}")
    except Exception as e:
        open(status, "w").write(f"failed:{repr(e)}\n")
        print(f"[fail] {env_name}: {e}")
        traceback.print_exc()


def _plot(name, x, y, nA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(x, y, color="#0254a3")
    ax.set_title(f"{name}  (N={nA})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean reward / step")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"{name}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("env")
    p.add_argument("--timesteps", type=int, default=DEFAULT["TOTAL_TIMESTEPS"])
    p.add_argument("--num_envs", type=int, default=DEFAULT["NUM_ENVS"])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=a.timesteps, NUM_ENVS=a.num_envs)
    run(a.env, cfg, a.seed)
