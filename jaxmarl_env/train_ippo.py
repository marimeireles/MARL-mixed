"""
Self-contained feed-forward IPPO for HeterogeneousIPD, adapted from JaxMARL's
`baselines/IPPO/ippo_ff_mpe.py` (PureJaxRL lineage). Agents share one policy
network (homogeneous learners), each acting on its own masked observation.

NOTE: this requires a newer-Python / recent-JAX environment with `jaxmarl`,
`flax`, `optax`, and `distrax` installed, ideally on GPU. It cannot run in the
Python-3.8 / jax-0.4.13 environment used by the CRLD side of this repo. If you
prefer the maintained implementation, register HeterogeneousIPD with
`jaxmarl.registration` and run the official `ippo_ff_*` baseline instead (see
README).

`train(config) -> metrics`; `final_coop_rate(...)` returns the mean cooperation
rate over the last rollouts, which is what the regime x N table records.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax

from heterogeneous_ipd import make_hetero_ipd


class Categorical:
    """Minimal stand-in for distrax.Categorical (avoids the tfp dependency)."""

    def __init__(self, logits):
        self.logits = logits
        self.log_p = jax.nn.log_softmax(logits)

    def sample(self, seed):
        return jax.random.categorical(seed, self.logits)

    def log_prob(self, a):
        return jnp.take_along_axis(self.log_p, a[..., None], axis=-1)[..., 0]

    def entropy(self):
        return -jnp.sum(jnp.exp(self.log_p) * self.log_p, axis=-1)


class ActorCritic(nn.Module):
    action_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        act = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                               bias_init=constant(0.0))(x))
        act = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                               bias_init=constant(0.0))(act))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                          bias_init=constant(0.0))(act)
        pi = Categorical(logits=logits)

        val = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                               bias_init=constant(0.0))(x))
        val = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                               bias_init=constant(0.0))(val))
        val = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(val)
        return pi, jnp.squeeze(val, -1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def batchify(d, agents, n):
    return jnp.stack([d[a] for a in agents]).reshape(n, -1)


def unbatchify(x, agents):
    return {a: x[i] for i, a in enumerate(agents)}


def make_train(config):
    env = make_hetero_ipd(num_agents=config["NUM_AGENTS"], regime=config["REGIME"],
                          num_steps=config["ENV_STEPS"])
    agents = env.agents
    nA = env.num_agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    act_dim = env.action_space(agents[0]).n

    n_actors = nA * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (
        config["NUM_STEPS"] * config["NUM_ENVS"])
    minibatch = (config["NUM_STEPS"] * n_actors) // config["NUM_MINIBATCHES"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                      ) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        net = ActorCritic(act_dim, config["HIDDEN"])
        rng, _r = jax.random.split(rng)
        params = net.init(_r, jnp.zeros((1, obs_dim)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(linear_schedule if config["ANNEAL_LR"] else config["LR"],
                       eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        rng, _r = jax.random.split(rng)
        reset_keys = jax.random.split(_r, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_keys)

        def update_step(runner, _):
            def env_step(runner, _):
                train_state, env_state, last_obs, rng = runner
                obs_b = batchify(last_obs, agents, n_actors)
                pi, value = net.apply(train_state.params, obs_b)
                rng, _r = jax.random.split(rng)
                action = pi.sample(seed=_r)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action.reshape(nA, config["NUM_ENVS"]), agents)

                rng, _r = jax.random.split(rng)
                step_keys = jax.random.split(_r, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    step_keys, env_state, env_act)
                transition = Transition(
                    batchify(done, agents, n_actors).squeeze(),
                    action,
                    value,
                    batchify(reward, agents, n_actors).squeeze(),
                    log_prob,
                    obs_b,
                )
                return (train_state, env_state, obsv, rng), (transition, info)

            runner, (traj, info) = jax.lax.scan(
                env_step, runner, None, config["NUM_STEPS"])

            train_state, env_state, last_obs, rng = runner
            last_b = batchify(last_obs, agents, n_actors)
            _, last_val = net.apply(train_state.params, last_b)

            def gae(traj, last_val):
                def step(carry, t):
                    gae, next_val = carry
                    delta = t.reward + config["GAMMA"] * next_val * (1 - t.done) - t.value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * gae
                    return (gae, t.value), gae
                _, advantages = jax.lax.scan(
                    step, (jnp.zeros_like(last_val), last_val), traj,
                    reverse=True, unroll=16)
                return advantages, advantages + traj.value

            advantages, targets = gae(traj, last_val)

            def epoch(state, _):
                def minibatch_update(train_state, batch):
                    traj, adv, tgt = batch

                    def loss_fn(params, traj, adv, tgt):
                        pi, value = net.apply(params, traj.obs)
                        log_prob = pi.log_prob(traj.action)
                        v_clipped = traj.value + (value - traj.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"])
                        v_loss = 0.5 * jnp.maximum(
                            (value - tgt) ** 2, (v_clipped - tgt) ** 2).mean()
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
                        l1 = ratio * adv_n
                        l2 = jnp.clip(ratio, 1 - config["CLIP_EPS"],
                                      1 + config["CLIP_EPS"]) * adv_n
                        actor_loss = -jnp.minimum(l1, l2).mean()
                        entropy = pi.entropy().mean()
                        return (actor_loss + config["VF_COEF"] * v_loss
                                - config["ENT_COEF"] * entropy)

                    grads = jax.grad(loss_fn)(train_state.params, traj, adv, tgt)
                    return train_state.apply_gradients(grads=grads), None

                train_state, traj, adv, tgt, rng = state
                rng, _r = jax.random.split(rng)
                batch = (traj, adv, tgt)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_STEPS"] * n_actors,) + x.shape[2:]),
                    batch)
                perm = jax.random.permutation(_r, config["NUM_STEPS"] * n_actors)
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                    batch)
                train_state, _ = jax.lax.scan(
                    minibatch_update, train_state, minibatches)
                return (train_state, traj, adv, tgt, rng), None

            state = (train_state, traj, advantages, targets, rng)
            state, _ = jax.lax.scan(epoch, state, None, config["UPDATE_EPOCHS"])
            train_state = state[0]
            rng = state[-1]
            metric = {"coop_rate": info["coop_rate"].mean()}
            return (train_state, env_state, last_obs, rng), metric

        rng, _r = jax.random.split(rng)
        runner = (train_state, env_state, obsv, _r)
        runner, metrics = jax.lax.scan(update_step, runner, None,
                                       config["NUM_UPDATES"])
        return {"runner": runner, "metrics": metrics}

    return train


DEFAULT_CONFIG = dict(
    LR=2.5e-4, ANNEAL_LR=True, NUM_ENVS=64, NUM_STEPS=128, ENV_STEPS=128,
    TOTAL_TIMESTEPS=2_000_000, UPDATE_EPOCHS=4, NUM_MINIBATCHES=4,
    GAMMA=0.99, GAE_LAMBDA=0.95, CLIP_EPS=0.2, ENT_COEF=0.01, VF_COEF=0.5,
    MAX_GRAD_NORM=0.5, HIDDEN=64, NUM_AGENTS=2, REGIME="full",
)


def final_coop_rate(num_agents, regime, seed=0, **overrides):
    """Train once; return mean cooperation rate over the final 10% of updates."""
    config = {**DEFAULT_CONFIG, "NUM_AGENTS": num_agents, "REGIME": regime,
              **overrides}
    out = jax.jit(make_train(config))(jax.random.PRNGKey(seed))
    coop = np.asarray(out["metrics"]["coop_rate"])
    tail = max(1, len(coop) // 10)
    return float(coop[-tail:].mean())


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    regime = sys.argv[2] if len(sys.argv) > 2 else "full"
    print(f"N={N} regime={regime} -> coop_rate={final_coop_rate(N, regime):.3f}")
