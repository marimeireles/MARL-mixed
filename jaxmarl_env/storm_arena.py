"""CNN IPPO trainer for the JaxMARL STORM "in-the-matrix" arena (storm_np).

STORM (class InTheMatrix) is a general-sum, N-agent gridworld where agents move
on a 2D map, pick up red ("cooperate") / blue ("defect") coins that fill an
inventory, and resolve a matrix game (default a Prisoner's Dilemma) when they
interact. The observation is a per-agent egocentric (5, 5, 14) image, so the
policy is a small CNN actor-critic with parameters shared across all agents
(IPPO with parameter sharing).

This file adapts the CNN PPO loop from `cnn_algos.py` to storm_np's NONSTANDARD
array API:
  * `obs, state = env.reset(key)`  ->  obs is a plain array (N, 5, 5, 14).
  * `env.step(key, state, actions)` wants `actions` as jnp.array shape (N,).
  * `reward` is an array shape (N,); `done` is a dict with '__all__'.
We vmap reset/step over NUM_ENVS and keep every quantity in env-major flat
order (index = env * N + agent), reshaping to (NUM_ENVS, N) only at the env
boundary.

Experiments (each writes a figure + .npy into results/):
  1. number of agents: N in {2, 4, 8}        -> results/storm_nagents.{png,npy}
  2. observability   : N=2, full vs blind     -> results/storm_obs.{png,npy}
     ("blind to others" zeroes channel 5, the other-agent-present channel,
      in the observation before the CNN.)

Usage:
    python storm_arena.py --timesteps 3000000          # run both experiments
    python storm_arena.py --debug                      # quick 100k smoke test
"""
import argparse
import os
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
import optax
import jaxmarl

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

OTHER_AGENT_CHANNEL = 5   # channel that marks "another agent is present here"


# ------------------------------------------------------------------- network
class CNNAC(nn.Module):
    """Shared CNN actor-critic for (5, 5, 14) egocentric STORM observations."""
    action_dim: int

    @nn.compact
    def __call__(self, x):                       # x: (B, 5, 5, 14) float
        h = nn.relu(nn.Conv(32, (3, 3), padding="SAME",
                            kernel_init=orthogonal(np.sqrt(2)))(x))
        h = nn.relu(nn.Conv(64, (3, 3), padding="SAME",
                            kernel_init=orthogonal(np.sqrt(2)))(h))
        h = h.reshape(h.shape[0], -1)
        h = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(h))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)
        v = nn.Dense(1, kernel_init=orthogonal(1.0))(h)
        return logits, jnp.squeeze(v, -1)


class Categorical:
    def __init__(self, logits):
        self.logits = logits
        self.log_p = jax.nn.log_softmax(logits)

    def sample(self, k):
        return jax.random.categorical(k, self.logits)

    def log_prob(self, a):
        return jnp.take_along_axis(self.log_p, a[..., None], -1)[..., 0]

    def entropy(self):
        return -jnp.sum(jnp.exp(self.log_p) * self.log_p, -1)


class Tr(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


DEFAULT = dict(LR=2.5e-4, NUM_ENVS=64, NUM_STEPS=128, TOTAL_TIMESTEPS=3_000_000,
               UPDATE_EPOCHS=4, NUM_MINIBATCHES=4, GAMMA=0.99, GAE_LAMBDA=0.95,
               CLIP_EPS=0.2, ENT_COEF=0.01, VF_COEF=0.5, MAX_GRAD_NORM=0.5,
               NUM_INNER_STEPS=152, SEED=0)


# --------------------------------------------------------------------- train
def make_train(num_agents, config, mask_others=False):
    """Build a jit-able IPPO training function for storm_np with `num_agents`."""
    N = num_agents
    NE = config["NUM_ENVS"]
    n_actors = N * NE
    act_dim = 8
    img_shape = (5, 5, 14)
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * NE)

    env = jaxmarl.make("storm_np", num_agents=N,
                       num_inner_steps=config["NUM_INNER_STEPS"])

    def batch(ob):
        """(NE, N, 5,5,14) -> (NE*N, 5,5,14), optionally blinding to others."""
        x = ob.reshape((n_actors,) + img_shape).astype(jnp.float32)
        if mask_others:
            x = x.at[..., OTHER_AGENT_CHANNEL].set(0.0)
        return x

    def fmt_act(act):                # (NE*N,) -> (NE, N) array the env wants
        return act.reshape(NE, N)

    def rew_b(rew):                  # (NE, N) -> (NE*N,)
        return rew.reshape(n_actors)

    def done_b(done):                # __all__ (NE,) -> (NE*N,) env-major
        d = done["__all__"].reshape(NE, 1)
        return jnp.broadcast_to(d, (NE, N)).reshape(n_actors)

    def train(rng):
        net = CNNAC(act_dim)
        rng, r = jax.random.split(rng)
        params = net.init(r, jnp.zeros((1,) + img_shape))
        ts = TrainState.create(
            apply_fn=net.apply, params=params,
            tx=optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                           optax.adam(config["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(run, _):
            def step(run, _):
                ts, st, ob, rng = run
                logits, val = net.apply(ts.params, batch(ob))
                pi = Categorical(logits)
                rng, r = jax.random.split(rng)
                act = pi.sample(r)
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(
                    jax.random.split(r, NE), st, fmt_act(act))
                rb = rew_b(rew)
                tr = Tr(done_b(dn), act, val, rb, pi.log_prob(act), batch(ob))
                return (ts, st, ob2, rng), (tr, rb.mean())

            run, (traj, rm) = jax.lax.scan(step, run, None, config["NUM_STEPS"])
            ts, st, last_ob, rng = run
            _, last_val = net.apply(ts.params, batch(last_ob))

            def gae(traj, lv):
                def f(carry, t):
                    g, nv = carry
                    d = t.reward + config["GAMMA"] * nv * (1 - t.done) - t.value
                    g = d + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * g
                    return (g, t.value), g
                _, adv = jax.lax.scan(f, (jnp.zeros_like(lv), lv), traj,
                                      reverse=True, unroll=16)
                return adv, adv + traj.value

            adv, tgt = gae(traj, last_val)

            def epoch(s, _):
                def mb(ts, b):
                    traj, adv, tgt = b

                    def loss(p):
                        logits, val = net.apply(p, traj.obs)
                        pi = Categorical(logits)
                        lp = pi.log_prob(traj.action)
                        an = (adv - adv.mean()) / (adv.std() + 1e-8)
                        ratio = jnp.exp(lp - traj.log_prob)
                        al = -jnp.minimum(
                            ratio * an,
                            jnp.clip(ratio, 1 - config["CLIP_EPS"],
                                     1 + config["CLIP_EPS"]) * an).mean()
                        vl = 0.5 * ((val - tgt) ** 2).mean()
                        return al + config["VF_COEF"] * vl \
                            - config["ENT_COEF"] * pi.entropy().mean()

                    return ts.apply_gradients(grads=jax.grad(loss)(ts.params)), None

                ts, traj, adv, tgt, rng = s
                rng, r = jax.random.split(rng)
                B = config["NUM_STEPS"] * n_actors
                flat = jax.tree_util.tree_map(
                    lambda x: x.reshape((B,) + x.shape[2:]), (traj, adv, tgt))
                perm = jax.random.permutation(r, B)
                flat = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, 0), flat)
                mbs = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                    flat)
                ts, _ = jax.lax.scan(mb, ts, mbs)
                return (ts, traj, adv, tgt, rng), None

            s, _ = jax.lax.scan(epoch, (ts, traj, adv, tgt, rng), None,
                                config["UPDATE_EPOCHS"])
            return (s[0], st, last_ob, s[-1]), rm.mean()

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, config["NUM_UPDATES"])
        return dyn        # (NUM_UPDATES,) mean reward / agent / step per update

    return train


def run_one(num_agents, config, mask_others=False):
    train = jax.jit(make_train(num_agents, config, mask_others))
    dyn = np.asarray(train(jax.random.PRNGKey(config["SEED"])))
    return dyn


# ----------------------------------------------------------------- plotting
def _smooth(y, k=9):
    if len(y) < k:
        return y
    kern = np.ones(k) / k
    return np.convolve(y, kern, mode="same")


def plot_nagents(curves, x_by_n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {2: "#0254a3", 4: "#d1691f", 8: "#2a8f3c"}
    for n in sorted(curves):
        y = curves[n]
        x = x_by_n[n]
        ax.plot(x, y, color=colors.get(n, None), alpha=0.25)
        ax.plot(x, _smooth(y), color=colors.get(n, None), label=f"N = {n}")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean reward / agent / step")
    ax.set_title("STORM in-the-matrix: IPPO vs. number of agents")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_obs(curves, x, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"full obs": "#0254a3", "blind to others": "#b3243f"}
    for name in ["full obs", "blind to others"]:
        y = curves[name]
        ax.plot(x, y, color=styles[name], alpha=0.25)
        ax.plot(x, _smooth(y), color=styles[name], label=name)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean reward / agent / step")
    ax.set_title("STORM in-the-matrix (N=2): full vs. blind-to-others")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def steps_axis(config):
    n_up = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])
    return (np.arange(n_up) + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--debug", action="store_true",
                   help="quick 100k-timestep smoke test on N=2 only")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfg = dict(DEFAULT, NUM_ENVS=a.num_envs, NUM_STEPS=a.num_steps, SEED=a.seed)

    if a.debug:
        cfg["TOTAL_TIMESTEPS"] = 100_000
        print(f"[debug] N=2 full, {cfg['TOTAL_TIMESTEPS']} steps "
              f"({cfg['TOTAL_TIMESTEPS'] // (cfg['NUM_STEPS'] * cfg['NUM_ENVS'])} updates)")
        dyn = run_one(2, cfg, mask_others=False)
        print(f"[debug] N=2 reward {dyn[0]:+.4f} -> {dyn[-1]:+.4f}  "
              f"(min {dyn.min():+.4f} max {dyn.max():+.4f})")
        dyn_m = run_one(2, dict(cfg), mask_others=True)
        print(f"[debug] N=2 blind reward {dyn_m[0]:+.4f} -> {dyn_m[-1]:+.4f}")
        print("[debug] OK -- training ran without error")
        return

    cfg["TOTAL_TIMESTEPS"] = a.timesteps

    # ---- Experiment 1: number of agents ----------------------------------
    print(f"=== Experiment 1: number of agents (timesteps={a.timesteps}) ===")
    curves, x_by_n = {}, {}
    for n in (2, 4, 8):
        c = dict(cfg)
        dyn = run_one(n, c, mask_others=False)
        curves[n] = dyn
        x_by_n[n] = steps_axis(c)
        print(f"  N={n}: reward {dyn[0]:+.4f} -> {dyn[-1]:+.4f} "
              f"(final-10 mean {dyn[-10:].mean():+.4f})")
    np.save(os.path.join(RESULTS, "storm_nagents.npy"),
            {"curves": curves, "x": x_by_n}, allow_pickle=True)
    plot_nagents(curves, x_by_n, os.path.join(RESULTS, "storm_nagents.png"))
    print("  saved results/storm_nagents.{png,npy}")

    # ---- Experiment 2: observability (N=2) -------------------------------
    print(f"=== Experiment 2: observability, N=2 (timesteps={a.timesteps}) ===")
    x = steps_axis(cfg)
    # reuse the N=2 full-obs curve from experiment 1 (same config/seed)
    obs_curves = {"full obs": curves[2]}
    dyn_blind = run_one(2, dict(cfg), mask_others=True)
    obs_curves["blind to others"] = dyn_blind
    print(f"  full obs       : {obs_curves['full obs'][-1]:+.4f} (final)")
    print(f"  blind to others: {dyn_blind[-1]:+.4f} (final, "
          f"final-10 mean {dyn_blind[-10:].mean():+.4f})")
    np.save(os.path.join(RESULTS, "storm_obs.npy"),
            {"curves": obs_curves, "x": x}, allow_pickle=True)
    plot_obs(obs_curves, x, os.path.join(RESULTS, "storm_obs.png"))
    print("  saved results/storm_obs.{png,npy}")


if __name__ == "__main__":
    main()
