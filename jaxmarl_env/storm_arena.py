"""CNN multi-agent RL trainer for the JaxMARL STORM "in-the-matrix" arena
(storm_np / class InTheMatrix).

STORM is a general-sum, N-agent gridworld where agents move on a 2D map, pick up
red ("cooperate") / blue ("defect") coins that fill an inventory, and resolve a
matrix game (default a Prisoner's Dilemma) when they interact. The observation
is a per-agent egocentric (5, 5, 14) image, so the policy uses a small CNN with
parameters shared across all agents (parameter sharing).

This file adapts the CNN loops from `cnn_algos.py` to storm_np's NONSTANDARD
array API:
  * `obs, state = env.reset(key)`  ->  obs is a plain array (N, 5, 5, 14).
  * `env.step(key, state, actions)` wants `actions` as jnp.array shape (N,).
  * `reward` is an array shape (N,); `done` is a dict with '__all__'.
We vmap reset/step over NUM_ENVS and keep every quantity in env-major flat
order (index = env * N + agent), reshaping to (NUM_ENVS, N) only at the env
boundary.

Algorithms (all share the CNN encoder):
  * IPPO : clipped PPO, UPDATE_EPOCHS minibatch passes (the workhorse).
  * A2C  : the same actor-critic loop with a single epoch and no clipping.
  * IQL  : independent CNN Q-learning, epsilon-greedy, target network.

Observability regimes are channel masks applied to the (5,5,14) obs BEFORE the
CNN (see REGIMES). Channels: 0-3 items [wall, beam, red_coin, blue_coin];
4 self-present; 5 other-agent-present; 6-9 orientation; 10 can-interact;
11-12 shown inventory [red, blue]; 13 frozen.

Experiments (each writes <name>.png @ dpi=300 + <name>.npy into results/):
  nagents       : IPPO, N in {2,4,8}, 3M steps   -> storm_nagents
  obs           : IPPO, N=2, full vs blind_others -> storm_obs
  obs_combos    : IPPO, N=2, 4 mask regimes       -> storm_obs_combos
  algos         : N=2 full, IPPO/A2C/IQL          -> storm_algos
  nagents_long  : IPPO, N in {2,4,8}, 8M steps     -> storm_nagents_long

Usage:
    python storm_arena.py                                  # all experiments
    python storm_arena.py --experiments obs_combos,algos   # a subset
    python storm_arena.py --debug                          # quick smoke test
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

IMG_SHAPE = (5, 5, 14)
ACT_DIM = 8

# Observability regimes: channels to ZERO in the obs before the CNN.
REGIMES = {
    "full":         (),            # no masking
    "blind_others": (5,),          # cannot see the other agent's cell
    "self_only":    (5, 11, 12),   # no other agent and no shown inventories
    "blind":        (4, 5),        # cannot localize self OR others
}


# ------------------------------------------------------------------- networks
class CNNAC(nn.Module):
    """Shared CNN actor-critic for (5,5,14) egocentric STORM observations."""
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


class CNNQ(nn.Module):
    """Shared CNN Q-network (same trunk as CNNAC)."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Conv(32, (3, 3), padding="SAME",
                            kernel_init=orthogonal(np.sqrt(2)))(x))
        h = nn.relu(nn.Conv(64, (3, 3), padding="SAME",
                            kernel_init=orthogonal(np.sqrt(2)))(h))
        h = h.reshape(h.shape[0], -1)
        h = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(h))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)


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
               NUM_INNER_STEPS=152, SEED=0,
               EPS_START=1.0, EPS_END=0.05, TARGET_PERIOD=10)


# ----------------------------------------------------------------- helpers
def _make_env(N, config):
    return jaxmarl.make("storm_np", num_agents=N,
                        num_inner_steps=config["NUM_INNER_STEPS"])


def _make_batchers(N, NE, mask_channels):
    n_actors = N * NE

    def batch(ob):                 # (NE, N, 5,5,14) -> (NE*N, 5,5,14)
        x = ob.reshape((n_actors,) + IMG_SHAPE).astype(jnp.float32)
        for ch in mask_channels:
            x = x.at[..., ch].set(0.0)
        return x

    def fmt_act(act):              # (NE*N,) -> (NE, N) array the env wants
        return act.reshape(NE, N)

    def rew_b(rew):                # (NE, N) -> (NE*N,)
        return rew.reshape(n_actors)

    def done_b(done):              # __all__ (NE,) -> (NE*N,) env-major
        d = done["__all__"].reshape(NE, 1)
        return jnp.broadcast_to(d, (NE, N)).reshape(n_actors)

    return n_actors, batch, fmt_act, rew_b, done_b


# ----------------------------------------------------- actor-critic (IPPO/A2C)
def make_train_pg(N, config, mask_channels=(), clip=True):
    NE = config["NUM_ENVS"]
    n_actors, batch, fmt_act, rew_b, done_b = _make_batchers(N, NE, mask_channels)
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * NE)
    epochs = config["UPDATE_EPOCHS"] if clip else 1
    env = _make_env(N, config)

    def train(rng):
        net = CNNAC(ACT_DIM)
        rng, r = jax.random.split(rng)
        params = net.init(r, jnp.zeros((1,) + IMG_SHAPE))
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
                        if clip:
                            ratio = jnp.exp(lp - traj.log_prob)
                            al = -jnp.minimum(
                                ratio * an,
                                jnp.clip(ratio, 1 - config["CLIP_EPS"],
                                         1 + config["CLIP_EPS"]) * an).mean()
                        else:
                            al = -(lp * an).mean()
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

            s, _ = jax.lax.scan(epoch, (ts, traj, adv, tgt, rng), None, epochs)
            return (s[0], st, last_ob, s[-1]), rm.mean()

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, config["NUM_UPDATES"])
        return dyn

    return train


# ------------------------------------------------------------------- IQL
def make_train_iql(N, config, mask_channels=()):
    NE = config["NUM_ENVS"]
    n_actors, batch, fmt_act, rew_b, done_b = _make_batchers(N, NE, mask_channels)
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * NE)
    env = _make_env(N, config)

    def train(rng):
        net = CNNQ(ACT_DIM)
        rng, r = jax.random.split(rng)
        params = net.init(r, jnp.zeros((1,) + IMG_SHAPE))
        ts = TrainState.create(
            apply_fn=net.apply, params=params,
            tx=optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                           optax.adam(config["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(carry, t):
            ts, tgt_p, st, ob, rng = carry
            eps = config["EPS_END"] + (config["EPS_START"] - config["EPS_END"]) \
                * (1 - t / config["NUM_UPDATES"])

            def step(run, _):
                ts, st, ob, rng = run
                q = net.apply(ts.params, batch(ob))
                rng, r1, r2 = jax.random.split(rng, 3)
                act = jnp.where(
                    jax.random.uniform(r2, (q.shape[0],)) < eps,
                    jax.random.randint(r1, (q.shape[0],), 0, ACT_DIM),
                    jnp.argmax(q, -1))
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(
                    jax.random.split(r, NE), st, fmt_act(act))
                rb = rew_b(rew)
                return (ts, st, ob2, rng), \
                    ((batch(ob), act, rb, batch(ob2), done_b(dn)), rb.mean())

            (ts, st, ob, rng), (tr, rm) = jax.lax.scan(
                step, (ts, st, ob, rng), None, config["NUM_STEPS"])
            obs_b, act_b, r_b, nobs_b, done_b_ = [
                x.reshape((-1,) + x.shape[2:]) for x in tr]

            def loss(p):
                qa = jnp.take_along_axis(net.apply(p, obs_b),
                                         act_b[:, None], 1)[:, 0]
                nq = net.apply(tgt_p, nobs_b).max(-1)
                target = r_b + config["GAMMA"] * nq * (1 - done_b_)
                return ((qa - jax.lax.stop_gradient(target)) ** 2).mean()

            ts = ts.apply_gradients(grads=jax.grad(loss)(ts.params))
            tgt_p = jax.lax.cond((t % config["TARGET_PERIOD"]) == 0,
                                 lambda: ts.params, lambda: tgt_p)
            return (ts, tgt_p, st, ob, rng), rm.mean()

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, params, st, obsv, r),
                              jnp.arange(config["NUM_UPDATES"]))
        return dyn

    return train


# ------------------------------------------------------------------- runner
def run_one(N, config, algo="IPPO", mask_channels=()):
    if algo == "IPPO":
        fn = make_train_pg(N, config, mask_channels, clip=True)
    elif algo == "A2C":
        fn = make_train_pg(N, config, mask_channels, clip=False)
    elif algo == "IQL":
        fn = make_train_iql(N, config, mask_channels)
    else:
        raise ValueError(algo)
    dyn = np.asarray(jax.jit(fn)(jax.random.PRNGKey(config["SEED"])))
    return dyn


def steps_axis(config):
    n_up = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * config["NUM_ENVS"])
    return (np.arange(n_up) + 1) * config["NUM_STEPS"] * config["NUM_ENVS"]


# ----------------------------------------------------------------- plotting
def _smooth(y, k=9):
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def _plot_lines(named_curves, x_by_key, title, path, colors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, key in enumerate(named_curves):
        y = np.asarray(named_curves[key])
        x = x_by_key[key] if isinstance(x_by_key, dict) else x_by_key
        c = (colors or {}).get(key)
        ax.plot(x, y, color=c, alpha=0.22)
        ax.plot(x, _smooth(y), color=c, label=str(key))
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean reward / agent / step")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------- experiments
N_COLORS = {2: "#0254a3", 4: "#d1691f", 8: "#2a8f3c"}


def exp_nagents(cfg, timesteps, tag="storm_nagents"):
    print(f"=== {tag}: number of agents (timesteps={timesteps}) ===")
    c = dict(cfg, TOTAL_TIMESTEPS=timesteps)
    curves, x_by_n = {}, {}
    for n in (2, 4, 8):
        ci = dict(c)
        dyn = run_one(n, ci, algo="IPPO")
        curves[n] = dyn
        x_by_n[n] = steps_axis(ci)
        print(f"  N={n}: {dyn[0]:+.4f} -> {dyn[-1]:+.4f} "
              f"(final-10 {dyn[-10:].mean():+.4f})")
    np.save(os.path.join(RESULTS, f"{tag}.npy"),
            {"curves": curves, "x": x_by_n}, allow_pickle=True)
    _plot_lines({f"N = {n}": curves[n] for n in curves},
                {f"N = {n}": x_by_n[n] for n in curves},
                f"STORM in-the-matrix: IPPO vs. number of agents ({timesteps//10**6}M)",
                os.path.join(RESULTS, f"{tag}.png"),
                colors={f"N = {n}": N_COLORS[n] for n in curves})
    print(f"  saved results/{tag}.{{png,npy}}")
    return curves, x_by_n


def exp_obs(cfg):
    print("=== storm_obs: observability, N=2 (full vs blind_others) ===")
    x = steps_axis(cfg)
    curves = {}
    curves["full obs"] = run_one(2, dict(cfg), algo="IPPO",
                                 mask_channels=REGIMES["full"])
    curves["blind to others"] = run_one(2, dict(cfg), algo="IPPO",
                                        mask_channels=REGIMES["blind_others"])
    for k, v in curves.items():
        print(f"  {k:16s}: {v[-1]:+.4f} (final-10 {v[-10:].mean():+.4f})")
    np.save(os.path.join(RESULTS, "storm_obs.npy"),
            {"curves": curves, "x": x}, allow_pickle=True)
    _plot_lines(curves, x, "STORM in-the-matrix (N=2): full vs. blind-to-others",
                os.path.join(RESULTS, "storm_obs.png"),
                colors={"full obs": "#0254a3", "blind to others": "#b3243f"})
    print("  saved results/storm_obs.{png,npy}")


def exp_obs_combos(cfg):
    print("=== storm_obs_combos: observability regimes, N=2 ===")
    x = steps_axis(cfg)
    curves = {}
    cols = {"full": "#0254a3", "blind_others": "#d1691f",
            "self_only": "#2a8f3c", "blind": "#b3243f"}
    for name, chans in REGIMES.items():
        dyn = run_one(2, dict(cfg), algo="IPPO", mask_channels=chans)
        curves[name] = dyn
        print(f"  {name:13s} mask{str(chans):10s}: {dyn[0]:+.4f} -> {dyn[-1]:+.4f} "
              f"(final-10 {dyn[-10:].mean():+.4f})")
    np.save(os.path.join(RESULTS, "storm_obs_combos.npy"),
            {"curves": curves, "x": x, "regimes": REGIMES}, allow_pickle=True)
    _plot_lines(curves, x, "STORM in-the-matrix (N=2): observability regimes",
                os.path.join(RESULTS, "storm_obs_combos.png"), colors=cols)
    print("  saved results/storm_obs_combos.{png,npy}")


def exp_algos(cfg):
    print("=== storm_algos: IPPO vs A2C vs IQL, N=2 full-obs ===")
    x = steps_axis(cfg)
    curves = {}
    cols = {"IPPO": "#0254a3", "A2C": "#d1691f", "IQL": "#2a8f3c"}
    for algo in ("IPPO", "A2C", "IQL"):
        dyn = run_one(2, dict(cfg), algo=algo)
        curves[algo] = dyn
        print(f"  {algo:5s}: {dyn[0]:+.4f} -> {dyn[-1]:+.4f} "
              f"(final-10 {dyn[-10:].mean():+.4f})")
    np.save(os.path.join(RESULTS, "storm_algos.npy"),
            {"curves": curves, "x": x}, allow_pickle=True)
    _plot_lines(curves, x, "STORM in-the-matrix (N=2): IPPO vs A2C vs IQL",
                os.path.join(RESULTS, "storm_algos.png"), colors=cols)
    print("  saved results/storm_algos.{png,npy}")


EXPERIMENTS = {
    "nagents":      lambda cfg: exp_nagents(cfg, 3_000_000, "storm_nagents"),
    "obs":          exp_obs,
    "obs_combos":   exp_obs_combos,
    "algos":        exp_algos,
    "nagents_long": lambda cfg: exp_nagents(cfg, 8_000_000, "storm_nagents_long"),
}


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=3_000_000,
                   help="budget for the 3M-style experiments (nagents/obs/...)")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--experiments", type=str,
                   default="nagents,obs,obs_combos,algos,nagents_long",
                   help="comma list: " + ",".join(EXPERIMENTS))
    p.add_argument("--debug", action="store_true",
                   help="quick 100k smoke test of every algo + a mask")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    cfg = dict(DEFAULT, NUM_ENVS=a.num_envs, NUM_STEPS=a.num_steps,
               SEED=a.seed, TOTAL_TIMESTEPS=a.timesteps)

    if a.debug:
        c = dict(cfg, TOTAL_TIMESTEPS=100_000)
        nup = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * c["NUM_ENVS"])
        print(f"[debug] 100k steps ({nup} updates) per run")
        for algo in ("IPPO", "A2C", "IQL"):
            d = run_one(2, dict(c), algo=algo)
            print(f"[debug] {algo:5s} N=2 full : {d[0]:+.4f} -> {d[-1]:+.4f}")
        for name in ("blind_others", "self_only", "blind"):
            d = run_one(2, dict(c), algo="IPPO", mask_channels=REGIMES[name])
            print(f"[debug] IPPO  N=2 {name:12s}: {d[0]:+.4f} -> {d[-1]:+.4f}")
        d = run_one(8, dict(c), algo="IPPO")
        print(f"[debug] IPPO  N=8 full : {d[0]:+.4f} -> {d[-1]:+.4f}")
        print("[debug] OK -- all paths ran without error")
        return

    chosen = [e.strip() for e in a.experiments.split(",") if e.strip()]
    for name in chosen:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; "
                             f"choices: {list(EXPERIMENTS)}")
        fn = EXPERIMENTS[name]
        if name in ("nagents", "nagents_long"):
            fn(cfg)
        else:
            fn(dict(cfg))


if __name__ == "__main__":
    main()
