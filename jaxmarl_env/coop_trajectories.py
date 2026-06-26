"""
Cooperation phase-portraits (FlowPlot style) for the social-dilemma games.

For each observability regime of the heterogeneous-observation IPD (and for
coin_game) we train IPPO while logging each agent's realized cooperation
frequency P(play C) per update, then draw the joint policy's path through the
(agent-0 cooperation, agent-1 cooperation) plane -- the deep-RL analogue of
pyCRLD's `fp.plot_trajectories([xtraj], x, y, cols=["purple"], axes=ax)`.

Outputs:
    results/coop_<regime>.npy        trajectory, shape (num_updates, 2)
    results/coop_portraits.png       grid of phase portraits
"""
import os
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heterogeneous_ipd import make_hetero_ipd, REGIMES

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


# --- minimal shared-policy FF-IPPO that also logs per-agent cooperation -------
class Categorical:
    def __init__(self, logits):
        self.logits = logits
        self.log_p = jax.nn.log_softmax(logits)
    def sample(self, k): return jax.random.categorical(k, self.logits)
    def log_prob(self, a): return jnp.take_along_axis(self.log_p, a[..., None], -1)[..., 0]
    def entropy(self): return -jnp.sum(jnp.exp(self.log_p) * self.log_p, -1)


class AC(nn.Module):
    action_dim: int
    hidden: int = 64
    @nn.compact
    def __call__(self, x):
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(a))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(a)
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(v))
        return logits, jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0))(v), -1)


class T(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray


CFG = dict(LR=2.5e-4, NUM_ENVS=64, NUM_STEPS=128, TOTAL_TIMESTEPS=3_000_000,
           UPDATE_EPOCHS=4, NUM_MINIBATCHES=4, GAMMA=0.99, GAE_LAMBDA=0.95,
           CLIP_EPS=0.2, ENT_COEF=0.01, VF_COEF=0.5, MAX_GRAD_NORM=0.5, HIDDEN=64)


def coop_trajectory(regime, num_agents=2, seed=0, cfg=CFG,
                    payoffs=(1.0, 1.2, -0.5, 0.0)):
    """Train and return a (num_updates, num_agents) array of cooperation freq."""
    c = dict(cfg)
    env = make_hetero_ipd(num_agents=num_agents, regime=regime,
                          num_steps=c["NUM_STEPS"], payoffs=payoffs,
                          memory=c.get("MEMORY", 1))
    agents = env.agents; nA = env.num_agents; NE = c["NUM_ENVS"]
    n_actors = nA * NE
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    obs_dim = env.observation_space(agents[0]).shape[0]

    def batch(d): return jnp.stack([d[a] for a in agents]).reshape(n_actors, -1)
    def rew_b(d): return jnp.stack([d[a] for a in agents]).reshape(n_actors)
    def unb(x): x = x.reshape(nA, NE); return {a: x[i] for i, a in enumerate(agents)}

    @jax.jit
    def train(rng):
        net = AC(2, c["HIDDEN"])
        rng, r = jax.random.split(rng)
        params = net.init(r, jnp.zeros((1, obs_dim)))
        tx = optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                         optax.adam(c["LR"], eps=1e-5))
        ts = TrainState.create(apply_fn=net.apply, params=params, tx=tx)
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(run, _):
            def step(run, _):
                ts, st, ob, rng = run
                ob_b = batch(ob)
                logits, val = net.apply(ts.params, ob_b)
                pi = Categorical(logits)
                rng, r = jax.random.split(rng)
                act = pi.sample(r)
                env_act = unb(act)
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(
                    jax.random.split(r, NE), st, env_act)
                tr = T(rew_b(dn), act, val, rew_b(rew), pi.log_prob(act), ob_b)
                # per-agent cooperation (action 0) frequency this step
                coop = (act.reshape(nA, NE) == 0).mean(axis=1)   # (nA,)
                return (ts, st, ob2, rng), (tr, coop)
            run, (traj, coop) = jax.lax.scan(step, run, None, c["NUM_STEPS"])
            ts, st, last_ob, rng = run
            _, last_val = net.apply(ts.params, batch(last_ob))

            def gae(traj, lv):
                def f(carry, t):
                    g, nv = carry
                    d = t.reward + c["GAMMA"] * nv * (1 - t.done) - t.value
                    g = d + c["GAMMA"] * c["GAE_LAMBDA"] * (1 - t.done) * g
                    return (g, t.value), g
                _, adv = jax.lax.scan(f, (jnp.zeros_like(lv), lv), traj,
                                      reverse=True, unroll=16)
                return adv, adv + traj.value
            adv, tgt = gae(traj, last_val)

            def epoch(s, _):
                def mb(ts, b):
                    tr, adv, tgt = b
                    def loss(p):
                        logits, val = net.apply(p, tr.obs)
                        pi = Categorical(logits)
                        lp = pi.log_prob(tr.action)
                        vcl = tr.value + (val - tr.value).clip(-c["CLIP_EPS"], c["CLIP_EPS"])
                        vl = 0.5 * jnp.maximum((val - tgt) ** 2, (vcl - tgt) ** 2).mean()
                        ratio = jnp.exp(lp - tr.log_prob)
                        an = (adv - adv.mean()) / (adv.std() + 1e-8)
                        al = -jnp.minimum(ratio * an, jnp.clip(
                            ratio, 1 - c["CLIP_EPS"], 1 + c["CLIP_EPS"]) * an).mean()
                        return al + c["VF_COEF"] * vl - c["ENT_COEF"] * pi.entropy().mean()
                    return ts.apply_gradients(grads=jax.grad(loss)(ts.params)), None
                ts, tr, adv, tgt, rng = s
                rng, r = jax.random.split(rng)
                B = c["NUM_STEPS"] * n_actors
                b = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]),
                                           (tr, adv, tgt))
                perm = jax.random.permutation(r, B)
                b = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, 0), b)
                mbs = jax.tree_util.tree_map(
                    lambda x: x.reshape((c["NUM_MINIBATCHES"], -1) + x.shape[1:]), b)
                ts, _ = jax.lax.scan(mb, ts, mbs)
                return (ts, tr, adv, tgt, rng), None
            s = (ts, traj, adv, tgt, rng)
            s, _ = jax.lax.scan(epoch, s, None, c["UPDATE_EPOCHS"])
            return (s[0], st, last_ob, s[-1]), coop.mean(axis=0)   # mean over steps -> (nA,)

        rng, r = jax.random.split(rng)
        run = (ts, st, obsv, r)
        run, coop_traj = jax.lax.scan(upd, run, None, c["NUM_UPDATES"])
        return coop_traj   # (num_updates, nA)

    return np.asarray(train(jax.random.PRNGKey(seed)))


def flowplot_trajectory(traj, ax, col="purple", label=None):
    """Replicates pyCRLD fp.plot_trajectories style for a 2-agent coop path."""
    xs, ys = traj[:, 0], traj[:, 1]
    ax.plot(xs, ys, lw=2, color=col, alpha=0.9)
    ax.scatter(xs[0], ys[0], color=col, marker="x", s=40)     # start
    ax.scatter(xs[-1], ys[-1], color=col, marker="o", s=40)   # end (fixed point)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=0.7)
    ax.set_xlabel("Agent 1  P(cooperate)"); ax.set_ylabel("Agent 2  P(cooperate)")
    if label: ax.set_title(label, fontsize=10)


if __name__ == "__main__":
    regimes = list(REGIMES)
    trajs = {}
    for r in regimes:
        t = coop_trajectory(r)
        np.save(os.path.join(RESULTS, f"coop_{r}.npy"), t)
        trajs[r] = t
        print(f"[{r}] coop start ({t[0,0]:.2f},{t[0,1]:.2f}) -> "
              f"end ({t[-1,0]:.2f},{t[-1,1]:.2f})")
    ncol = 3; nrow = (len(regimes) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4*ncol, 3.2*nrow), squeeze=False)
    for ax in axes.flat: ax.axis("off")
    names = {"full": "Full observability", "blind": "Blind",
             "self": "Self-observation only", "others": "Others only",
             "coop": "Cooperation-tracking", "def": "Defection-tracking"}
    for k, r in enumerate(regimes):
        ax = axes[k//ncol][k%ncol]; ax.axis("on")
        flowplot_trajectory(trajs[r], ax, label=names.get(r, r))
    fig.suptitle("Cooperation dynamics (IPPO) across observability regimes — "
                 "memory-1 IPD, N=2", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(RESULTS, "coop_portraits.png")
    fig.savefig(out, dpi=150); print("saved", out)
