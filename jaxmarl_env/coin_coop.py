"""Cooperation dynamics on coin_game for IPPO / A2C / IQL.

In coin_game, picking up your OWN colour coin is cooperation; taking the other
agent's coin (+1 to you, -2 to them) is defection. We recompute that event from
(state, actions) each step (info is empty) and log the cooperation fraction
over training -- the coin_game analogue of P(cooperate) in the IPD.
"""
import os
from typing import NamedTuple
import numpy as np
import jax, jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import jaxmarl
from jaxmarl.environments.coin_game.coin_game import MOVES

from run_all import introspect, ActorCritic, Categorical
from gen_algos import QNet, _env_helpers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "results")
C = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}


def _coop_step(st, act, NE):
    """cooperation & total pickups this step (summed over the NE parallel envs)."""
    a0, a1 = act[:NE], act[NE:2 * NE]                  # agent-major batchify order
    new_red = (st.red_pos + MOVES[a0]) % 3
    new_blue = (st.blue_pos + MOVES[a1]) % 3
    rr = jnp.all(new_red == st.red_coin_pos, -1)
    bb = jnp.all(new_blue == st.blue_coin_pos, -1)
    rb = jnp.all(new_red == st.blue_coin_pos, -1)
    br = jnp.all(new_blue == st.red_coin_pos, -1)
    coop = (rr + bb).sum()
    total = (rr + bb + rb + br).sum()
    return coop, total


class Tr(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray


def _pg(c, clip):
    NE = c["NUM_ENVS"]
    env = jaxmarl.make("coin_game"); meta = introspect(env)
    agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb = _env_helpers(env, meta, NE)
    n_actors = nA * NE
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    epochs = c["UPDATE_EPOCHS"] if clip else 1

    @jax.jit
    def train(rng):
        net = ActorCritic(act_dim, c["HIDDEN"])
        rng, r = jax.random.split(rng)
        ts = TrainState.create(apply_fn=net.apply, params=net.init(r, jnp.zeros((1, obs_dim))),
                               tx=optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                                              optax.adam(c["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(run, _):
            def step(run, _):
                ts, st, ob, rng = run
                (logits,), val = net.apply(ts.params, batch(ob))
                pi = Categorical(logits, mask=full_mask)
                rng, r = jax.random.split(rng); act = pi.sample(r)
                coop, total = _coop_step(st, act, NE)
                rng, r = jax.random.split(rng)
                ob2, st2, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                return (ts, st2, ob2, rng), (Tr(d, act, val, rew_b(rew), pi.log_prob(act),
                                            batch(ob)), coop, total)
            run, (traj, coop, total) = jax.lax.scan(step, run, None, c["NUM_STEPS"])
            ts, st, last_ob, rng = run
            (_,), last_val = net.apply(ts.params, batch(last_ob))

            def gae(traj, lv):
                def f(carry, t):
                    g, nv = carry
                    d = t.reward + c["GAMMA"] * nv * (1 - t.done) - t.value
                    g = d + c["GAMMA"] * c["GAE_LAMBDA"] * (1 - t.done) * g
                    return (g, t.value), g
                _, adv = jax.lax.scan(f, (jnp.zeros_like(lv), lv), traj, reverse=True, unroll=16)
                return adv, adv + traj.value
            adv, tgt = gae(traj, last_val)

            def epoch(s, _):
                def mb(ts, b):
                    traj, adv, tgt = b
                    def loss(p):
                        (logits,), val = net.apply(p, traj.obs)
                        pi = Categorical(logits)          # coin_game: all 5 actions valid
                        lp = pi.log_prob(traj.action)
                        an = (adv - adv.mean()) / (adv.std() + 1e-8)
                        if clip:
                            ratio = jnp.exp(lp - traj.log_prob)
                            al = -jnp.minimum(ratio * an, jnp.clip(
                                ratio, 1 - c["CLIP_EPS"], 1 + c["CLIP_EPS"]) * an).mean()
                        else:
                            al = -(lp * an).mean()
                        return al + c["VF_COEF"] * 0.5 * ((val - tgt) ** 2).mean() \
                            - c["ENT_COEF"] * pi.entropy().mean()
                    return ts.apply_gradients(grads=jax.grad(loss)(ts.params)), None
                ts, traj, adv, tgt, rng = s
                rng, r = jax.random.split(rng)
                B = c["NUM_STEPS"] * n_actors
                fl = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), (traj, adv, tgt))
                perm = jax.random.permutation(r, B)
                fl = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, 0), fl)
                mbs = jax.tree_util.tree_map(
                    lambda x: x.reshape((c["NUM_MINIBATCHES"], -1) + x.shape[1:]), fl)
                ts, _ = jax.lax.scan(mb, ts, mbs)
                return (ts, traj, adv, tgt, rng), None
            s, _ = jax.lax.scan(epoch, (ts, traj, adv, tgt, rng), None, epochs)
            coop_frac = coop.sum() / jnp.maximum(total.sum(), 1.0)
            return (s[0], st, last_ob, s[-1]), coop_frac

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, c["NUM_UPDATES"])
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


def _iql(c, eps_start=1.0, eps_end=0.05, target_period=200):
    NE = c["NUM_ENVS"]
    env = jaxmarl.make("coin_game"); meta = introspect(env)
    agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb = _env_helpers(env, meta, NE)
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    neg = jnp.where(full_mask, 0.0, -1e9)

    @jax.jit
    def train(rng):
        net = QNet(act_dim, c["HIDDEN"])
        rng, r = jax.random.split(rng); params = net.init(r, jnp.zeros((1, obs_dim)))
        ts = TrainState.create(apply_fn=net.apply, params=params,
                               tx=optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                                              optax.adam(c["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(carry, t):
            ts, tgt_p, st, ob, rng = carry
            eps = eps_end + (eps_start - eps_end) * (1 - t / c["NUM_UPDATES"])
            def step(run, _):
                ts, st, ob, rng = run
                q = net.apply(ts.params, batch(ob)) + neg
                rng, r1, r2 = jax.random.split(rng, 3)
                act = jnp.where(jax.random.uniform(r2, (q.shape[0],)) < eps,
                                jax.random.randint(r1, (q.shape[0],), 0, act_dim), jnp.argmax(q, -1))
                coop, total = _coop_step(st, act, NE)
                rng, r = jax.random.split(rng)
                ob2, st2, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                return (ts, st2, ob2, rng), ((batch(ob), act, rew_b(rew), batch(ob2), d), coop, total)
            (ts, st, ob, rng), (tr, coop, total) = jax.lax.scan(step, (ts, st, ob, rng), None, c["NUM_STEPS"])
            obs_b, act_b, r_b, nobs_b, done_b = [x.reshape((-1,) + x.shape[2:]) for x in tr]
            negB = jnp.tile(neg, (c["NUM_STEPS"], 1))
            def loss(p):
                qa = jnp.take_along_axis(net.apply(p, obs_b) + negB, act_b[:, None], 1)[:, 0]
                nq = (net.apply(tgt_p, nobs_b) + negB).max(-1)
                target = r_b + c["GAMMA"] * nq * (1 - done_b)
                return ((qa - jax.lax.stop_gradient(target)) ** 2).mean()
            ts = ts.apply_gradients(grads=jax.grad(loss)(ts.params))
            tgt_p = jax.lax.cond((t % target_period) == 0, lambda: ts.params, lambda: tgt_p)
            return (ts, tgt_p, st, ob, rng), coop.sum() / jnp.maximum(total.sum(), 1.0)
        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, params, st, obsv, r), jnp.arange(c["NUM_UPDATES"]))
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


def train(algo, cfg):
    c = dict(cfg)
    if algo == "IPPO": return _pg(c, clip=True)
    if algo == "A2C":  return _pg(c, clip=False)
    if algo == "IQL":  return _iql(c)


if __name__ == "__main__":
    from run_all import DEFAULT
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=2_000_000, NUM_ENVS=64)
    curves = {}
    for algo in ["IPPO", "A2C", "IQL"]:
        d = train(algo, cfg); curves[algo] = d
        print(f"  {algo:5s} coin_game coop {d[0]:.3f} -> {d[-1]:.3f}", flush=True)
    np.save(os.path.join(RESULTS, "coin_coop.npy"), np.array([curves[a] for a in curves]))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for algo in curves:
        y = curves[algo]
        ax.plot(np.arange(len(y)) * cfg["NUM_STEPS"] * cfg["NUM_ENVS"], y, color=C[algo], label=algo, lw=2)
    ax.set_xlabel("environment steps"); ax.set_ylabel("cooperation fraction (own-coin pickups)")
    ax.set_ylim(-0.03, 1.03); ax.set_title("Cooperation dynamics on coin_game"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "coin_coop.png"), dpi=300)
    print("saved coin_coop.png")
