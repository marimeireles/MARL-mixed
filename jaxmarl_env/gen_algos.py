"""Run IPPO / A2C / IQL on ANY discrete-action JaxMARL game (general-sum included),
logging mean reward per update. Generalises the IPD-only trainers in
algorithms.py so coin_game and friends can be compared across algorithms.

Usage:  python gen_algos.py <env_name> [--timesteps N]
"""
import argparse, os
from functools import partial
from typing import NamedTuple

import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
import optax
import jaxmarl

from run_all import introspect, ActorCritic, Categorical, SkipEnv, DEFAULT

RESULTS = os.path.join(os.path.dirname(__file__), "results")


class QNet(nn.Module):
    action_dim: int
    hidden: int = 64
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        h = nn.relu(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(h))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)


def _env_helpers(env, meta, NE):
    agents, obs_dim, act_dim, act_mask, act_dims, kind = meta
    nA = len(agents)
    def flat_pad(x):
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] < obs_dim: x = jnp.pad(x, ((0, 0), (0, obs_dim - x.shape[1])))
        elif x.shape[1] > obs_dim: x = x[:, :obs_dim]
        return x
    batch = lambda d: jnp.concatenate([flat_pad(d[a]) for a in agents], 0)
    rew_b = lambda d: jnp.concatenate([d[a].reshape(-1) for a in agents], 0)
    unb = lambda x: {a: x.reshape(nA, NE)[i] for i, a in enumerate(agents)}
    full_mask = jnp.repeat(act_mask, NE, axis=0)
    return agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb


# --------------------------------------------------------------------------- PG
class Tr(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray; mask: jnp.ndarray


def _train_pg(env, meta, c, clip):       # clip=True -> IPPO, clip=False -> A2C
    NE = c["NUM_ENVS"]
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
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                return (ts, st, ob2, rng), (Tr(d, act, val, rew_b(rew), pi.log_prob(act),
                                              batch(ob), full_mask), rew_b(rew).mean())
            run, (traj, rm) = jax.lax.scan(step, run, None, c["NUM_STEPS"])
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
                        pi = Categorical(logits, mask=traj.mask)
                        lp = pi.log_prob(traj.action)
                        an = (adv - adv.mean()) / (adv.std() + 1e-8)
                        if clip:
                            ratio = jnp.exp(lp - traj.log_prob)
                            al = -jnp.minimum(ratio * an, jnp.clip(
                                ratio, 1 - c["CLIP_EPS"], 1 + c["CLIP_EPS"]) * an).mean()
                        else:
                            al = -(lp * an).mean()
                        vl = 0.5 * ((val - tgt) ** 2).mean()
                        return al + c["VF_COEF"] * vl - c["ENT_COEF"] * pi.entropy().mean()
                    return ts.apply_gradients(grads=jax.grad(loss)(ts.params)), None
                ts, traj, adv, tgt, rng = s
                rng, r = jax.random.split(rng)
                B = c["NUM_STEPS"] * n_actors
                b = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), (traj, adv, tgt))
                perm = jax.random.permutation(r, B)
                b = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, 0), b)
                mbs = jax.tree_util.tree_map(
                    lambda x: x.reshape((c["NUM_MINIBATCHES"], -1) + x.shape[1:]), b)
                ts, _ = jax.lax.scan(mb, ts, mbs)
                return (ts, traj, adv, tgt, rng), None
            s, _ = jax.lax.scan(epoch, (ts, traj, adv, tgt, rng), None, epochs)
            return (s[0], st, last_ob, s[-1]), rm.mean()

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, c["NUM_UPDATES"])
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


def _train_iql(env, meta, c, eps_start=1.0, eps_end=0.05, target_period=200):
    NE = c["NUM_ENVS"]
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
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                return (ts, st, ob2, rng), ((batch(ob), act, rew_b(rew), batch(ob2), d),
                                            rew_b(rew).mean())
            (ts, st, ob, rng), (tr, rm) = jax.lax.scan(step, (ts, st, ob, rng), None, c["NUM_STEPS"])
            obs_b, act_b, r_b, nobs_b, done_b = [x.reshape((-1,) + x.shape[2:]) for x in tr]
            negB = jnp.tile(neg, (c["NUM_STEPS"], 1))     # mask aligned to flattened batch
            def loss(p):
                qa = jnp.take_along_axis(net.apply(p, obs_b) + negB, act_b[:, None], 1)[:, 0]
                nq = (net.apply(tgt_p, nobs_b) + negB).max(-1)
                target = r_b + c["GAMMA"] * nq * (1 - done_b)
                return ((qa - jax.lax.stop_gradient(target)) ** 2).mean()
            ts = ts.apply_gradients(grads=jax.grad(loss)(ts.params))
            tgt_p = jax.lax.cond((t % target_period) == 0, lambda: ts.params, lambda: tgt_p)
            return (ts, tgt_p, st, ob, rng), rm.mean()
        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, params, st, obsv, r), jnp.arange(c["NUM_UPDATES"]))
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


def train(env_name, algo, cfg):
    env = jaxmarl.make(env_name)
    meta = introspect(env)             # raises SkipEnv for continuous / huge-obs
    if meta[5] != "discrete":
        raise SkipEnv("A2C/IQL comparison here is discrete-only")
    if algo == "IPPO": return _train_pg(env, meta, dict(cfg), clip=True)
    if algo == "A2C":  return _train_pg(env, meta, dict(cfg), clip=False)
    if algo == "IQL":  return _train_iql(env, meta, dict(cfg))
    raise ValueError(algo)


ALGOS = ["IPPO", "A2C", "IQL"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("env"); p.add_argument("--timesteps", type=int, default=3_000_000)
    a = p.parse_args()
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=a.timesteps)
    for algo in ALGOS:
        try:
            d = train(a.env, algo, cfg)
            print(f"{algo:5s} [{a.env}] reward {d[0]:+.3f} -> {d[-1]:+.3f}")
        except SkipEnv as e:
            print(f"{algo:5s} [{a.env}] skipped: {e}")
