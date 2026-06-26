"""CNN versions of IPPO / A2C / IQL for image-observation JaxMARL games
(Overcooked, and other grid-observation envs). Same training loops as
gen_algos.py, but the observation keeps its (H, W, C) spatial structure and is
encoded by a small conv net instead of being flattened.

Usage:  python cnn_algos.py overcooked [--timesteps N]
"""
import argparse, os
from typing import NamedTuple

import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
import optax
import jaxmarl

from run_all import DEFAULT
RESULTS = os.path.join(os.path.dirname(__file__), "results")


class CNNAC(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):                          # x: (B, H, W, C) float
        h = nn.relu(nn.Conv(32, (3, 3), padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x))
        h = nn.relu(nn.Conv(64, (3, 3), padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(h))
        h = h.reshape(h.shape[0], -1)
        h = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(h))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)
        v = nn.Dense(1, kernel_init=orthogonal(1.0))(h)
        return logits, jnp.squeeze(v, -1)


class CNNQ(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Conv(32, (3, 3), padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(x))
        h = nn.relu(nn.Conv(64, (3, 3), padding="SAME", kernel_init=orthogonal(np.sqrt(2)))(h))
        h = h.reshape(h.shape[0], -1)
        h = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)))(h))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)


class Categorical:
    def __init__(self, logits):
        self.logits = logits; self.log_p = jax.nn.log_softmax(logits)
    def sample(self, k): return jax.random.categorical(k, self.logits)
    def log_prob(self, a): return jnp.take_along_axis(self.log_p, a[..., None], -1)[..., 0]
    def entropy(self): return -jnp.sum(jnp.exp(self.log_p) * self.log_p, -1)


class Tr(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray


def _setup(env_name, NE):
    env = jaxmarl.make(env_name)
    agents = env.agents; nA = env.num_agents
    obs0, _ = env.reset(jax.random.PRNGKey(0))
    img_shape = np.asarray(obs0[agents[0]]).shape          # (H, W, C)
    act_dim = env.action_space(agents[0]).n
    n_actors = nA * NE
    def batch(d): return jnp.stack([d[a] for a in agents]).reshape(
        (n_actors,) + img_shape).astype(jnp.float32)
    def rew_b(d): return jnp.stack([d[a] for a in agents]).reshape(n_actors)
    def unb(x): x = x.reshape(nA, NE); return {a: x[i] for i, a in enumerate(agents)}
    return env, agents, nA, act_dim, img_shape, n_actors, batch, rew_b, unb


def _shaped(info, agents, n_actors, coef):
    """Per-actor shaped reward (0 if the env exposes none / coef is 0)."""
    if coef <= 0 or "shaped_reward" not in info:
        return 0.0
    return coef * jnp.stack([info["shaped_reward"][a] for a in agents]).reshape(n_actors)


def _pg(env_name, c, clip):
    NE = c["NUM_ENVS"]
    env, agents, nA, act_dim, img_shape, n_actors, batch, rew_b, unb = _setup(env_name, NE)
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    epochs = c["UPDATE_EPOCHS"] if clip else 1
    sc = c.get("SHAPED_COEF", 0.0)

    @jax.jit
    def train(rng):
        net = CNNAC(act_dim)
        rng, r = jax.random.split(rng)
        params = net.init(r, jnp.zeros((1,) + img_shape))
        ts = TrainState.create(apply_fn=net.apply, params=params,
                               tx=optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                                              optax.adam(c["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(run, _):
            def step(run, _):
                ts, st, ob, rng = run
                logits, val = net.apply(ts.params, batch(ob))
                pi = Categorical(logits)
                rng, r = jax.random.split(rng); act = pi.sample(r)
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                sparse = rew_b(rew)
                train_rew = sparse + _shaped(info, agents, n_actors, sc)   # train on shaped
                return (ts, st, ob2, rng), (Tr(d, act, val, train_rew, pi.log_prob(act),
                                              batch(ob)), sparse.mean())   # log sparse
            run, (traj, rm) = jax.lax.scan(step, run, None, c["NUM_STEPS"])
            ts, st, last_ob, rng = run
            _, last_val = net.apply(ts.params, batch(last_ob))

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
                        logits, val = net.apply(p, traj.obs)
                        pi = Categorical(logits); lp = pi.log_prob(traj.action)
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
                flat = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), (traj, adv, tgt))
                perm = jax.random.permutation(r, B)
                flat = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, 0), flat)
                mbs = jax.tree_util.tree_map(
                    lambda x: x.reshape((c["NUM_MINIBATCHES"], -1) + x.shape[1:]), flat)
                ts, _ = jax.lax.scan(mb, ts, mbs)
                return (ts, traj, adv, tgt, rng), None
            s, _ = jax.lax.scan(epoch, (ts, traj, adv, tgt, rng), None, epochs)
            return (s[0], st, last_ob, s[-1]), rm.mean()

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, c["NUM_UPDATES"])
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


def _iql(env_name, c, eps_start=1.0, eps_end=0.05, target_period=200):
    NE = c["NUM_ENVS"]
    env, agents, nA, act_dim, img_shape, n_actors, batch, rew_b, unb = _setup(env_name, NE)
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)

    @jax.jit
    def train(rng):
        net = CNNQ(act_dim)
        rng, r = jax.random.split(rng); params = net.init(r, jnp.zeros((1,) + img_shape))
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
                q = net.apply(ts.params, batch(ob))
                rng, r1, r2 = jax.random.split(rng, 3)
                act = jnp.where(jax.random.uniform(r2, (q.shape[0],)) < eps,
                                jax.random.randint(r1, (q.shape[0],), 0, act_dim), jnp.argmax(q, -1))
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                return (ts, st, ob2, rng), ((batch(ob), act, rew_b(rew), batch(ob2), d), rew_b(rew).mean())
            (ts, st, ob, rng), (tr, rm) = jax.lax.scan(step, (ts, st, ob, rng), None, c["NUM_STEPS"])
            obs_b, act_b, r_b, nobs_b, done_b = [x.reshape((-1,) + x.shape[2:]) for x in tr]
            def loss(p):
                qa = jnp.take_along_axis(net.apply(p, obs_b), act_b[:, None], 1)[:, 0]
                nq = net.apply(tgt_p, nobs_b).max(-1)
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
    c = dict(cfg)
    if algo == "IPPO": return _pg(env_name, c, clip=True)
    if algo == "A2C":  return _pg(env_name, c, clip=False)
    if algo == "IQL":  return _iql(env_name, c)
    raise ValueError(algo)


ALGOS = ["IPPO", "A2C", "IQL"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("env"); p.add_argument("--timesteps", type=int, default=3_000_000)
    a = p.parse_args()
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=a.timesteps, NUM_ENVS=64)
    for algo in ALGOS:
        d = train(a.env, algo, cfg)
        print(f"{algo:5s} [{a.env}] reward {d[0]:+.4f} -> {d[-1]:+.4f}")
