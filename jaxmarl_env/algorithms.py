"""
Three MARL algorithm families on the heterogeneous-observation IPD, all logging
per-agent cooperation over training so their *behaviour* can be compared:

  IPPO  - clipped policy-gradient actor-critic   (coop_trajectories.coop_trajectory)
  A2C   - independent advantage actor-critic      (on-policy, no clipping)        [here]
  IQL   - independent Q-learning                   (value-based, epsilon-greedy)  [here]

Each trainer returns a (num_updates, num_agents) array of cooperation frequency.
"""
import os
from functools import partial
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState
import optax

from heterogeneous_ipd import make_hetero_ipd
import coop_trajectories as ippo   # reuse env + IPPO trainer

CFG = dict(ippo.CFG)


# ===========================================================================
#  A2C  - independent advantage actor-critic (on-policy, single epoch, no clip)
# ===========================================================================
class Tr(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; obs: jnp.ndarray


def coop_trajectory_a2c(regime, num_agents=2, seed=0, cfg=CFG):
    c = dict(cfg)
    env = make_hetero_ipd(num_agents=num_agents, regime=regime, num_steps=c["NUM_STEPS"])
    agents = env.agents; nA = env.num_agents; NE = c["NUM_ENVS"]; n_actors = nA * NE
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    obs_dim = env.observation_space(agents[0]).shape[0]

    def batch(d): return jnp.stack([d[a] for a in agents]).reshape(n_actors, -1)
    def rew_b(d): return jnp.stack([d[a] for a in agents]).reshape(n_actors)
    def unb(x): x = x.reshape(nA, NE); return {a: x[i] for i, a in enumerate(agents)}

    @jax.jit
    def train(rng):
        net = ippo.AC(2, c["HIDDEN"])
        rng, r = jax.random.split(rng)
        ts = TrainState.create(apply_fn=net.apply, params=net.init(r, jnp.zeros((1, obs_dim))),
                               tx=optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                                              optax.adam(c["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(run, _):
            def step(run, _):
                ts, st, ob, rng = run
                logits, val = net.apply(ts.params, batch(ob))
                pi = ippo.Categorical(logits)
                rng, r = jax.random.split(rng)
                act = pi.sample(r)
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                coop = (act.reshape(nA, NE) == 0).mean(1)
                return (ts, st, ob2, rng), (Tr(rew_b(dn), act, val, rew_b(rew), batch(ob)), coop)
            run, (traj, coop) = jax.lax.scan(step, run, None, c["NUM_STEPS"])
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

            def loss(p):                              # single full-batch A2C update
                logits, val = net.apply(p, traj.obs.reshape(-1, obs_dim))
                pi = ippo.Categorical(logits)
                lp = pi.log_prob(traj.action.reshape(-1))
                a = adv.reshape(-1); a = (a - a.mean()) / (a.std() + 1e-8)
                actor = -(lp * a).mean()
                critic = 0.5 * ((val - tgt.reshape(-1)) ** 2).mean()
                return actor + c["VF_COEF"] * critic - c["ENT_COEF"] * pi.entropy().mean()
            ts = ts.apply_gradients(grads=jax.grad(loss)(ts.params))
            return (ts, st, last_ob, rng), coop.mean(0)

        rng, r = jax.random.split(rng)
        _, coop_traj = jax.lax.scan(upd, (ts, st, obsv, r), None, c["NUM_UPDATES"])
        return coop_traj

    return np.asarray(train(jax.random.PRNGKey(seed)))


# ===========================================================================
#  IQL - independent Q-learning (value-based, epsilon-greedy, target network)
# ===========================================================================
class QNet(nn.Module):
    action_dim: int
    hidden: int = 64
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        h = nn.relu(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(h))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(h)


def coop_trajectory_iql(regime, num_agents=2, seed=0, cfg=CFG,
                        eps_start=1.0, eps_end=0.05, target_period=200):
    c = dict(cfg)
    env = make_hetero_ipd(num_agents=num_agents, regime=regime, num_steps=c["NUM_STEPS"])
    agents = env.agents; nA = env.num_agents; NE = c["NUM_ENVS"]; n_actors = nA * NE
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    obs_dim = env.observation_space(agents[0]).shape[0]

    def batch(d): return jnp.stack([d[a] for a in agents]).reshape(n_actors, -1)
    def rew_b(d): return jnp.stack([d[a] for a in agents]).reshape(n_actors)
    def unb(x): x = x.reshape(nA, NE); return {a: x[i] for i, a in enumerate(agents)}

    @jax.jit
    def train(rng):
        net = QNet(2, c["HIDDEN"])
        rng, r = jax.random.split(rng)
        params = net.init(r, jnp.zeros((1, obs_dim)))
        ts = TrainState.create(apply_fn=net.apply, params=params,
                               tx=optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                                              optax.adam(c["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(carry, t):
            ts, tgt_params, st, ob, rng = carry
            eps = eps_end + (eps_start - eps_end) * (1 - t / c["NUM_UPDATES"])

            def step(run, _):
                ts, st, ob, rng = run
                q = net.apply(ts.params, batch(ob))
                rng, r1, r2 = jax.random.split(rng, 3)
                greedy = jnp.argmax(q, -1)
                rand = jax.random.randint(r1, greedy.shape, 0, 2)
                explore = jax.random.uniform(r2, greedy.shape) < eps
                act = jnp.where(explore, rand, greedy)
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                coop = (act.reshape(nA, NE) == 0).mean(1)
                trans = (batch(ob), act, rew_b(rew), batch(ob2), rew_b(dn))
                return (ts, st, ob2, rng), (trans, coop)
            (ts, st, ob, rng), (trans, coop) = jax.lax.scan(step, (ts, st, ob, rng),
                                                            None, c["NUM_STEPS"])
            obs_b, act_b, rew_b_, nobs_b, done_b = [x.reshape((-1,) + x.shape[2:]) for x in trans]

            def loss(p):
                q = net.apply(p, obs_b)
                qa = jnp.take_along_axis(q, act_b[:, None], axis=1)[:, 0]
                nq = net.apply(tgt_params, nobs_b).max(-1)
                target = rew_b_ + c["GAMMA"] * nq * (1 - done_b)
                return ((qa - jax.lax.stop_gradient(target)) ** 2).mean()
            ts = ts.apply_gradients(grads=jax.grad(loss)(ts.params))
            tgt_params = jax.lax.cond((t % target_period) == 0,
                                      lambda: ts.params, lambda: tgt_params)
            return (ts, tgt_params, st, ob, rng), coop.mean(0)

        rng, r = jax.random.split(rng)
        _, coop_traj = jax.lax.scan(upd, (ts, params, st, obsv, r),
                                    jnp.arange(c["NUM_UPDATES"]))
        return coop_traj

    return np.asarray(train(jax.random.PRNGKey(seed)))


ALGOS = {"IPPO": ippo.coop_trajectory, "A2C": coop_trajectory_a2c, "IQL": coop_trajectory_iql}


if __name__ == "__main__":
    import sys
    regime = sys.argv[1] if len(sys.argv) > 1 else "full"
    cfg = dict(CFG, TOTAL_TIMESTEPS=int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000)
    for name, fn in ALGOS.items():
        t = fn(regime, cfg=cfg)
        print(f"{name:5s} [{regime}] coop {t[0].mean():.2f} -> {t[-1].mean():.2f}")
