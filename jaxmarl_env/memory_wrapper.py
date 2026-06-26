"""A generic JaxMARL observation-memory (frame-stacking) wrapper, plus a
coin_game memory experiment.

`ObsMemoryWrapper(base_env, k)` stacks the last k per-agent observations and
returns their concatenation, so obs_dim becomes k * base_obs_dim. It preserves
the JaxMARL API and is jittable / vmappable:

    reset(key)               -> (obs_dict, state)
    step(key, state, actions)-> (obs_dict, state, reward_dict, done_dict, info)

This implements "add memory" via a reusable wrapper: any feed-forward policy
trained on the wrapped env now conditions on the last k frames, recovering a
memory-k agent without touching the policy or the base environment.

Deep-RL context (GPU):
  cd jaxmarl_env && export PATH=".../cuda_nvcc/bin:$PATH" \
    && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    && ../.venv-jaxmarl/bin/python memory_wrapper.py
"""
import os
from functools import partial
from typing import Dict, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import chex
from flax import struct
from flax.training.train_state import TrainState
import optax

import jaxmarl
from jaxmarl.environments.coin_game.coin_game import MOVES
from jaxmarl.wrappers.baselines import JaxMARLWrapper

from run_all import introspect, ActorCritic, Categorical, DEFAULT

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


@struct.dataclass
class MemoryState:
    env_state: chex.ArrayTree            # the wrapped env's own state
    buffer: Dict[str, chex.Array]        # per agent: (k, base_obs_dim), index -1 = newest


class ObsMemoryWrapper(JaxMARLWrapper):
    """Frame-stacking wrapper: observation = concat of the last k frames."""

    def __init__(self, env, k: int):
        super().__init__(env)
        assert k >= 1, "memory length k must be >= 1"
        self.k = k

    def _flat(self, x):
        return jnp.asarray(x).reshape(-1)            # base obs -> flat (base_obs_dim,)

    def _stack(self, buf):
        # concat oldest..newest -> (k * base_obs_dim,)
        return {a: buf[a].reshape(-1) for a in self._env.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        # initialise every slot with the first frame (standard frame-stacking)
        buffer = {a: jnp.tile(self._flat(obs[a])[None], (self.k, 1))
                  for a in self._env.agents}
        return self._stack(buffer), MemoryState(env_state, buffer)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, actions)
        new_buffer = {}
        for a in self._env.agents:
            frame = self._flat(obs[a])
            # roll the window: drop oldest, append newest
            rolled = jnp.concatenate([state.buffer[a][1:], frame[None]], axis=0)
            # on episode boundary, refill the window with the fresh frame so we
            # never mix observations across episodes
            fresh = jnp.tile(frame[None], (self.k, 1))
            new_buffer[a] = jnp.where(done[a], fresh, rolled)
        return (self._stack(new_buffer), MemoryState(env_state, new_buffer),
                reward, done, info)


# =========================================================================== #
# Verification: shapes scale with k, rewards are identical to the base env.    #
# =========================================================================== #
def verify():
    base = jaxmarl.make("coin_game")
    key = jax.random.PRNGKey(0)
    obs0, _ = base.reset(key)
    base_dim = int(np.prod(obs0["0"].shape))
    print(f"[verify] base coin_game obs_dim = {base_dim}")

    for k in (1, 2, 4):
        env = ObsMemoryWrapper(base, k)
        # reset shape check
        obs, st = env.reset(key)
        ok_shape = all(obs[a].shape == (k * base_dim,) for a in base.agents)
        # step shape check (jit + a real step)
        acts = {a: jnp.array(4) for a in base.agents}    # "stay"
        kk = jax.random.PRNGKey(1)
        wobs, wst, wrew, wdone, _ = env.step(kk, st, acts)
        ok_step = all(wobs[a].shape == (k * base_dim,) for a in base.agents)

        # reward equivalence: same key/state/action through the BASE env must
        # give identical rewards (wrapper only touches observations)
        _, bst = base.reset(key)
        _, _, brew, bdone, _ = base.step(kk, bst, acts)
        same_rew = all(bool(jnp.allclose(wrew[a], brew[a])) for a in base.agents)
        same_done = all(bool(wdone[a] == bdone[a]) for a in base.agents)
        print(f"[verify] k={k}: obs {obs['0'].shape} reset_ok={ok_shape} "
              f"step_ok={ok_step} rewards_identical={same_rew} dones_identical={same_done}")

    # vmap/jit sanity: vmapped reset+step over a batch
    env = ObsMemoryWrapper(base, 4)
    keys = jax.random.split(key, 8)
    vobs, vst = jax.vmap(env.reset)(keys)
    vacts = {a: jnp.full((8,), 4) for a in base.agents}
    vobs2, vst2, vrew, vdone, _ = jax.vmap(env.step)(keys, vst, vacts)
    print(f"[verify] vmapped(8) obs shape = {vobs['0'].shape}  step obs = {vobs2['0'].shape}  (jittable/vmappable OK)")


# =========================================================================== #
# coin_game cooperation metric (own-coin pickups), recomputed from base state. #
# =========================================================================== #
def _coop_step(env_state, act, NE):
    a0, a1 = act[:NE], act[NE:2 * NE]
    new_red = (env_state.red_pos + MOVES[a0]) % 3
    new_blue = (env_state.blue_pos + MOVES[a1]) % 3
    rr = jnp.all(new_red == env_state.red_coin_pos, -1)     # red grabs own coin (coop)
    bb = jnp.all(new_blue == env_state.blue_coin_pos, -1)   # blue grabs own coin (coop)
    rb = jnp.all(new_red == env_state.blue_coin_pos, -1)    # red steals (defect)
    br = jnp.all(new_blue == env_state.red_coin_pos, -1)    # blue steals (defect)
    coop = (rr + bb).sum()
    total = (rr + bb + rb + br).sum()
    return coop, total


class Tr(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray


def _ippo_memory(k, c):
    """IPPO on coin_game wrapped with k-frame memory; logs coop fraction/update."""
    NE = c["NUM_ENVS"]
    base = jaxmarl.make("coin_game")
    env = ObsMemoryWrapper(base, k)
    agents = base.agents
    nA = len(agents)
    obs0, _ = env.reset(jax.random.PRNGKey(0))
    obs_dim = int(obs0["0"].shape[0])            # == k * 36
    act_dim = 5
    n_actors = nA * NE
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    full_mask = jnp.ones((n_actors, act_dim), bool)

    batch = lambda d: jnp.concatenate([d[a].reshape(d[a].shape[0], -1) for a in agents], 0)
    rew_b = lambda d: jnp.concatenate([d[a].reshape(-1) for a in agents], 0)
    unb = lambda x: {a: x.reshape(nA, NE)[i] for i, a in enumerate(agents)}

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
                coop, total = _coop_step(st.env_state, act, NE)
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
                        pi = Categorical(logits)
                        lp = pi.log_prob(traj.action)
                        an = (adv - adv.mean()) / (adv.std() + 1e-8)
                        ratio = jnp.exp(lp - traj.log_prob)
                        al = -jnp.minimum(ratio * an, jnp.clip(
                            ratio, 1 - c["CLIP_EPS"], 1 + c["CLIP_EPS"]) * an).mean()
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
            s, _ = jax.lax.scan(epoch, (ts, traj, adv, tgt, rng), None, c["UPDATE_EPOCHS"])
            coop_frac = coop.sum() / jnp.maximum(total.sum(), 1.0)
            return (s[0], st, last_ob, s[-1]), coop_frac

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, c["NUM_UPDATES"])
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


def memory_experiment(ks=(1, 2, 4), timesteps=1_500_000, num_envs=64):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=timesteps, NUM_ENVS=num_envs)
    curves = {}
    for k in ks:
        d = _ippo_memory(k, dict(cfg))
        curves[k] = d
        print(f"  IPPO coin_game k={k}: coop {d[0]:.3f} -> {d[-1]:.3f}", flush=True)
    np.save(os.path.join(RESULTS, "coin_memory.npy"),
            np.array([curves[k] for k in ks]))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    cmap = {1: "#0254a3", 2: "#2a8c5a", 4: "#d1495b"}
    xstep = cfg["NUM_STEPS"] * cfg["NUM_ENVS"]
    for k in ks:
        y = curves[k]
        ax.plot(np.arange(len(y)) * xstep, y, color=cmap.get(k, None), lw=2,
                label=f"memory k={k}  (obs_dim={k*36})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("cooperation fraction (own-coin pickups)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("coin_game cooperation vs. observation memory (IPPO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "coin_memory.png"), dpi=300)
    print("saved coin_memory.png + coin_memory.npy")


if __name__ == "__main__":
    verify()
    memory_experiment()
