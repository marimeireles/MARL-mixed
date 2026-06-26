"""Deep-RL per-agent cooperation phase portrait for coin_game (IPPO).

coin_coop.py logs an AGGREGATE cooperation fraction. Here we log PER-AGENT
cooperation -- agent-0 (red) own-coin fraction vs agent-1 (blue) own-coin
fraction -- over training, and plot the sampled trajectory in the
(agent0 coop, agent1 coop) plane. This is the deep-RL analogue of the CRLD
"game dynamics diagram": a trajectory only (no flow field, since it is sampled,
not an analytic vector field).

For each agent, cooperation fraction this update =
    own-coin pickups / (own-coin pickups + other-coin pickups)
(red: rr / (rr + rb);  blue: bb / (bb + br)), summed over envs and steps.

GPU context:
  cd jaxmarl_env && export PATH=".../cuda_nvcc/bin:$PATH" \
    && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    && ../.venv-jaxmarl/bin/python coin_phase.py
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
from gen_algos import _env_helpers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _coop_step_peragent(st, act, NE):
    """Per-agent cooperation & total pickups this step (summed over the NE envs)."""
    a0, a1 = act[:NE], act[NE:2 * NE]
    new_red = (st.red_pos + MOVES[a0]) % 3
    new_blue = (st.blue_pos + MOVES[a1]) % 3
    rr = jnp.all(new_red == st.red_coin_pos, -1)     # red grabs own coin (coop0)
    bb = jnp.all(new_blue == st.blue_coin_pos, -1)   # blue grabs own coin (coop1)
    rb = jnp.all(new_red == st.blue_coin_pos, -1)    # red steals (defect0)
    br = jnp.all(new_blue == st.red_coin_pos, -1)    # blue steals (defect1)
    return rr.sum(), (rr + rb).sum(), bb.sum(), (bb + br).sum()


class Tr(NamedTuple):
    done: jnp.ndarray; action: jnp.ndarray; value: jnp.ndarray
    reward: jnp.ndarray; log_prob: jnp.ndarray; obs: jnp.ndarray


def ippo_phase(c):
    """IPPO on coin_game returning (coop0, coop1) per update."""
    NE = c["NUM_ENVS"]
    env = jaxmarl.make("coin_game"); meta = introspect(env)
    agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb = _env_helpers(env, meta, NE)
    n_actors = nA * NE
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)

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
                c0, t0, c1, t1 = _coop_step_peragent(st, act, NE)
                rng, r = jax.random.split(rng)
                ob2, st2, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                return (ts, st2, ob2, rng), (Tr(d, act, val, rew_b(rew), pi.log_prob(act),
                                                batch(ob)), jnp.array([c0, t0, c1, t1]))
            run, (traj, counts) = jax.lax.scan(step, run, None, c["NUM_STEPS"])
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
            tot = counts.sum(0)        # [c0, t0, c1, t1] summed over steps
            coop0 = tot[0] / jnp.maximum(tot[1], 1.0)
            coop1 = tot[2] / jnp.maximum(tot[3], 1.0)
            return (s[0], st, last_ob, s[-1]), jnp.array([coop0, coop1])

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, c["NUM_UPDATES"])
        return dyn                     # (NUM_UPDATES, 2)
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


def run(timesteps=1_500_000, num_envs=64, smooth=15):
    from run_all import DEFAULT
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=timesteps, NUM_ENVS=num_envs)
    dyn = ippo_phase(cfg)              # (T, 2): columns = coop0, coop1
    np.save(os.path.join(RESULTS, "coin_phase.npy"), dyn)
    assert (dyn >= 0).all() and (dyn <= 1).all(), "per-agent coop out of [0,1]!"

    # light smoothing so the sampled path is readable
    def mav(x, w):
        if w <= 1 or len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")
    c0, c1 = mav(dyn[:, 0], smooth), mav(dyn[:, 1], smooth)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(c0, c1, color="purple", lw=1.4, alpha=0.85)
    ax.plot(c0[0], c1[0], "x", color="black", ms=11, mew=2.5, label="start")
    ax.plot(c0[-1], c1[-1], "o", color="black", ms=10, label="end")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)      # symmetry diagonal
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("Agent 0 (red) cooperation  P(grab own coin)")
    ax.set_ylabel("Agent 1 (blue) cooperation  P(grab own coin)")
    ax.set_title("coin_game per-agent cooperation phase portrait (IPPO)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "coin_phase.png"), dpi=300)
    print(f"saved coin_phase.png ; start=({dyn[0,0]:.3f},{dyn[0,1]:.3f}) "
          f"end=({dyn[-1,0]:.3f},{dyn[-1,1]:.3f})  in[0,1]=OK")


if __name__ == "__main__":
    run()
