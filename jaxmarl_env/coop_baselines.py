"""Cooperative MARL baselines with CENTRALISED training, benchmarked against the
independent learners in gen_algos.py on a fully cooperative JaxMARL env.

Three new algorithms (discrete actions, shared team reward):

  MAPPO  PPO with a CENTRALISED critic. The actor is per-agent / decentralised
         (shared params, sees only its own obs), exactly like IPPO. The value
         function instead takes the concatenation of ALL agents' observations
         (a global-state proxy) and outputs one value that scores every agent's
         advantage. Trained on the team return (sum of per-agent rewards).

  VDN    Value decomposition. A shared per-agent Q-net picks epsilon-greedy
         actions. Q_tot = sum_i Q_i(o_i, a_i). The TD target uses the TEAM
         reward and Q_tot of the next state's per-agent argmax actions, bootstrapped
         from a target network.

  QMIX   Like VDN, but Q_tot is a monotonic MIXING network of the per-agent Q_i,
         whose (non-negative) weights come from a hypernetwork conditioned on the
         global state. Monotonicity (abs weights) guarantees that argmax_a Q_tot
         factorises into per-agent argmaxes, so decentralised execution is greedy.

All three reuse the env plumbing (introspect / _env_helpers / QNet / Categorical)
from gen_algos.py + run_all.py. Every algorithm returns a (NUM_UPDATES,) array of
the MEAN PER-AGENT reward per update -- the SAME quantity gen_algos logs -- so the
6-way comparison is on identical axes. (In simple_spread the reward is shared, so
mean per-agent reward == the team reward / N.)

Usage:
    python coop_baselines.py --debug          # 100k-timestep shape check
    python coop_baselines.py                  # full 2M-timestep benchmark + figure
"""
import argparse, os, time
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
import optax
import jaxmarl

from run_all import Categorical, DEFAULT, SkipEnv, introspect
from gen_algos import QNet, _env_helpers
import gen_algos as G

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

ENV = "MPE_simple_spread_v3"
ALL_ALGOS = ["IPPO", "A2C", "IQL", "MAPPO", "VDN", "QMIX"]
COLORS = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b",
          "MAPPO": "#7b2cbf", "VDN": "#e07b00", "QMIX": "#118ab2"}


# --------------------------------------------------------------------------- nets
class Actor(nn.Module):
    """Decentralised categorical actor (per-agent obs in, logits out)."""
    action_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        a = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(a))
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(a)


class Critic(nn.Module):
    """Centralised value (global-state in, one scalar out)."""
    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(x))
        v = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)))(v))
        v = nn.Dense(1, kernel_init=orthogonal(1.0))(v)
        return jnp.squeeze(v, -1)


class Mixer(nn.Module):
    """Monotonic QMIX mixing network. Combines per-agent Qs into Q_tot with
    non-negative (abs) weights produced by a hypernetwork over the global state."""
    n_agents: int
    embed_dim: int = 32
    hyper_hidden: int = 64

    @nn.compact
    def __call__(self, q_agents, state):     # q_agents (B, nA), state (B, S)
        nA, E = self.n_agents, self.embed_dim
        # first layer: |W1| (B, nA, E), b1 (B, E)
        w1 = jnp.abs(nn.Dense(nA * E, kernel_init=orthogonal(1.0))(state))
        w1 = w1.reshape(-1, nA, E)
        b1 = nn.Dense(E, kernel_init=orthogonal(1.0))(state)
        hidden = nn.elu(jnp.einsum("bn,bne->be", q_agents, w1) + b1)   # (B, E)
        # second layer: |W2| (B, E, 1), b2 (B, 1) via a 2-layer hypernet
        w2 = jnp.abs(nn.Dense(E, kernel_init=orthogonal(1.0))(state)).reshape(-1, E, 1)
        b2 = nn.Dense(self.hyper_hidden, kernel_init=orthogonal(1.0))(state)
        b2 = nn.Dense(1, kernel_init=orthogonal(1.0))(nn.relu(b2))     # (B, 1)
        y = jnp.einsum("be,bef->bf", hidden, w2) + b2                  # (B, 1)
        return y[:, 0]


# ------------------------------------------------------------------ shared helpers
def _coop_helpers(env, meta, NE):
    """Extends _env_helpers with global-state / team-reward builders.

    Batched obs are agent-major: rows [0:NE]=agent_0, [NE:2NE]=agent_1, ...
    `gstate(obs_b)` -> (NE, nA*obs_dim) per-env concat of all agents' obs.
    `team(rew_b_vec)` -> (NE,) sum of per-agent rewards (the team reward).
    `done_env(done_dict)` -> (NE,) per-env done (agents terminate together in MPE).
    """
    agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb = _env_helpers(env, meta, NE)

    def gstate(obs_b):                       # (nA*NE, obs_dim) -> (NE, nA*obs_dim)
        g = obs_b.reshape(nA, NE, obs_dim).transpose(1, 0, 2).reshape(NE, nA * obs_dim)
        return g

    def team(rb):                            # (nA*NE,) -> (NE,)
        return rb.reshape(nA, NE).sum(0)

    def done_env(dn):
        return dn[agents[0]].reshape(-1)     # (NE,)

    return (agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb,
            gstate, team, done_env)


# --------------------------------------------------------------------------- MAPPO
def _train_mappo(env, meta, c):
    NE = c["NUM_ENVS"]
    (agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb,
     gstate, team, done_env) = _coop_helpers(env, meta, NE)
    n_actors = nA * NE
    state_dim = nA * obs_dim
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)

    def gstate_actor(obs_b):                 # per-actor global state (nA*NE, state_dim)
        return jnp.tile(gstate(obs_b), (nA, 1))

    actor = Actor(act_dim, c["HIDDEN"])
    critic = Critic(c["HIDDEN"])

    @jax.jit
    def train(rng):
        rng, ra, rc = jax.random.split(rng, 3)
        params = {"actor": actor.init(ra, jnp.zeros((1, obs_dim))),
                  "critic": critic.init(rc, jnp.zeros((1, state_dim)))}
        ts = TrainState.create(
            apply_fn=None, params=params,
            tx=optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                           optax.adam(c["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def upd(run, _):
            def step(run, _):
                ts, st, ob, rng = run
                ob_b = batch(ob); g = gstate_actor(ob_b)
                logits = actor.apply(ts.params["actor"], ob_b)
                pi = Categorical(logits, mask=full_mask)
                rng, r = jax.random.split(rng); act = pi.sample(r)
                val = critic.apply(ts.params["critic"], g)
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                d = jnp.concatenate([dn[a].reshape(-1) for a in agents], 0)
                team_r = jnp.tile(team(rew_b(rew)), nA)          # per-actor team reward
                tr = (d, act, val, team_r, pi.log_prob(act), ob_b, g, full_mask)
                return (ts, st, ob2, rng), (tr, rew_b(rew).mean())
            run, (traj, rm) = jax.lax.scan(step, run, None, c["NUM_STEPS"])
            d, act, val, team_r, logp, obs_b, gst, mask = traj
            ts, st, last_ob, rng = run
            last_val = critic.apply(ts.params["critic"], gstate_actor(batch(last_ob)))

            # GAE over per-actor trajectories (identical across agent copies)
            def scan_gae(carry, t):
                g_acc, nv = carry
                rwd, dn, vl = t
                delta = rwd + c["GAMMA"] * nv * (1 - dn) - vl
                g_acc = delta + c["GAMMA"] * c["GAE_LAMBDA"] * (1 - dn) * g_acc
                return (g_acc, vl), g_acc
            _, adv = jax.lax.scan(scan_gae, (jnp.zeros_like(last_val), last_val),
                                  (team_r, d, val), reverse=True, unroll=16)
            tgt = adv + val

            def epoch(s, _):
                def mb(ts, b):
                    (o, gs, a, lp, mk), ad, tg = b
                    def loss(p):
                        logits = actor.apply(p["actor"], o)
                        pi = Categorical(logits, mask=mk)
                        nlp = pi.log_prob(a)
                        value = critic.apply(p["critic"], gs)
                        an = (ad - ad.mean()) / (ad.std() + 1e-8)
                        ratio = jnp.exp(nlp - lp)
                        al = -jnp.minimum(ratio * an, jnp.clip(
                            ratio, 1 - c["CLIP_EPS"], 1 + c["CLIP_EPS"]) * an).mean()
                        vl = 0.5 * ((value - tg) ** 2).mean()
                        return al + c["VF_COEF"] * vl - c["ENT_COEF"] * pi.entropy().mean()
                    return ts.apply_gradients(grads=jax.grad(loss)(ts.params)), None
                ts, rng = s
                rng, r = jax.random.split(rng)
                B = c["NUM_STEPS"] * n_actors
                data = ((obs_b, gst, act, logp, mask), adv, tgt)
                flat = jax.tree_util.tree_map(lambda x: x.reshape((B,) + x.shape[2:]), data)
                perm = jax.random.permutation(r, B)
                flat = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, 0), flat)
                mbs = jax.tree_util.tree_map(
                    lambda x: x.reshape((c["NUM_MINIBATCHES"], -1) + x.shape[1:]), flat)
                ts, _ = jax.lax.scan(mb, ts, mbs)
                return (ts, rng), None
            (ts, rng), _ = jax.lax.scan(epoch, (ts, rng), None, c["UPDATE_EPOCHS"])
            return (ts, st, last_ob, rng), rm.mean()

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, st, obsv, r), None, c["NUM_UPDATES"])
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


# ----------------------------------------------------------------------- VDN / QMIX
def _train_value_mix(env, meta, c, mixing, eps_start=1.0, eps_end=0.05, target_period=10):
    """Shared trainer for VDN (mixing='vdn') and QMIX (mixing='qmix').

    NOTE on `target_period`: with one gradient step per update there are only
    ~244 updates in a 2M-step run. Syncing the target net every 200 updates (the
    IQL default in gen_algos) means it barely moves, and the bootstrapped Q_tot
    diverges (classic deadly-triad / overestimation) -- QMIX in particular blows
    up to ~-2.6. A standard, more frequent sync (every 10 updates) keeps both
    methods stable, so we use that here; it is a stability knob, not a change to
    the data / gradient-step budget."""
    NE = c["NUM_ENVS"]
    (agents, nA, obs_dim, act_dim, full_mask, batch, rew_b, unb,
     gstate, team, done_env) = _coop_helpers(env, meta, NE)
    state_dim = nA * obs_dim
    c["NUM_UPDATES"] = c["TOTAL_TIMESTEPS"] // (c["NUM_STEPS"] * NE)
    neg = jnp.where(full_mask, 0.0, -1e9)                # (nA*NE, act_dim)
    net = QNet(act_dim, c["HIDDEN"])
    mixer = Mixer(nA) if mixing == "qmix" else None

    @jax.jit
    def train(rng):
        rng, rq = jax.random.split(rng)
        qp = net.init(rq, jnp.zeros((1, obs_dim)))
        params = {"q": qp}
        if mixing == "qmix":
            rng, rm = jax.random.split(rng)
            params["mix"] = mixer.init(rm, jnp.zeros((1, nA)), jnp.zeros((1, state_dim)))
        ts = TrainState.create(
            apply_fn=None, params=params,
            tx=optax.chain(optax.clip_by_global_norm(c["MAX_GRAD_NORM"]),
                           optax.adam(c["LR"], eps=1e-5)))
        rng, r = jax.random.split(rng)
        obsv, st = jax.vmap(env.reset)(jax.random.split(r, NE))

        def mix_tot(p, qa_actor, gst):       # qa_actor (T, nA*NE) -> (T, NE)
            T = qa_actor.shape[0]
            if mixing == "vdn":
                return qa_actor.reshape(T, nA, NE).sum(1)
            qe = qa_actor.reshape(T, nA, NE).transpose(0, 2, 1).reshape(T * NE, nA)
            ge = gst.reshape(T * NE, state_dim)
            return mixer.apply(p["mix"], qe, ge).reshape(T, NE)

        def upd(carry, t):
            ts, tgt_p, st, ob, rng = carry
            eps = eps_end + (eps_start - eps_end) * (1 - t / c["NUM_UPDATES"])
            def step(run, _):
                ts, st, ob, rng = run
                ob_b = batch(ob)
                q = net.apply(ts.params["q"], ob_b) + neg
                rng, r1, r2 = jax.random.split(rng, 3)
                act = jnp.where(jax.random.uniform(r2, (q.shape[0],)) < eps,
                                jax.random.randint(r1, (q.shape[0],), 0, act_dim),
                                jnp.argmax(q, -1))
                rng, r = jax.random.split(rng)
                ob2, st, rew, dn, info = jax.vmap(env.step)(jax.random.split(r, NE), st, unb(act))
                return (ts, st, ob2, rng), ((ob_b, act, team(rew_b(rew)), batch(ob2),
                                             gstate(ob_b), done_env(dn)), rew_b(rew).mean())
            (ts, st, ob, rng), (tr, rm) = jax.lax.scan(step, (ts, st, ob, rng), None, c["NUM_STEPS"])
            obs_b, act_b, teamr, nobs_b, gst, done_b = tr   # leading dim T
            T = c["NUM_STEPS"]
            negT = neg[None]                                # broadcast over T
            # global state of the NEXT obs (for the target mixer / QMIX)
            n_gst = nobs_b.reshape(T, nA, NE, obs_dim).transpose(0, 2, 1, 3).reshape(T, NE, state_dim)

            def loss(p):
                q = net.apply(p["q"], obs_b) + negT         # (T, nA*NE, act_dim)
                qa = jnp.take_along_axis(q, act_b[..., None], -1)[..., 0]  # (T, nA*NE)
                qtot = mix_tot(p, qa, gst)                  # (T, NE)
                nq = (net.apply(tgt_p["q"], nobs_b) + negT).max(-1)        # (T, nA*NE)
                qtot_n = mix_tot(tgt_p, nq, n_gst)          # target mixer on next global state
                target = teamr + c["GAMMA"] * qtot_n * (1 - done_b)
                return ((qtot - jax.lax.stop_gradient(target)) ** 2).mean()
            ts = ts.apply_gradients(grads=jax.grad(loss)(ts.params))
            tgt_p = jax.lax.cond((t % target_period) == 0, lambda: ts.params, lambda: tgt_p)
            return (ts, tgt_p, st, ob, rng), rm.mean()

        rng, r = jax.random.split(rng)
        _, dyn = jax.lax.scan(upd, (ts, params, st, obsv, r), jnp.arange(c["NUM_UPDATES"]))
        return dyn
    return np.asarray(train(jax.random.PRNGKey(c.get("SEED", 0))))


# --------------------------------------------------------------------------- driver
def train(env_name, algo, cfg):
    if algo in G.ALGOS:                          # reuse independent learners
        return G.train(env_name, algo, cfg)
    env = jaxmarl.make(env_name)
    meta = introspect(env)
    if meta[5] != "discrete":
        raise SkipEnv("coop baselines are discrete-only")
    if algo == "MAPPO": return _train_mappo(env, meta, dict(cfg))
    if algo == "VDN":   return _train_value_mix(env, meta, dict(cfg), "vdn")
    if algo == "QMIX":  return _train_value_mix(env, meta, dict(cfg), "qmix")
    raise ValueError(algo)


def benchmark(timesteps, algos=ALL_ALGOS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=timesteps, NUM_ENVS=64)
    curves = {}
    for algo in algos:
        t0 = time.time()
        d = train(ENV, algo, cfg)
        curves[algo] = d
        print(f"  {algo:6s} {d[0]:+.3f} -> {d[-1]:+.3f}   "
              f"(best {d.max():+.3f}, {time.time()-t0:.1f}s)", flush=True)
    np.save(os.path.join(RESULTS, "coop_baselines.npy"),
            np.array(curves, dtype=object), allow_pickle=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in algos:
        y = curves[algo]
        x = (np.arange(len(y)) + 1) * cfg["NUM_STEPS"] * cfg["NUM_ENVS"]
        ax.plot(x, y, color=COLORS.get(algo), label=algo, lw=2)
    ax.set_title(f"Cooperative MARL baselines on {ENV}\n(shared reward; higher = better)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean per-agent reward / step")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULTS, "coop_baselines.png")
    fig.savefig(out, dpi=300)
    print("saved", out)
    return curves


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true", help="100k-step shape check, new algos only")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--algos", nargs="*", default=None)
    a = p.parse_args()
    if a.debug:
        algos = a.algos or ["MAPPO", "VDN", "QMIX"]
        cfg = dict(DEFAULT, TOTAL_TIMESTEPS=100_000, NUM_ENVS=64)
        for algo in algos:
            d = train(ENV, algo, cfg)
            print(f"[debug] {algo:6s} shape={d.shape} {d[0]:+.3f} -> {d[-1]:+.3f}", flush=True)
    else:
        benchmark(a.timesteps, a.algos or ALL_ALGOS)
