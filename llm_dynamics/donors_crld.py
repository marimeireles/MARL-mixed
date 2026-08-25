"""The donors game as a CRLD dynamical system.

The mapping from donorSim's dyadic donors game onto pyCRLD:

  * Payoffs. Both players are simultaneously donors (donorSim's training
    framing): cooperating costs you c and gives the partner b. That IS
    the donation-game PD: (R, T, S, P) = (b-c, b, -c, 0), or, on the
    normalized scale that drove the GRPO gradient (paper Eq. 2, payoff
    (pi+c)/(b+c)): (b/(b+c), 1, 0, c/(b+c)).

  * w (re-encounter probability) -> the discount factor gamma. In iterated
    game theory the continuation probability and the discount factor are
    the same object; Nowak's direct-reciprocity condition w > c/b is the
    discounted-repeated-game cooperation threshold.

  * q (reputation availability) -> observability. With probability q the
    agent's information about the partner's last action(s) is available,
    else the agent sees only its own actions. In pyCRLD this is the
    row-stochastic blend  O = q * O_full + (1-q) * O_self  of the 'full'
    and 'self' observability regimes of the MARL-mixed paper.

  * memory-m -> HistoryEmbedded(env, h=(m, m, m)).

This module also provides fixed-opponent flow fields: the learning flow
of a single CRLD learner while the other seat is pinned to a scripted
strategy (AllC/AllD/TFT/...), which is the theoretical counterpart of the
"set strategy vs LLM" dyadic scenario. Axes are the learner's
P(cooperate | history state) in two chosen states — the reciprocity plane.

Observability helpers (_mask_round/_oset_for_mem/_obs_matrix) are copied
from scratch_repro/mem_obs_grids.py (importing it would drag in
matplotlib figure code).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from pyCRLD.Environments.SocialDilemma import SocialDilemma
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded


# ── Payoff reductions ─────────────────────────────────────────────────────

def donation_game_payoffs(b: float, c: float, normalized: bool = True) -> dict:
    """Donors game -> PD payoffs (R, T, S, P).

    normalized=True uses the Eq.-2 scale (pi+c)/(b+c) in [0,1] — the same
    numbers the RL reward used, so the CRLD learning-rate scale is
    comparable across (b, c) settings."""
    if normalized:
        s = b + c if (b + c) > 1e-9 else 1.0
        return dict(R=b / s, T=1.0, S=0.0, P=c / s)
    return dict(R=b - c, T=b, S=-c, P=0.0)


# Canonical 2x2 games for the LLM matrix-game experiments. Orderings:
#   PD:        T > R > P > S      Chicken:  T > R > S > P
#   Stag Hunt: R > T > P > S      Harmony:  R > T, S > P
MATRIX_GAMES: dict[str, dict] = {
    "ipd":      dict(label="Prisoner's Dilemma", R=3.0, T=5.0, S=0.0, P=1.0),
    "chicken":  dict(label="Chicken / Snowdrift", R=3.0, T=5.0, S=1.0, P=0.0),
    "staghunt": dict(label="Stag Hunt", R=5.0, T=3.0, S=0.0, P=1.0),
    "harmony":  dict(label="Harmony", R=5.0, T=3.0, S=2.0, P=0.0),
}


from .strategies import nowak_thresholds  # noqa: F401  (re-export)


# ── Observability (vendored from scratch_repro/mem_obs_grids.py) ──────────

def _mask_round(round_str, regime, ag, N):
    p = round_str.split(",")
    acts = p[:N]
    out = list(p)
    if regime == "full":
        pass
    elif regime == "blind":
        for j in range(N):
            out[j] = "."
    elif regime == "self":
        for j in range(N):
            if j != ag:
                out[j] = "."
    elif regime == "others":
        out[ag] = "."
    elif regime == "coop":
        for j in range(N):
            if j != ag and acts[j] != "c":
                out[j] = "."
    elif regime == "def":
        for j in range(N):
            if j != ag and acts[j] != "d":
                out[j] = "."
    else:
        raise ValueError(regime)
    return ",".join(out)


def _oset_for_mem(regime, base_entry, ag, N):
    rounds = base_entry.strip("|").split("|")
    return "|".join(_mask_round(r, regime, ag, N) for r in rounds) + "|"


def _obs_matrix(descriptions):
    n = len(descriptions)
    M = np.zeros((n, n))
    parts = [d.replace("|", ",").strip(",").split(",") for d in descriptions]
    fully = [all(p != "." for p in pp) for pp in parts]
    for i in range(n):
        if fully[i]:
            M[i, i] = 1.0
        else:
            for j in range(n):
                if all(a == "." or a == b for a, b in zip(parts[i], parts[j])):
                    M[i, j] = 1.0
    return M / M.sum(1, keepdims=True)


def apply_q_observability(memo, q: float, N: int = 2) -> None:
    """Blend each agent's observation matrix: full obs w.p. q, self-only
    obs w.p. 1-q. q=1 leaves the env untouched."""
    if q >= 1.0:
        return
    base = list(memo.Oset[0])
    O_full = _obs_matrix(base)  # identity
    for ag in range(N):
        masked = [_oset_for_mem("self", e, ag, N) for e in base]
        O_self = _obs_matrix(masked)
        memo.O[ag] = q * O_full + (1.0 - q) * O_self


# ── Environment / agent builders ──────────────────────────────────────────

def build_memo_env(R: float, T: float, S: float, P: float,
                   memory: int = 1, q: float = 1.0):
    """Matrix game -> memory-m history-embedded env with q-observability."""
    env = SocialDilemma(R=float(R), T=float(T), S=float(S), P=float(P))
    memo = HistoryEmbedded(env, h=(memory,) * 3)
    apply_q_observability(memo, q)
    return memo


def donors_memo_env(b: float, c: float, memory: int = 1, q: float = 1.0,
                    normalized: bool = True):
    pay = donation_game_payoffs(b, c, normalized)
    return build_memo_env(pay["R"], pay["T"], pay["S"], pay["P"], memory, q)


def build_mae(memo, w: float, algo: str = "ac", q: float = 1.0,
              learning_rate: float = 0.05, beta: float = 1.0):
    """CRLD learner on `memo` with discount gamma = w.

    gamma must be < 1 (the (1-gamma) prefactor vanishes at 1), so w is
    clamped to 0.99; w=1 in donorSim means 'guaranteed re-encounter',
    which is the gamma -> 1 limit."""
    gamma = float(np.clip(w, 0.0, 0.99))
    if q >= 1.0:
        if algo == "ac":
            from pyCRLD.Agents.StrategyActorCritic import stratAC as cls
        elif algo == "sarsa":
            from pyCRLD.Agents.StrategySARSA import stratSARSA as cls
        else:
            raise ValueError(algo)
    else:
        if algo == "ac":
            from pyCRLD.Agents.POStrategyActorCritic import POstratAC as cls
        elif algo == "sarsa":
            from pyCRLD.Agents.APOStrategySarsa import stratSARSA as cls
        else:
            raise ValueError(algo)
    return cls(env=memo, learning_rates=learning_rate,
               discount_factors=gamma, choice_intensities=beta)


# ── State-space utilities ─────────────────────────────────────────────────

def parse_state_rounds(label: str) -> list[tuple[str, str]]:
    """'c,d,.|c,c,.|' -> [('c','d'), ('c','c')], oldest -> newest."""
    out = []
    for r in label.strip("|").split("|"):
        p = r.split(",")
        out.append((p[0], p[1]))
    return out


def state_index(memo, rounds: list[tuple[str, str]]) -> int:
    """Index of the state whose remembered rounds (oldest -> newest) are
    `rounds`, each a ('c'|'d', 'c'|'d') pair (agent0, agent1)."""
    label = "".join(f"{a},{b},.|" for a, b in rounds)
    return list(memo.Sset).index(label)


def uniform_state(memo, own: str, other: str, memory: int,
                  agent: int = 0) -> int:
    """State where agent `agent` played `own` and the other played `other`
    in every remembered round."""
    pair = (own, other) if agent == 0 else (other, own)
    return state_index(memo, [pair] * memory)


def allc_state(memo) -> int:
    for i, lab in enumerate(memo.Sset):
        if "d" not in lab:
            return i
    return 0


# ── Scripted strategies as pinned CRLD policies ───────────────────────────

def strategy_policy(strategy: str, memo, agent: int = 1,
                    eps: float = 0.01) -> np.ndarray:
    """The scripted strategy as a policy X[s, a] over memo's state space
    (action 0 = cooperate), eps-mixed to keep full support.

    grim_trigger is approximated by its memory-m truncation (defect iff
    the other defected within the remembered window). tit_for_two_tats
    requires memory >= 2."""
    other = 1 - agent if agent in (0, 1) else 0
    Z = memo.Z
    memory = len(parse_state_rounds(memo.Sset[0]))
    if strategy == "tit_for_two_tats" and memory < 2:
        raise ValueError("tit_for_two_tats needs memory >= 2")
    pc = np.zeros(Z)
    for s, lab in enumerate(memo.Sset):
        rounds = parse_state_rounds(lab)
        theirs = [r[other] for r in rounds]  # other's actions, oldest->newest
        if strategy == "always_cooperate":
            p = 1.0
        elif strategy == "always_defect":
            p = 0.0
        elif strategy == "random":
            p = 0.5
        elif strategy in ("tit_for_tat", "suspicious_tit_for_tat"):
            p = 1.0 if theirs[-1] == "c" else 0.0  # opening move is not a state
        elif strategy == "tit_for_two_tats":
            p = 0.0 if (theirs[-1] == "d" and theirs[-2] == "d") else 1.0
        elif strategy == "grim_trigger":
            p = 0.0 if "d" in theirs else 1.0
        elif strategy == "wsls":
            own_last = rounds[-1][agent]
            p = (1.0 if own_last == "c" else 0.0) if theirs[-1] == "c" \
                else (0.0 if own_last == "c" else 1.0)
        elif strategy == "generous_tit_for_tat":
            p = 1.0 if theirs[-1] == "c" else 0.3
        else:
            raise ValueError(strategy)
        pc[s] = p
    pc = (1.0 - 2.0 * eps) * pc + eps
    return np.stack([pc, 1.0 - pc], axis=-1)


# ── Fixed-opponent dynamics ───────────────────────────────────────────────

def pinned_step(mae, X: np.ndarray, X_opp: np.ndarray,
                opp: int = 1) -> np.ndarray:
    """One CRLD step with the opponent's policy re-pinned afterwards."""
    Xj = np.array(X, dtype=float)
    Xj[opp] = X_opp
    Xn, _ = mae.step(jnp.array(Xj))
    Xn = np.array(Xn)
    Xn[opp] = X_opp
    return Xn


def fixed_opponent_trajectory(mae, memo, X_opp: np.ndarray,
                              X0_learner: np.ndarray, Tmax: int = 4000,
                              tolerance: float = 1e-6, learner: int = 0,
                              opp: int = 1) -> np.ndarray:
    """Iterate pinned steps; returns the learner's P(cooperate) per state
    over time, shape (T, Z)."""
    X = np.zeros((2, memo.Z, 2))
    X[learner] = X0_learner
    X[opp] = X_opp
    traj = [X[learner, :, 0].copy()]
    for _ in range(Tmax):
        Xn = pinned_step(mae, X, X_opp, opp)
        traj.append(Xn[learner, :, 0].copy())
        if np.max(np.abs(Xn[learner] - X[learner])) < tolerance:
            break
        X = Xn
    return np.array(traj)


def fixed_opponent_flow(mae, memo, X_opp: np.ndarray, sx: int, sy: int,
                        points=None, NrRandom: int = 8, seed: int = 0,
                        learner: int = 0, opp: int = 1):
    """Learning flow of the learner in the (P(C|sx), P(C|sy)) plane with
    the opponent pinned. Off-plane learner coordinates are randomized
    (marginalized by sampling, as in FlowPlot).

    Returns (XX, YY, dX, dY) with dX/dY of shape (ny, nx, NrRandom)."""
    if points is None:
        points = np.linspace(0.01, 0.99, 9)
    rng = np.random.default_rng(seed)
    Z = memo.Z
    nx = ny = len(points)
    dX = np.zeros((ny, nx, NrRandom))
    dY = np.zeros((ny, nx, NrRandom))
    for iy, yv in enumerate(points):
        for ix, xv in enumerate(points):
            for k in range(NrRandom):
                pc = rng.random(Z)
                pc[sx] = xv
                pc[sy] = yv
                X0 = np.stack([pc, 1.0 - pc], axis=-1)
                X = np.zeros((2, Z, 2))
                X[learner] = X0
                X[opp] = X_opp
                Xn = pinned_step(mae, X, X_opp, opp)
                dX[iy, ix, k] = Xn[learner, sx, 0] - xv
                dY[iy, ix, k] = Xn[learner, sy, 0] - yv
    XX, YY = np.meshgrid(points, points)
    return XX, YY, dX, dY
