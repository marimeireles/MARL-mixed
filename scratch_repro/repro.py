"""
Clean reproduction of the heterogeneous-observation IPD experiments from
'Modeling Cooperation in Heterogeneous Multi-Agent Systems'.

Memory-2 Iterated Prisoner's Dilemma with payoffs R=1, T=1.2, S=-0.5, P=0.
Agent 0 = full observer. Agent 1 = partial observer (varied per condition).
CRLD (deterministic Expected-SARSA / actor-critic) learning dynamics via pyCRLD.

Fixes vs. the original notebook:
  * discount factor kept in [0,1] (original Case1 used discount_factors=10, invalid)
  * consistent learning_rate=0.1, discount_factor=0.9 for every condition
  * identical pipeline for every condition (no per-case drift)
"""
import os, sys, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded

RNG = np.random.default_rng(0)
np.random.seed(0)

# ----------------------------------------------------------------------------
# observation-set -> observation tensor (from the notebook, cleaned up)
# Each embedded state is a 2-step history; each step = "a0,a1,state".
# '.' marks an unobserved (masked) coordinate. States that look identical to an
# agent are aliased: the agent's observation row is uniform over the group.
# ----------------------------------------------------------------------------
def obs_matrix_from_obs_set(descriptions):
    n = len(descriptions)
    M = np.zeros((n, n))
    fully = []
    for d in descriptions:
        nstates = d.count("c") + d.count("d")
        fully.append(nstates == d.count(","))
    for i, di in enumerate(descriptions):
        for j, dj in enumerate(descriptions):
            if fully[i] and i == j:
                M[i, j] = 1
            elif not fully[i]:
                pi = di.replace("|", ",").split(",")
                pj = dj.replace("|", ",").split(",")
                if all((a == "." or b == "." or a == b) for a, b in zip(pi, pj)):
                    M[i, j] = 1
    M /= M.sum(1, keepdims=True)
    return M


# Full-observability reference observation set for agent (both coords seen).
FULL = [
    "c,c,.|c,c,.|", "c,c,.|c,d,.|", "c,c,.|d,c,.|", "c,c,.|d,d,.|",
    "c,d,.|c,c,.|", "c,d,.|c,d,.|", "c,d,.|d,c,.|", "c,d,.|d,d,.|",
    "d,c,.|c,c,.|", "d,c,.|c,d,.|", "d,c,.|d,c,.|", "d,c,.|d,d,.|",
    "d,d,.|c,c,.|", "d,d,.|c,d,.|", "d,d,.|d,c,.|", "d,d,.|d,d,.|",
]

# Agent-1 observation sets for each experimental condition (recent step masked).
OSET_A1 = {
    # Self-aware: keeps own action a1 in game2, masks agent0's action a0.
    "self_aware": [
        "c,c,.|.,c,.|", "c,c,.|.,d,.|", "c,c,.|.,c,.|", "c,c,.|.,d,.|",
        "c,d,.|.,c,.|", "c,d,.|.,d,.|", "c,d,.|.,c,.|", "c,d,.|.,d,.|",
        "d,c,.|.,c,.|", "d,c,.|.,d,.|", "d,c,.|.,c,.|", "d,c,.|.,d,.|",
        "d,d,.|.,c,.|", "d,d,.|.,d,.|", "d,d,.|.,c,.|", "d,d,.|.,d,.|",
    ],
    # Non-self-aware: keeps agent0's action a0 in game2, masks own action a1.
    "non_self_aware": [
        "c,c,.|c,.,.|", "c,c,.|c,.,.|", "c,c,.|d,.,.|", "c,c,.|d,.,.|",
        "c,d,.|c,.,.|", "c,d,.|c,.,.|", "c,d,.|d,.,.|", "c,d,.|d,.,.|",
        "d,c,.|c,.,.|", "d,c,.|c,.,.|", "d,c,.|d,.,.|", "d,c,.|d,.,.|",
        "d,d,.|c,.,.|", "d,d,.|c,.,.|", "d,d,.|d,.,.|", "d,d,.|d,.,.|",
    ],
    # Cooperation-focus: sees agent0 only when agent0 cooperated, else masked.
    "coop_focus": [
        "c,c,.|c,c,.|", "c,c,.|c,d,.|", "c,c,.|.,c,.|", "c,c,.|.,d,.|",
        "c,d,.|c,c,.|", "c,d,.|c,d,.|", "c,d,.|.,c,.|", "c,d,.|.,d,.|",
        "d,c,.|c,c,.|", "d,c,.|c,d,.|", "d,c,.|.,c,.|", "d,c,.|.,d,.|",
        "d,d,.|c,c,.|", "d,d,.|c,d,.|", "d,d,.|.,c,.|", "d,d,.|.,d,.|",
    ],
    # Defection-focus: sees agent0 only when agent0 defected, else masked.
    "def_focus": [
        "c,c,.|.,c,.|", "c,c,.|.,d,.|", "c,c,.|d,c,.|", "c,c,.|d,d,.|",
        "c,d,.|.,c,.|", "c,d,.|.,d,.|", "c,d,.|d,c,.|", "c,d,.|d,d,.|",
        "d,c,.|.,c,.|", "d,c,.|.,d,.|", "d,c,.|d,c,.|", "d,c,.|d,d,.|",
        "d,d,.|.,c,.|", "d,d,.|.,d,.|", "d,d,.|d,c,.|", "d,d,.|d,d,.|",
    ],
}


def make_agents(condition, lr=0.1, gamma=0.9):
    env = MultipleObsSocialDilemma(
        rewards=1, temptations=1.2, suckers_payoffs=-0.5, punishments=0,
        observation_value=[1, 1],
    )
    h = HistoryEmbedded(env, h=(2, 2, 2))
    if condition == "full":
        h.O[0] = obs_matrix_from_obs_set(FULL)
        h.O[1] = obs_matrix_from_obs_set(FULL)
    else:
        h.O[0] = obs_matrix_from_obs_set(FULL)            # agent0 always full
        h.O[1] = obs_matrix_from_obs_set(OSET_A1[condition])
    mae = POstratAC(env=h, learning_rates=lr, discount_factors=gamma)
    return mae


def run_condition(condition, n_sims, Tmax=4000, tol=1e-5):
    mae = make_agents(condition)
    # Prime the (non-jit) observation-distribution path once so the jitted Ri
    # can use the fast _jobsdist branch afterwards.
    mae.obsdist(mae.random_softmax_strategy())
    finalR, lengths, nconv = [], [], 0
    for _ in range(n_sims):
        x = mae.random_softmax_strategy()
        Xtisa, conv = mae.trajectory(x, Tmax=Tmax, tolerance=tol)
        if conv:
            nconv += 1
            finalR.append(np.asarray(mae.Ri(Xtisa[-1])))
            lengths.append(len(Xtisa))
    finalR = np.array(finalR) if finalR else np.zeros((0, 2))
    return dict(
        condition=condition, n_sims=n_sims, n_converged=nconv,
        finalR=finalR, lengths=np.array(lengths),
    )


if __name__ == "__main__":
    cond = sys.argv[1] if len(sys.argv) > 1 else "full"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    t0 = time.time()
    res = run_condition(cond, n)
    dt = time.time() - t0
    R = res["finalR"]
    print(f"\n[{cond}] sims={n} converged={res['n_converged']} time={dt:.1f}s")
    if len(R):
        print(f"  mean reward agent0={R[:,0].mean():.3f} agent1={R[:,1].mean():.3f}")
        print(f"  %coop>0.9 agent0={100*np.mean(R[:,0]>0.9):.1f} "
              f"agent1={100*np.mean(R[:,1]>0.9):.1f}")
