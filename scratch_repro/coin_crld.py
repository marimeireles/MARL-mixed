"""CRLD "game dynamics diagrams" (learning-dynamics flow fields) for coin_game.

coin_game is a spatial, sequential social dilemma. Its ESSENTIAL strategic
structure, however, is a 2-action symmetric matrix game:

    COOPERATE = go pick up your OWN coin
    DEFECT    = go take the OTHER agent's coin

We represent that embedded social dilemma as a pyCRLD SocialDilemma(R, T, S, P)
and generate the flow-field game-dynamics diagram exactly the way
flowplots.canonical_example() / flowplots.memory_observability() do, so coin_game
slots straight into our analysis framework.

------------------------------------------------------------------------------
(R, T, S, P) MAPPING FROM coin_game's payoff matrix  [[1, 1, -2], [1, 1, -2]]
------------------------------------------------------------------------------
coin_game's raw payoffs (per coin event):
  * picking up your OWN colour coin      -> +1 to you            (cooperation)
  * picking up the OTHER agent's coin    -> +1 to you, -2 to them (defection / theft)

Reducing the spatial game to the canonical symmetric 2x2 dilemma (C = grab own
coin, D = grab the other's coin), a defensible mapping is:

  R = +1  (mutual cooperation): both agents collect their own coin, +1 each.
  T = +2  (temptation, lone defector): the unilateral defector both secures a
          coin and steals the opponent's, the best-case single-round spatial
          payoff "own +1 and steal +1" -> +2.
  S = -2  (sucker, the victim): the lone cooperator whose coin is stolen eats
          the -2 victim penalty -- coin_game's defining sucker payoff.
  P =  0  (mutual defection): both lunge for the other's coin; in the wash
          neither cleanly secures an own coin, normalised to a neutral 0 that
          sits below mutual cooperation.

This gives the strict Prisoner's-Dilemma ordering  T(2) > R(1) > P(0) > S(-2)
and satisfies the repeated-game condition 2R > T + S  (2 > 0), so mutual
cooperation is socially optimal yet individually unstable -- a genuine PD.

CAVEAT: this is the MATRIX REDUCTION of the spatial game, not an exact port.
A stricter book-keeping of mutual defection (each steals +1 while its own coin
is taken -2) would give P = -1; we use P = 0 (the value suggested for this study)
which is still a valid PD and keeps the diagram comparable to the canonical IPD
flow plots. The qualitative dynamics (flow toward mutual defection) are identical
either way.

Runs on CPU:
  JAX_PLATFORMS=cpu PYTHONPATH=. .venv-jaxmarl/bin/python scratch_repro/coin_crld.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCRLD.Agents.StrategyActorCritic import stratAC
from pyCRLD.Environments.SocialDilemma import SocialDilemma
from pyCRLD.Utils import FlowPlot as fp

# figures go to the shared results directory (jaxmarl_env/results/)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jaxmarl_env", "results"))
os.makedirs(OUT, exist_ok=True)

# coin_game embedded Prisoner's Dilemma (see module docstring for derivation)
COIN_PD = dict(R=1.0, T=2.0, S=-2.0, P=0.0)


def coin_flow():
    """(a) Flow field + trajectory + cooperation time-series for the coin-game PD."""
    env = SocialDilemma(**COIN_PD)
    mae = stratAC(env=env, learning_rates=0.1, discount_factors=0.9)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    plt.subplots_adjust(wspace=0.3)
    x = ([0], [0], [0])
    y = ([1], [0], [0])
    ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False,
                               flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[axs[0]])
    # a few trajectories from random starts to reveal the basin of attraction
    end_pts = []
    for seed in range(4):
        np.random.seed(seed)
        xt, reached = mae.trajectory(mae.random_softmax_strategy(),
                                     Tmax=10000, tolerance=1e-5)
        fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax, lws=[1.3])
        end_pts.append((xt[-1, 0, 0, 0], xt[-1, 1, 0, 0]))
    ax[0].set_xlabel("Agent 0  P(grab own coin)  [cooperate]")
    ax[0].set_ylabel("Agent 1  P(grab own coin)  [cooperate]")
    ax[0].set_title("coin_game PD: CRLD flow field")

    # time-series of the last trajectory
    axs[1].plot(xt[:, 0, 0, 0], label="Agent 0", c="red")
    axs[1].plot(xt[:, 1, 0, 0], label="Agent 1", c="blue")
    axs[1].set_xlabel("CRLD time steps")
    axs[1].set_ylabel("P(cooperate) = P(grab own coin)")
    axs[1].set_ylim(-0.03, 1.03)
    axs[1].legend(); axs[1].set_title("Cooperation trajectory")
    fig.suptitle("coin_game as an embedded Prisoner's Dilemma  "
                 "(R=1, T=2, S=-2, P=0)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "coin_crld_flow.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[coin_flow] saved -> coin_crld_flow.png ; trajectory endpoints "
          f"(coop0,coop1) = {[(round(a,3),round(b,3)) for a,b in end_pts]}")


# ---- observability grid (mirrors flowplots.memory_observability) -------------
def _obs_matrix(descriptions):
    n = len(descriptions)
    M = np.zeros((n, n))
    parts = [d.strip("|").split(",") for d in descriptions]
    fully = [all(p != "." for p in pp) for pp in parts]
    for i in range(n):
        if fully[i]:
            M[i, i] = 1.0
        else:
            for j in range(n):
                if all(a == "." or a == b for a, b in zip(parts[i], parts[j])):
                    M[i, j] = 1.0
    return M / M.sum(1, keepdims=True)


def _oset_for(regime, base):
    out = []
    for s in base:
        p = s.strip("|").split(",")            # [a0, a1, state]
        a0 = p[0]
        if regime == "full":      q = p[:]
        elif regime == "self":    q = [".", p[1], p[2]]
        elif regime == "others":  q = [p[0], ".", p[2]]
        elif regime == "blind":   q = [".", ".", "."]
        elif regime == "coop":    q = p[:] if a0 == "c" else [".", p[1], p[2]]
        elif regime == "def":     q = p[:] if a0 == "d" else [".", p[1], p[2]]
        else: raise ValueError(regime)
        out.append(",".join(q) + "|")
    return out


def coin_obs_grid():
    """(b) Six-regime observability flow grid for the memory-1 coin-game PD.

    Same machinery as flowplots.memory_observability(), but the embedded dilemma
    uses the coin_game payoffs (R=1, T=2, S=-2, P=0). The plotted state is "both
    agents just grabbed their own coin" (mutual cooperation), where reciprocity
    is decided.
    """
    from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
    from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
    from pyCRLD.Agents.POStrategyActorCritic import POstratAC

    regimes = [("Full observability", "full"),
               ("Self-aware\n(sees own action)", "self"),
               ("Non-self-aware\n(sees other's action)", "others"),
               ("Cooperation-tracking\n(sees other iff it grabbed own coin)", "coop"),
               ("Defection-tracking\n(sees other iff it stole)", "def"),
               ("Blind", "blind")]
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.3 * nrow))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    x = ([0], [0], [0]); y = ([1], [0], [0])      # state 0 = (cooperate, cooperate) last round
    for k, (title, reg) in enumerate(regimes):
        ax0 = axes[k // ncol][k % ncol]
        env = MultipleObsSocialDilemma(rewards=COIN_PD["R"], temptations=COIN_PD["T"],
                                       suckers_payoffs=COIN_PD["S"], punishments=COIN_PD["P"],
                                       observation_value=[1, 1])
        memo = HistoryEmbedded(env, h=(1, 1, 1))
        memo.Oset[1] = _oset_for(reg, list(memo.Oset[0]))
        memo.O[1] = _obs_matrix(memo.Oset[1])
        mae = POstratAC(env=memo, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=24,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[ax0])
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=8000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        ax0.set_title(title, fontsize=10)
        ax0.set_xlabel("Agent 0  P(grab own coin)")
        ax0.set_ylabel("Agent 1  P(grab own coin)")
    fig.suptitle("coin_game PD: cooperation flow field by Agent-1 observability "
                 "regime (memory-1, after mutual cooperation)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "coin_crld_obs.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[coin_obs_grid] saved -> coin_crld_obs.png (6 regimes)")


if __name__ == "__main__":
    coin_flow()
    coin_obs_grid()
    print("done; figures in", OUT)
