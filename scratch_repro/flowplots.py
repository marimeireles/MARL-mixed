"""CRLD flow plots of the cooperation dynamics (runs in the py3.8 pyCRLD env).

Reproduces the canonical flow-plot + trajectory + time-series figure, then
extends it to the heterogeneous-observability social dilemma.
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

OUT = os.path.join(os.path.dirname(__file__), "flowplots")
os.makedirs(OUT, exist_ok=True)


def canonical_example():
    """The user's reference example: full-observability social dilemma."""
    env = SocialDilemma(R=1.0, T=0.8, S=-0.5, P=0.0)
    mae = stratAC(env=env, learning_rates=0.1, discount_factors=0.9)
    np.random.seed(0)
    x0 = mae.random_softmax_strategy()
    xtraj, reached = mae.trajectory(x0, Tmax=10000, tolerance=1e-5)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    plt.subplots_adjust(wspace=0.3)
    x = ([0], [0], [0])
    y = ([1], [0], [0])
    ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False,
                               flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[axs[0]])
    fp.plot_trajectories([xtraj], x, y, cols=["purple"], axes=ax)
    ax[0].set_xlabel("Agent 0's cooperation probability")
    ax[0].set_ylabel("Agent 1's cooperation probability")
    ax[0].set_title("Flow plot")

    axs[1].plot(xtraj[:, 0, 0, 0], label="Agent 0", c="red")
    axs[1].plot(xtraj[:, 1, 0, 0], label="Agent 1", c="blue")
    axs[1].set_xlabel("Time steps")
    axs[1].set_ylabel("Cooperation probability")
    axs[1].legend(); axs[1].set_title("Trajectory")
    fig.savefig(os.path.join(OUT, "canonical.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[canonical] fixed point reached={reached}, "
          f"end coop=({xtraj[-1,0,0,0]:.2f},{xtraj[-1,1,0,0]:.2f})")


def payoff_sweep():
    """Flow plots of cooperation dynamics from cooperative to defective games."""
    games = [("Harmony  (T=0.8)", 0.8), ("Prisoner's Dilemma  (T=1.2)", 1.2),
             ("Strong PD  (T=1.5)", 1.5)]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    plt.subplots_adjust(wspace=0.3)
    x = ([0], [0], [0]); y = ([1], [0], [0])
    for k, (title, Tval) in enumerate(games):
        env = SocialDilemma(R=1.0, T=Tval, S=-0.5, P=0.0)
        mae = stratAC(env=env, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9),
                                   axes=[axs[k]])
        # a few trajectories from different starts
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(),
                                   Tmax=10000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        axs[k].set_title(title)
        axs[k].set_xlabel("Agent 0  P(cooperate)")
        axs[k].set_ylabel("Agent 1  P(cooperate)")
    fig.suptitle("Cooperation dynamics (CRLD flow + trajectories) across the dilemma axis",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "payoff_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[payoff_sweep] saved")


def observability_example():
    """Heterogeneous-observability social dilemma (full vs partial), nb-10 style."""
    from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
    from pyCRLD.Agents.POStrategyActorCritic import POstratAC
    conds = [("Full observability", [1, 1]), ("Partial observability", [0.6, 0.6])]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    plt.subplots_adjust(wspace=0.3)
    x = ([0], [0], [0]); y = ([1], [0], [0])
    for k, (title, ov) in enumerate(conds):
        env = MultipleObsSocialDilemma(rewards=1, temptations=1.5, suckers_payoffs=-0.5,
                                       punishments=0, observation_value=ov)
        mae = POstratAC(env=env, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9),
                                   NrRandom=16, axes=[axs[k]])
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(),
                                   Tmax=10000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        axs[k].set_title(title)
        axs[k].set_xlabel("Agent 0  P(cooperate)")
        axs[k].set_ylabel("Agent 1  P(cooperate)")
    fig.suptitle("Cooperation dynamics under observation heterogeneity", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "observability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[observability] saved")


def _obs_matrix(descriptions):
    """Observation tensor: a partial observer's row is uniform over the states
    that look identical to it (non-masked coordinates agree)."""
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


def _mask(label, pos):
    p = label.strip("|").split(",")
    p[pos] = "."
    return ",".join(p) + "|"


def memory_observability():
    """Memory-1 IPD: how Agent-1's observability reshapes the cooperation flow,
    in the state where both agents just cooperated (where reciprocity lives)."""
    from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
    from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
    from pyCRLD.Agents.POStrategyActorCritic import POstratAC

    regimes = [("Full observability", None),
               ("Self-aware\n(sees own action)", 0),       # mask a0 -> keep own a1
               ("Non-self-aware\n(sees other's action)", 1),  # mask a1 -> keep a0
               ("Blind", "blind")]
    fig, axs = plt.subplots(1, len(regimes), figsize=(4.2 * len(regimes), 4.2))
    plt.subplots_adjust(wspace=0.3)
    x = ([0], [0], [0]); y = ([1], [0], [0])     # state 0 = (c, c) last round
    for k, (title, reg) in enumerate(regimes):
        env = MultipleObsSocialDilemma(rewards=1, temptations=1.2, suckers_payoffs=-0.5,
                                       punishments=0, observation_value=[1, 1])
        memo = HistoryEmbedded(env, h=(1, 1, 1))
        base = list(memo.Oset[0])
        if reg == "blind":
            memo.Oset[1] = [".,.,.|" for _ in base]
            memo.O[1] = _obs_matrix(memo.Oset[1])
        elif reg is not None:
            memo.Oset[1] = [_mask(s, reg) for s in base]
            memo.O[1] = _obs_matrix(memo.Oset[1])
        mae = POstratAC(env=memo, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=24,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[axs[k]])
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=8000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        axs[k].set_title(title, fontsize=10)
        axs[k].set_xlabel("Agent 0  P(cooperate)")
        axs[k].set_ylabel("Agent 1  P(cooperate)")
    fig.suptitle("Cooperation dynamics after mutual cooperation, by Agent-1 observability "
                 "(memory-1 IPD)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "memory_observability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[memory_observability] saved")


def game_menagerie():
    """CRLD flow fields for the four canonical symmetric 2x2 games (R=1, P=0).
    Each game has a qualitatively different cooperation dynamic."""
    games = [
        ("Harmony\n(C dominant)",        dict(R=1, T=0.5, S=0.5, P=0)),
        ("Stag Hunt\n(bistable)",         dict(R=1, T=0.5, S=-0.5, P=0)),
        ("Snowdrift / Chicken\n(anti-coordination)", dict(R=1, T=1.5, S=0.5, P=0)),
        ("Prisoner's Dilemma\n(D dominant)", dict(R=1, T=1.5, S=-0.5, P=0)),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3))
    plt.subplots_adjust(wspace=0.3)
    x = ([0], [0], [0]); y = ([1], [0], [0])
    for k, (title, pay) in enumerate(games):
        env = SocialDilemma(**pay)
        mae = stratAC(env=env, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False,
                                   flowarrow_points=np.linspace(0.01, 0.99, 11), axes=[axs[k]])
        for seed in range(5):           # several trajectories to reveal the basins
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=10000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax, lws=[1.3])
        axs[k].set_title(title, fontsize=10)
        axs[k].set_xlabel("Agent 0  P(cooperate)")
        axs[k].set_ylabel("Agent 1  P(cooperate)")
    fig.suptitle("Cooperation dynamics across the four canonical 2x2 games (CRLD flow fields)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "game_menagerie.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[game_menagerie] saved")


if __name__ == "__main__":
    canonical_example()
    payoff_sweep()
    memory_observability()
    game_menagerie()
    print("saved to", OUT)
