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
        for seed in range(4):
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
        for seed in range(4):
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


if __name__ == "__main__":
    canonical_example()
    payoff_sweep()
    # observability_example()  # needs memory-embedded env (single-state obs is degenerate)
    print("saved to", OUT)
