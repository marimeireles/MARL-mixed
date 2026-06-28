"""Generate the two new paper figures into paper/figures/:
  fig_obs_flow_grid.png : heterogeneous IPD observability flow grid (Agent 1 full
                          observer vs Agent 2 partial observer), memory-2, actor-critic.
  fig_memory_basin.png  : cross-memory cooperative-basin growth (full observability,
                          IPD actor-critic, memory 1 -> 2 -> 4).
Payoffs match Table 1 of the paper: R=1, T=1.2, S=-0.5, P=0.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCRLD.Environments.MultipleObsSocialDilemma import MultipleObsSocialDilemma
from pyCRLD.Environments.HistoryEmbedding import HistoryEmbedded
from pyCRLD.Agents.POStrategyActorCritic import POstratAC
from pyCRLD.Utils import FlowPlot as fp
from scratch_repro.mem_obs_grids import _oset_for_mem, _obs_matrix, _allc_state

FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper", "figures"))
R, T, S, P = 1.0, 1.2, -0.5, 0.0
PANELS = [("Full observability\n(baseline)", "full"),
          ("Self-aware\n(sees own last action)", "self"),
          ("Non-self-aware\n(sees other's last action)", "others"),
          ("Cooperation-tracking\n(sees other iff it cooperated)", "coop"),
          ("Defection-tracking\n(sees other iff it defected)", "def"),
          ("Blind", "blind")]


def _ipd(memory):
    env = MultipleObsSocialDilemma(rewards=R, temptations=T, suckers_payoffs=S,
                                   punishments=P, observation_value=[1, 1])
    return HistoryEmbedded(env, h=(memory,) * 3)


def _mae_hetero(memory, reg):
    """Agent 0 = full observer; Agent 1 = partial (regime reg). N=2."""
    memo = _ipd(memory)
    base = list(memo.Oset[0])
    memo.Oset[0] = [_oset_for_mem("full", e, 0, 2) for e in base]
    memo.O[0] = _obs_matrix(memo.Oset[0])
    memo.Oset[1] = [_oset_for_mem(reg, e, 1, 2) for e in base]
    memo.O[1] = _obs_matrix(memo.Oset[1])
    return POstratAC(env=memo, learning_rates=0.1, discount_factors=0.9), _allc_state(memo)


def obs_flow_grid(memory=2):
    fig, axes = plt.subplots(2, 3, figsize=(4.3 * 3, 4.1 * 2))
    plt.subplots_adjust(wspace=0.32, hspace=0.4)
    for k, (title, reg) in enumerate(PANELS):
        ax0 = axes[k // 3][k % 3]
        mae, si = _mae_hetero(memory, reg)
        x = ([0], [si], [0]); y = ([1], [si], [0])
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=16,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[ax0])
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=6000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        ax0.set_title(title, fontsize=10)
        ax0.set_xlabel("Agent 1 (full observer)  P(cooperate)", fontsize=9)
        ax0.set_ylabel("Agent 2 (partial)  P(cooperate)", fontsize=9)
    fig.suptitle("Heterogeneous observability — learning flow in the Iterated Prisoner's Dilemma "
                 "(memory-2, actor-critic)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGS, "fig_obs_flow_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def memory_basin():
    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 4.3))
    for ax0, M in zip(axes, [1, 2, 4]):
        memo = _ipd(M)
        mae = POstratAC(env=memo, learning_rates=0.1, discount_factors=0.9)
        si = _allc_state(memo)
        x = ([0], [si], [0]); y = ([1], [si], [0])
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=16,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[ax0])
        for seed in range(3):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=6000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        ax0.set_title(f"memory = {M}", fontsize=12)
        ax0.set_xlabel("Agent 1  P(cooperate)", fontsize=9)
        ax0.set_ylabel("Agent 2  P(cooperate)", fontsize=9)
    fig.suptitle("Full observability: the cooperative basin emerges and grows with memory "
                 "(IPD, actor-critic)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGS, "fig_memory_basin.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def algorithm_compare(memory=2):
    """IPD full-observability flow under actor-critic vs SARSA (value-based)."""
    from pyCRLD.Agents.APOStrategySarsa import stratSARSA as POSARSA
    fig, axes = plt.subplots(1, 2, figsize=(4.7 * 2, 4.4))
    for ax0, (cls, lab) in zip(axes, [(POstratAC, "Actor-critic (policy gradient)"),
                                       (POSARSA, "SARSA (value-based)")]):
        memo = _ipd(memory)
        mae = cls(env=memo, learning_rates=0.1, discount_factors=0.9)
        si = _allc_state(memo)
        x = ([0], [si], [0]); y = ([1], [si], [0])
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=16,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[ax0])
        for seed in range(3):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=6000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        ax0.set_title(lab, fontsize=11)
        ax0.set_xlabel("Agent 1  P(cooperate)", fontsize=9)
        ax0.set_ylabel("Agent 2  P(cooperate)", fontsize=9)
    fig.suptitle(f"Learning rule and cooperation: full-observability IPD flow (memory-{memory})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIGS, "fig_algorithm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    obs_flow_grid(2)
    memory_basin()
    algorithm_compare(2)
    print("done")
