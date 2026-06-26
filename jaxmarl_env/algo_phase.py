"""Algorithm dynamics as phase-plane paths.

Background (grey arrows): the CRLD flow field of the *memoryless* IPD — the
deterministic prediction, which points to mutual defection. Coloured paths: the
actual learning trajectories of IPPO / A2C / IQL on the memory-1 IPD, traced
through the (Agent-0 cooperation, Agent-1 cooperation) plane. The interesting
part is where the sampled algorithms *deviate* from the grey flow — value-based
IQL, using memory, escapes the defection basin the memoryless dynamics predict.

Runs on GPU; prepend the bundled ptxas to PATH (see slurm_sweep.sh) before
invoking.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCRLD.Agents.StrategyActorCritic import stratAC
from pyCRLD.Environments.SocialDilemma import SocialDilemma
from pyCRLD.Utils import FlowPlot as fp
import algorithms as A

RESULTS = os.path.join(os.path.dirname(__file__), "results")
COLORS = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}


def algo_phase_portraits(regime="full"):
    cfg = dict(A.CFG, TOTAL_TIMESTEPS=1_000_000, NUM_ENVS=64)
    trajs = {name: A.ALGOS[name](regime, cfg=cfg) for name in A.ALGOS}  # each (updates, 2)
    np.save(os.path.join(RESULTS, f"algo_phase_{regime}.npy"),
            np.array([trajs[n] for n in A.ALGOS]))

    env = SocialDilemma(R=1, T=1.2, S=-0.5, P=0)
    mae = stratAC(env=env, learning_rates=0.1, discount_factors=0.9)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    x = ([0], [0], [0]); y = ([1], [0], [0])
    fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, col="0.75",
                          flowarrow_points=np.linspace(0.05, 0.95, 11), axes=[ax])
    for name in A.ALGOS:
        t = trajs[name]
        ax.plot(t[:, 0], t[:, 1], color=COLORS[name], lw=2.5, label=name, alpha=0.9)
        ax.scatter(t[0, 0], t[0, 1], color=COLORS[name], marker="x", s=55, zorder=5)
        ax.scatter(t[-1, 0], t[-1, 1], color=COLORS[name], marker="o", s=55, zorder=5)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Agent 0  P(cooperate)"); ax.set_ylabel("Agent 1  P(cooperate)")
    ax.set_title("Algorithm dynamics over the CRLD flow field\n"
                 "(grey = memoryless deterministic prediction)", fontsize=11)
    ax.legend(loc="center right")
    fig.tight_layout()
    out = os.path.join(RESULTS, f"algo_phase_{regime}.png")
    fig.savefig(out, dpi=150); print("saved", out)
    for n in A.ALGOS:
        print(f"  {n}: ({trajs[n][0,0]:.2f},{trajs[n][0,1]:.2f}) -> "
              f"({trajs[n][-1,0]:.2f},{trajs[n][-1,1]:.2f})")


if __name__ == "__main__":
    algo_phase_portraits("full")
