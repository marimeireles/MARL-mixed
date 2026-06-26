"""The full matrix of CRLD observability flow grids.

One 6-panel grid per (game x algorithm x N-players): each panel = one of the six
Agent-1 observability regimes (full / self / others / coop / def / blind), drawn
as a CRLD flow field + trajectories, in the post-mutual-cooperation memory-1 state.

Saved as results/obsgrid_<game>_<algo>_N<N>.png. Runs on CPU (JAX_PLATFORMS=cpu).
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
from scratch_repro.flowplots import _obs_matrix, _oset_for   # reuse the regime helpers

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jaxmarl_env", "results"))

# (R, T, S, P) for each game
GAMES = {
    "ipd":      (("Prisoner's Dilemma", "(IPD)"),        dict(R=1, T=1.5, S=-0.5, P=0)),
    "harmony":  (("Harmony", "(C dominant)"),            dict(R=1, T=0.5, S=0.5,  P=0)),
    "staghunt": (("Stag Hunt", "(bistable)"),            dict(R=1, T=0.5, S=-0.5, P=0)),
    "snowdrift":(("Snowdrift / Chicken", "(anti-coord)"),dict(R=1, T=1.5, S=0.5,  P=0)),
    "coin":     (("coin_game-PD", "(R1 T2 S-2 P0)"),     dict(R=1, T=2,   S=-2,   P=0)),
    "arena":    (("STORM arena-PD", "(R3 T5 S0 P1)"),    dict(R=3, T=5,   S=0,    P=1)),
}
REGIMES = [("Full observability", "full"),
           ("Self-aware\n(sees own action)", "self"),
           ("Non-self-aware\n(sees other's action)", "others"),
           ("Cooperation-tracking\n(sees other iff it cooperated)", "coop"),
           ("Defection-tracking\n(sees other iff it defected)", "def"),
           ("Blind", "blind")]


def obs_grid(game_key, agent_cls=POstratAC, algo_tag="ac", algo_label="CRLD actor-critic"):
    (gname, gsub), pay = GAMES[game_key]
    x = ([0], [0], [0]); y = ([1], [0], [0])     # agent0 vs agent1 P(C) in the (c,c) state
    fig, axes = plt.subplots(2, 3, figsize=(4.4 * 3, 4.3 * 2))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    for k, (title, reg) in enumerate(REGIMES):
        ax0 = axes[k // 3][k % 3]
        env = MultipleObsSocialDilemma(rewards=pay["R"], temptations=pay["T"],
                                       suckers_payoffs=pay["S"], punishments=pay["P"],
                                       observation_value=[1, 1])
        memo = HistoryEmbedded(env, h=(1, 1, 1))
        memo.Oset[1] = _oset_for(reg, list(memo.Oset[0]))
        memo.O[1] = _obs_matrix(memo.Oset[1])
        mae = agent_cls(env=memo, learning_rates=0.1, discount_factors=0.9)
        ax = fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, NrRandom=24,
                                   flowarrow_points=np.linspace(0.01, 0.99, 9), axes=[ax0])
        for seed in range(2):
            np.random.seed(seed)
            xt, _ = mae.trajectory(mae.random_softmax_strategy(), Tmax=8000, tolerance=1e-5)
            fp.plot_trajectories([xt], x, y, cols=["purple"], axes=ax)
        ax0.set_title(title, fontsize=10)
        ax0.set_xlabel("Agent 0  P(cooperate)"); ax0.set_ylabel("Agent 1  P(cooperate)")
    fig.suptitle(f"{gname} {gsub}  |  {algo_label}  |  N=2  -  observability flow grid",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, f"obsgrid_{game_key}_{algo_tag}_N2.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[obsgrid] {game_key} / {algo_label} -> {os.path.basename(out)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ac"
    if which == "ac":
        for g in GAMES:
            obs_grid(g, POstratAC, "ac", "CRLD actor-critic")
    elif which == "sarsa":
        from pyCRLD.Agents.APOStrategySarsa import stratSARSA as POSARSA
        for g in GAMES:
            obs_grid(g, POSARSA, "sarsa", "CRLD SARSA")
    print("done")
