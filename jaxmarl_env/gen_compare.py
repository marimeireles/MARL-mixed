"""IPPO / A2C / IQL on several general-sum games — reward dynamics per algorithm.

Games (all general-sum, discrete, meaningful flat observations):
  coin_game              2-player social dilemma (the deep-RL IPD)
  MPE_simple_tag_v3      predator-prey (mixed/general-sum, 4 agents)
  MPE_simple_adversary   keep-away with an adversary (3 agents)
  MPE_simple_push_v3     push the other off the landmark (2 agents)

The STORM "in-the-matrix" arena games are general-sum too, but their observation
is a spatial grid that collapses to nothing when flattened — they need a CNN
encoder, which this feed-forward comparison does not have (logged as skipped).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gen_algos as G
from run_all import SkipEnv

RESULTS = os.path.join(os.path.dirname(__file__), "results")
GAMES = ["coin_game", "MPE_simple_tag_v3", "MPE_simple_adversary_v3", "MPE_simple_push_v3"]
COLORS = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}
CFG = dict(G.DEFAULT, TOTAL_TIMESTEPS=1_000_000, NUM_ENVS=64)

if __name__ == "__main__":
    curves = {}                       # (game, algo) -> dynamics
    for game in GAMES:
        for algo in G.ALGOS:
            try:
                d = G.train(game, algo, CFG)
                curves[(game, algo)] = d
                print(f"  {algo:5s} {game:24s}: {d[0]:+.3f} -> {d[-1]:+.3f}", flush=True)
            except SkipEnv as e:
                print(f"  {algo:5s} {game:24s}: skip ({e})", flush=True)
    np.save(os.path.join(RESULTS, "gen_compare.npy"),
            np.array({k: v for k, v in curves.items()}, dtype=object), allow_pickle=True)

    ncol = 2; nrow = (len(GAMES) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.4 * nrow), squeeze=False)
    for k, game in enumerate(GAMES):
        ax = axes[k // ncol][k % ncol]
        for algo in G.ALGOS:
            if (game, algo) in curves:
                y = curves[(game, algo)]
                ax.plot(np.arange(len(y)) * CFG["NUM_STEPS"] * CFG["NUM_ENVS"], y,
                        color=COLORS[algo], label=algo, lw=2)
        ax.set_title(game, fontsize=10); ax.set_xlabel("env steps")
        ax.set_ylabel("mean reward / step"); ax.legend(fontsize=8)
    fig.suptitle("IPPO vs A2C vs IQL on general-sum games", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(RESULTS, "gen_compare.png")
    fig.savefig(out, dpi=150); print("saved", out)
