"""Does longer memory change cooperation? Sweep memory length (rounds the agents
condition on) for each algorithm on the full-observability IPD. Direct-reciprocity
theory predicts longer memory enables richer reciprocal strategies."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import algorithms as A

RESULTS = os.path.join(os.path.dirname(__file__), "results")
MEMS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ALGOS = list(A.ALGOS)
COLORS = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}

if __name__ == "__main__":
    M = np.zeros((len(ALGOS), len(MEMS)))
    for ai, name in enumerate(ALGOS):
        for mi, mem in enumerate(MEMS):
            cfg = dict(A.CFG, TOTAL_TIMESTEPS=1_000_000, NUM_ENVS=64, MEMORY=mem)
            t = A.ALGOS[name]("full", cfg=cfg)
            M[ai, mi] = t[-max(1, len(t)//10):].mean()
            print(f"  {name:5s} memory={mem}: coop={M[ai, mi]:.3f}", flush=True)
    np.save(os.path.join(RESULTS, "memory_sweep.npy"), M)
    np.save(os.path.join(RESULTS, "memory_sweep_axes.npy"),
            np.array([ALGOS, MEMS], dtype=object), allow_pickle=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for ai, name in enumerate(ALGOS):
        ax.plot(MEMS, M[ai], "o-", color=COLORS[name], label=name, lw=2)
    ax.set_xlabel("memory length (rounds remembered)")
    ax.set_ylabel("final cooperation P(C)"); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(MEMS)
    ax.set_title("Cooperation vs memory length (full-obs IPD, N=2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "memory_sweep.png"), dpi=150)
    print("saved", os.path.join(RESULTS, "memory_sweep.png"))
