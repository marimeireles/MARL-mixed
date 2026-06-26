"""IPPO / A2C / IQL on Overcooked (CNN encoder, shaped-reward training).
Logs the *sparse* task reward (dishes delivered) per update."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cnn_algos as C
from run_all import DEFAULT

RESULTS = os.path.join(os.path.dirname(__file__), "results")
COLORS = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}

if __name__ == "__main__":
    cfg = dict(DEFAULT, TOTAL_TIMESTEPS=5_000_000, NUM_ENVS=64, NUM_STEPS=256,
               SHAPED_COEF=1.0)
    curves = {}
    for algo in C.ALGOS:
        d = C.train("overcooked", algo, cfg)
        curves[algo] = d
        print(f"  {algo:5s} overcooked: sparse reward {d[0]:+.3f} -> {d[-1]:+.3f} "
              f"(max {d.max():+.3f})", flush=True)
    np.save(os.path.join(RESULTS, "overcooked_curves.npy"),
            np.array([curves[a] for a in C.ALGOS]))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for algo in C.ALGOS:
        y = curves[algo]
        ax.plot(np.arange(len(y)) * cfg["NUM_STEPS"] * cfg["NUM_ENVS"], y,
                color=COLORS[algo], label=algo, lw=2)
    ax.set_xlabel("environment steps"); ax.set_ylabel("sparse reward / step (dishes)")
    ax.set_title("Overcooked (CNN, shaped-reward training) — IPPO vs A2C vs IQL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "overcooked_compare.png"), dpi=150)
    print("saved", os.path.join(RESULTS, "overcooked_compare.png"))
