"""Re-render the training figures from saved .npy data at high DPI (no retraining).
Run after sweeps finish; flow plots are regenerated separately by flowplots.py."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(__file__), "results")
DPI = 300
C = {"IPPO": "#0254a3", "A2C": "#2a8c5a", "IQL": "#d1495b"}


def _has(*f):
    return all(os.path.exists(os.path.join(R, x)) for x in f)


def line_sweep(npy, axes_npy, png, xlabel, title, logx=False):
    if not _has(npy, axes_npy):
        return
    M = np.load(os.path.join(R, npy))
    algos, xs = np.load(os.path.join(R, axes_npy), allow_pickle=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for ai, name in enumerate(algos):
        ax.plot(xs, M[ai], "o-", color=C.get(name, None), label=name, lw=2)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("final cooperation P(C)")
    ax.set_ylim(-0.03, 1.03); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, png), dpi=DPI); plt.close(fig)
    print("hi-res", png)


def algo_compare():
    if not _has("algo_final.npy", "algo_final_axes.npy", "algo_curves_full.npy"):
        return
    final = np.load(os.path.join(R, "algo_final.npy"))
    algos, regs = np.load(os.path.join(R, "algo_final_axes.npy"), allow_pickle=True)
    curves = np.load(os.path.join(R, "algo_curves_full.npy"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ai, name in enumerate(algos):
        a1.plot(np.arange(curves.shape[1]) * 128 * 64, curves[ai], color=C[name], label=name, lw=2)
    a1.set_xlabel("environment steps"); a1.set_ylabel("cooperation P(C)")
    a1.set_title("(a) Cooperation over training — full observability")
    a1.set_ylim(-0.03, 1.03); a1.legend()
    w = 0.25; xs = np.arange(len(regs))
    for ai, name in enumerate(algos):
        a2.bar(xs + (ai - 1) * w, final[ai], w, color=C[name], label=name)
    a2.set_xticks(xs); a2.set_xticklabels(list(regs), rotation=20, ha="right")
    a2.set_ylabel("final cooperation P(C)"); a2.set_ylim(0, 1.03)
    a2.set_title("(b) Final cooperation by regime & algorithm"); a2.legend()
    fig.suptitle("Algorithm comparison on the heterogeneous-observation IPD (N=2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(R, "algo_compare.png"), dpi=DPI); plt.close(fig)
    print("hi-res algo_compare.png")


def coop_vs_n():
    if not _has("coop_vs_n.npy", "coop_vs_n_axes.npy"):
        return
    M = np.load(os.path.join(R, "coop_vs_n.npy"))
    regimes, NS = np.load(os.path.join(R, "coop_vs_n_axes.npy"), allow_pickle=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"N={n}" for n in NS])
    ax.set_yticks(range(len(regimes))); ax.set_yticklabels(list(regimes))
    for i in range(len(regimes)):
        for j in range(len(NS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.5 else "black", fontsize=9)
    ax.set_title("Final cooperation rate (IPPO)\nobservability regime x group size")
    fig.colorbar(im, label="mean P(cooperate)"); fig.tight_layout()
    fig.savefig(os.path.join(R, "coop_vs_n.png"), dpi=DPI); plt.close(fig)
    print("hi-res coop_vs_n.png")


def gen_compare():
    f = os.path.join(R, "gen_compare.npy")
    if not os.path.exists(f):
        return
    curves = np.load(f, allow_pickle=True).item()
    games = sorted({g for g, _ in curves})
    algos = ["IPPO", "A2C", "IQL"]
    ncol = 2; nrow = (len(games) + 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.4 * nrow), squeeze=False)
    for k, game in enumerate(games):
        ax = axes[k // ncol][k % ncol]
        for algo in algos:
            if (game, algo) in curves:
                y = curves[(game, algo)]
                ax.plot(np.arange(len(y)) * 128 * 64, y, color=C[algo], label=algo, lw=2)
        ax.set_title(game, fontsize=10); ax.set_xlabel("env steps")
        ax.set_ylabel("mean reward / step"); ax.legend(fontsize=8)
    fig.suptitle("IPPO vs A2C vs IQL on general-sum games", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(R, "gen_compare.png"), dpi=DPI); plt.close(fig)
    print("hi-res gen_compare.png")


def overcooked():
    f = os.path.join(R, "overcooked_curves.npy")
    if not os.path.exists(f):
        return
    curves = np.load(f)
    algos = ["IPPO", "A2C", "IQL"]
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for ai, algo in enumerate(algos):
        y = curves[ai]
        ax.plot(np.arange(len(y)) * 256 * 64, y, color=C[algo], label=algo, lw=2)
    ax.set_xlabel("environment steps"); ax.set_ylabel("sparse reward / step (dishes)")
    ax.set_title("Overcooked (CNN, shaped-reward training) — IPPO vs A2C vs IQL"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(R, "overcooked_compare.png"), dpi=DPI); plt.close(fig)
    print("hi-res overcooked_compare.png")


def coop_portraits():
    import glob
    files = sorted(f for f in glob.glob(os.path.join(R, "coop_*.npy")) if "vs_n" not in f)
    if not files:
        return
    names = {"full": "Full obs", "blind": "Blind", "self": "Self only", "others": "Others only",
             "coop": "Coop-tracking", "def": "Def-tracking"}
    order = ["full", "others", "coop", "def", "self", "blind"]
    have = {os.path.basename(f)[5:-4]: np.load(f) for f in files}
    regs = [r for r in order if r in have] + [r for r in have if r not in order]
    ncol = 3; nrow = (len(regs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.3 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, r in enumerate(regs):
        t = have[r]; ax = axes[k // ncol][k % ncol]; ax.axis("on")
        ax.plot(t[:, 0], t[:, 1], lw=2, color="purple", alpha=0.9)
        ax.scatter(t[0, 0], t[0, 1], color="purple", marker="x", s=45)
        ax.scatter(t[-1, 0], t[-1, 1], color="purple", marker="o", s=45)
        ax.plot([0, 1], [0, 1], ls=":", c="gray", lw=0.7)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_title(names.get(r, r), fontsize=10)
        ax.set_xlabel("Agent 1  P(C)"); ax.set_ylabel("Agent 2  P(C)")
    fig.suptitle("Cooperation dynamics across observability regimes (IPD, N=2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(R, "coop_portraits.png"), dpi=DPI); plt.close(fig)
    print("hi-res coop_portraits.png")


if __name__ == "__main__":
    line_sweep("memory_sweep.npy", "memory_sweep_axes.npy", "memory_sweep.png",
               "memory length (rounds remembered)", "Cooperation vs memory length (full-obs IPD, N=2)")
    line_sweep("lr_sweep.npy", "lr_sweep_axes.npy", "lr_sweep.png",
               "learning rate", "Cooperation vs learning rate (full-obs IPD, N=2)", logx=True)
    algo_compare(); coop_vs_n(); gen_compare(); overcooked(); coop_portraits()
    print("done")
