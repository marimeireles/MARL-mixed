"""Combine per-env results/*.npz into one grid of learning-dynamics curves, and
print a coverage report (ok / skipped / failed) from the .status files.

Run after the SLURM array finishes:  python plot_all.py
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def coverage():
    rows = []
    for s in sorted(glob.glob(os.path.join(RESULTS, "*.status"))):
        name = os.path.basename(s)[:-7]
        rows.append((name, open(s).read().strip()))
    ok = [n for n, st in rows if st == "ok"]
    skip = [(n, st) for n, st in rows if st.startswith("skipped")]
    fail = [(n, st) for n, st in rows if st.startswith("failed")]
    print(f"=== coverage: {len(ok)} ok, {len(skip)} skipped, {len(fail)} failed ===")
    for n, st in skip:
        print(f"  skip {n}: {st.split(':', 1)[1]}")
    for n, st in fail:
        print(f"  FAIL {n}: {st.split(':', 1)[1][:80]}")
    return ok


def grid(ok):
    files = [os.path.join(RESULTS, f"{n}.npz") for n in ok]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("no successful runs to plot")
        return
    ncol = 4
    nrow = (len(files) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, f in enumerate(files):
        d = np.load(f)
        ax = axes[k // ncol][k % ncol]
        ax.axis("on")
        ax.plot(d["timesteps"], d["dynamics"], color="#0254a3")
        ax.set_title(f"{os.path.basename(f)[:-4]} (N={int(d['num_agents'])})",
                     fontsize=9)
        ax.set_xlabel("env steps", fontsize=8)
        ax.set_ylabel("reward/step", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("IPPO learning dynamics across JaxMARL environments", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(RESULTS, "all_dynamics.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}  ({len(files)} envs)")


if __name__ == "__main__":
    grid(coverage())
