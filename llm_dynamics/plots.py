"""Phase portraits: CRLD flow backgrounds + measured LLM trajectories.

Follows the MARL-mixed deep-RL convention (jaxmarl_env/algo_phase.py):
grey arrows are the deterministic CRLD prediction, coloured lines are the
sampled reality of a learner (here: an LLM) moving through the same
plane; 'x' marks the start, 'o' the end. The deviation between the two
is the finding.

Two planes are used:
  * cooperation plane — (model P(C), partner P(C)), sliding-window
    realized frequencies over a two-agent CRLD flow;
  * reciprocity plane — (P(C | sustained mutual C), P(C | own C, partner
    D)), the learner's conditional policy over a fixed-opponent flow;
    probed LLM policies are points in this plane.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyCRLD.Utils import FlowPlot as fp


# ── Series extraction from logged rows ────────────────────────────────────

def sliding_coop(rows: list[dict], key: str = "model_action",
                 window: int = 8) -> np.ndarray:
    """Sliding-window realized P(cooperate) from logged rounds."""
    acts = [1.0 if r[key] == "COOPERATE" else 0.0 for r in rows]
    out = []
    for t in range(len(acts)):
        lo = max(0, t - window + 1)
        out.append(float(np.mean(acts[lo:t + 1])))
    return np.array(out)


def sliding_p_coop(rows: list[dict], window: int = 8) -> np.ndarray:
    """Sliding-window mean of the continuous p_cooperate readout (falls
    back to the binary action when logprobs were unavailable)."""
    vals = []
    for r in rows:
        p = r.get("p_cooperate")
        if p is None:
            p = 1.0 if r["model_action"] == "COOPERATE" else 0.0
        vals.append(float(p))
    out = []
    for t in range(len(vals)):
        lo = max(0, t - window + 1)
        out.append(float(np.mean(vals[lo:t + 1])))
    return np.array(out)


def conditional_coop_series(rows: list[dict], window: int = 12
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window estimates of (P(C | partner cooperated last round),
    P(C | partner defected last round)) — the realized trajectory in the
    reciprocity plane. Uses the continuous p_cooperate when available;
    windows with no matching rounds carry the previous estimate."""
    xs, ys = [], []
    px, py = 0.5, 0.5
    for t in range(len(rows)):
        lo = max(0, t - window + 1)
        win = rows[lo:t + 1]
        after_c, after_d = [], []
        for r in win:
            p = r.get("p_cooperate")
            if p is None:
                p = 1.0 if r["model_action"] == "COOPERATE" else 0.0
            prev = r.get("prev_opp")
            if prev is None and r.get("visible_state") not in (None, "start"):
                prev = ("COOPERATE" if r["visible_state"].split("|")[-1][1] == "c"
                        else "DEFECT")
            if prev == "COOPERATE":
                after_c.append(p)
            elif prev == "DEFECT":
                after_d.append(p)
        if after_c:
            px = float(np.mean(after_c))
        if after_d:
            py = float(np.mean(after_d))
        xs.append(px)
        ys.append(py)
    return np.array(xs), np.array(ys)


# ── Portrait building blocks ──────────────────────────────────────────────

def crld_flow_background(ax, mae, si: int, NrRandom: int = 12,
                         n_points: int = 9, col="0.75") -> None:
    """Two-agent CRLD flow in the cooperation plane at state si.
    col="0.75" = grey background; col="LEN" = arrows coloured by flow
    magnitude (viridis, the FlowPlot default)."""
    x = ([0], [si], [0])
    y = ([1], [si], [0])
    fp.plot_strategy_flow(mae, x, y, use_RPEarrows=False, col=col,
                          NrRandom=NrRandom,
                          flowarrow_points=np.linspace(0.03, 0.97, n_points),
                          axes=[ax])


def plot_measured_trajectory(ax, xs, ys, color="purple", label=None,
                             lw=2.2) -> None:
    ax.plot(xs, ys, color=color, lw=lw, label=label, zorder=5)
    ax.scatter([xs[0]], [ys[0]], marker="x", s=70, color=color, zorder=6)
    ax.scatter([xs[-1]], [ys[-1]], marker="o", s=55, color=color, zorder=6)


def _finish_axes(ax, xlabel, ylabel, title):
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)


def cooperation_portrait(mae, si: int, traj_sets: dict[str, list[dict]],
                         title: str, out: str | Path,
                         partner_label: str = "partner",
                         window: int = 8, colors: Optional[dict] = None,
                         NrRandom: int = 12) -> Path:
    """CRLD two-agent flow + measured (model, partner) cooperation paths.

    traj_sets: {legend label: rows} — e.g. one entry per model/seed."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    crld_flow_background(ax, mae, si, NrRandom=NrRandom)
    cmap = plt.get_cmap("tab10")
    for k, (label, rows) in enumerate(traj_sets.items()):
        col = (colors or {}).get(label, cmap(k % 10))
        xs = sliding_p_coop(rows, window)
        ys = sliding_coop(rows, key="opp_action", window=window)
        plot_measured_trajectory(ax, xs, ys, color=col, label=label)
    _finish_axes(ax, "model  P(cooperate)", f"{partner_label}  P(cooperate)",
                 title)
    ax.legend(fontsize=8, loc="best")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def reciprocity_portrait(flow_data, probe_points: dict[str, tuple[float, float]],
                         traj_sets: dict[str, tuple[np.ndarray, np.ndarray]],
                         title: str, out: str | Path,
                         xlabel: str = "P(C | sustained mutual cooperation)",
                         ylabel: str = "P(C | own C, partner D)",
                         colors: Optional[dict] = None) -> Path:
    """Fixed-opponent CRLD flow + probed LLM policies (points) + realized
    conditional-cooperation paths (lines).

    flow_data: (XX, YY, dX, dY) from donors_crld.fixed_opponent_flow, or
    None for no background."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    if flow_data is not None:
        XX, YY, dX, dY = flow_data
        mdx, mdy = dX.mean(-1), dY.mean(-1)
        length = np.sqrt(mdx ** 2 + mdy ** 2)
        scale = np.power(length + 1e-12, 0.5) / (length + 1e-12)
        ax.quiver(XX, YY, mdx * scale, mdy * scale, color="0.75",
                  angles="xy", zorder=1)
    cmap = plt.get_cmap("tab10")
    for k, (label, (xs, ys)) in enumerate(traj_sets.items()):
        col = (colors or {}).get(label, cmap(k % 10))
        plot_measured_trajectory(ax, xs, ys, color=col, label=label)
    for k, (label, (px, py)) in enumerate(probe_points.items()):
        col = (colors or {}).get(label, cmap((k + len(traj_sets)) % 10))
        ax.scatter([px], [py], marker="*", s=260, color=col,
                   edgecolor="black", linewidth=0.6, zorder=7, label=label)
    _finish_axes(ax, xlabel, ylabel, title)
    ax.legend(fontsize=8, loc="best")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def sweep_heatmap(values: np.ndarray, xticks, yticks, xlabel, ylabel,
                  title, out: str | Path, threshold_line=None) -> Path:
    """Cooperation-rate heatmap over a 2-D parameter sweep (e.g. q x w),
    optionally with the Nowak threshold q = c/b marked."""
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(values, origin="lower", vmin=0, vmax=1, cmap="viridis",
                   aspect="auto")
    ax.set_xticks(range(len(xticks)), [f"{v:g}" for v in xticks])
    ax.set_yticks(range(len(yticks)), [f"{v:g}" for v in yticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    for (j, i), v in np.ndenumerate(values):
        if not math.isnan(v):
            ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 else "black", fontsize=8)
    if threshold_line is not None:
        ax.axvline(threshold_line, color="red", ls="--", lw=1.2,
                   label="Nowak threshold q = c/b")
        ax.legend(fontsize=8)
    fig.colorbar(im, ax=ax, label="model cooperation rate")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out
