"""Systematic flow-field figures: the CRLD arrow fields of every game /
parameter setting, with the measured LLM trajectories from the logged
sweeps overlaid. Pure plotting from results/ — no LLM calls.

  JAX_PLATFORMS=cpu .venv-jaxmarl/bin/python -m llm_dynamics.flow_grids [--tag qwen32b_base]

Figures (results/flow_grids/):
  theory_donors_m{1,2}.png          two-agent CRLD flow of the donors game over (w, q)
  theory_matrix_m{1,2,3}.png        two-agent CRLD flow of PD/Chicken/StagHunt/Harmony
  theory_reciprocity_donors.png     fixed-opponent flow per strategy (reciprocity plane)
                                    + probed base policy (star)
  donors_<strategy>_<mem>.png       (w, q) grid, LLM paths over the matching CRLD flow
  matrix_<game>_m<m>.png            strategy grid, LLM paths over the matching CRLD flow
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import donors_crld as dc
from . import plots
from . import policy_probe as pp
from .matrix_games import GAMES
from .strategies import STRATEGIES

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "flow_grids"
WS = QS = [0.0, 0.5, 1.0]
B, C = 4.0, 2.0
STRATS = [s for s in STRATEGIES if s != "random"] + ["random"]


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _panel(ax, mae, si, sets, title, partner, window=8, NrRandom=8):
    plots.crld_flow_background(ax, mae, si, NrRandom=NrRandom, n_points=8)
    cmap = plt.get_cmap("tab10")
    for k, (label, rows) in enumerate(sets.items()):
        xs = plots.sliding_p_coop(rows, window)
        ys = plots.sliding_coop(rows, key="opp_action", window=window)
        plots.plot_measured_trajectory(ax, xs, ys, color=cmap(k % 10), lw=1.8)
    ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("model P(C)", fontsize=7); ax.set_ylabel(f"{partner} P(C)", fontsize=7)
    ax.tick_params(labelsize=6)


def theory_donors(memory):
    fig, axs = plt.subplots(3, 3, figsize=(11, 10.5))
    for i, w in enumerate(WS):
        for j, q in enumerate(QS):
            memo = dc.donors_memo_env(B, C, memory=memory, q=q)
            mae = dc.build_mae(memo, w=w, q=q)
            _panel(axs[i][j], mae, dc.allc_state(memo), {}, f"w={w:g}  q={q:g}", "partner")
    fig.suptitle(f"CRLD flow of the donors game (b={B:g}, c/b={C/B:g}), memory {memory}: "
                 f"rows w (=γ), cols q (=observability)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); out = OUT / f"theory_donors_m{memory}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)


def theory_matrix(memory, gamma=0.9):
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (g, pay) in zip(axs, GAMES.items()):
        memo = dc.build_memo_env(pay["R"], pay["T"], pay["S"], pay["P"], memory=memory)
        mae = dc.build_mae(memo, w=gamma)
        _panel(ax, mae, dc.allc_state(memo), {}, f"{pay['label']} (R,T,S,P)=({pay['R']:g},{pay['T']:g},{pay['S']:g},{pay['P']:g})", "partner")
    fig.suptitle(f"CRLD flow of the four games, memory {memory}, γ={gamma}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); out = OUT / f"theory_matrix_m{memory}.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)


def theory_reciprocity(tag, w=0.75, q=0.75, memory=1):
    strats = [s for s in STRATS if s not in ("random",)]
    fig, axs = plt.subplots(2, 4, figsize=(15, 7.5))
    memo = dc.donors_memo_env(B, C, memory=memory, q=q)
    mae = dc.build_mae(memo, w=w, q=q)
    sx, sy = dc.uniform_state(memo, "c", "c", memory), dc.uniform_state(memo, "c", "d", memory)
    probe = None
    pf = RESULTS / "probes" / f"{tag}_donors_m{memory}_w{w:g}_q{q:g}.json"
    if pf.exists():
        probe = pp.load_probe(pf)
    for ax, s in zip(axs.flat, strats):
        try:
            X_opp = dc.strategy_policy(s, memo, agent=1)
        except ValueError:
            ax.set_visible(False); continue
        XX, YY, dX, dY = dc.fixed_opponent_flow(mae, memo, X_opp, sx, sy, NrRandom=6)
        mdx, mdy = dX.mean(-1), dY.mean(-1); L = np.sqrt(mdx**2 + mdy**2) + 1e-12
        ax.quiver(XX, YY, mdx*np.sqrt(L)/L, mdy*np.sqrt(L)/L, color="0.6", angles="xy")
        if probe:
            px, py = pp.uniform_state_p(probe, "c", "c"), pp.uniform_state_p(probe, "c", "d")
            if px is not None:
                ax.scatter([px], [py], marker="*", s=220, color="tab:red", edgecolor="k", zorder=5)
        ax.set_xlim(-.03, 1.03); ax.set_ylim(-.03, 1.03); ax.set_title(s, fontsize=9)
        ax.set_xlabel("P(C | mutual C)", fontsize=7); ax.set_ylabel("P(C | own C, partner D)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Fixed-opponent CRLD flow in the reciprocity plane (donors b={B:g} c/b={C/B:g} w={w} q={q}, memory {memory}); "
                 f"★ = probed {tag} policy", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); out = OUT / "theory_reciprocity_donors.png"
    fig.savefig(out, dpi=150); plt.close(fig); print("saved", out)


def donors_grid(tag, strat, mem_tag):
    d = RESULTS / f"{tag}_donors_v2" / strat / "rounds"
    fig, axs = plt.subplots(3, 3, figsize=(11, 10.5))
    cm = 1 if mem_tag == "full" else min(int(mem_tag[1:]), 2)
    for i, w in enumerate(WS):
        for j, q in enumerate(QS):
            files = sorted(glob.glob(str(d / f"*_q{q:g}_w{w:g}_{mem_tag}_s*.jsonl")))
            sets = {f"seed {k}": _rows(f) for k, f in enumerate(files)}
            memo = dc.donors_memo_env(B, C, memory=cm, q=q)
            mae = dc.build_mae(memo, w=w, q=q)
            _panel(axs[i][j], mae, dc.allc_state(memo), sets, f"w={w:g}  q={q:g}", strat)
    fig.suptitle(f"{tag} vs {strat} — LLM memory {mem_tag} — paths over the CRLD flow (memory {cm})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); out = OUT / f"donors_{strat}_{mem_tag}.png"
    fig.savefig(out, dpi=140); plt.close(fig); print("saved", out)


def matrix_grid(tag, game, memory, gamma=0.9):
    pay = GAMES[game]
    memo = dc.build_memo_env(pay["R"], pay["T"], pay["S"], pay["P"], memory=memory)
    mae = dc.build_mae(memo, w=gamma)
    si = dc.allc_state(memo)
    fig, axs = plt.subplots(3, 3, figsize=(11, 10.5))
    for ax, strat in zip(axs.flat, STRATS):
        d = RESULTS / f"{tag}_matrix_v2" / strat / "rounds"
        files = sorted(glob.glob(str(d / f"*_{game}_m{memory}_{strat}_s*.jsonl")))
        sets = {f"seed {k}": _rows(f) for k, f in enumerate(files)}
        _panel(ax, mae, si, sets, strat, strat)
    fig.suptitle(f"{pay['label']} — {tag} — memory {memory} — LLM paths over the CRLD flow (γ={gamma})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); out = OUT / f"matrix_{game}_m{memory}.png"
    fig.savefig(out, dpi=140); plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="qwen32b_base")
    ap.add_argument("--only", default=None, help="theory|donors|matrix")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "theory"):
        for m in (1, 2):
            theory_donors(m)
        for m in (1, 2, 3):
            theory_matrix(m)
        theory_reciprocity(a.tag)
    if a.only in (None, "donors"):
        for strat in STRATS:
            for mem_tag in ("full", "m1", "m2"):
                donors_grid(a.tag, strat, mem_tag)
    if a.only in (None, "matrix"):
        for game in GAMES:
            for m in (1, 2, 3):
                matrix_grid(a.tag, game, m)


if __name__ == "__main__":
    main()
