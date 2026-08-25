"""CLI for the LLM cooperation-dynamics experiments.

Run from the repo root with the jaxmarl venv:

  JAX_PLATFORMS=cpu .venv-jaxmarl/bin/python -m llm_dynamics.run_experiments \
      <command> [--api-base http://host:8000 --model NAME | --mock POLICY] ...

Commands
  donors-sweep        dyadic donors game (LLM vs set strategy) over a q x w
                      grid: per-round logs, cooperation-rate heatmaps, and
                      cooperation-plane portraits over the CRLD flow
  donors-probe        measure P(C | history state) under the donors framing
                      -> JSON (one file per model; compare across models
                      with reciprocity-figure)
  matrix-sweep        PD / Chicken / Stag Hunt / Harmony at memory m vs a
                      set strategy: logs + portraits over the CRLD flow
  matrix-probe        P(C | history state) under the matrix framing -> JSON
  reciprocity-figure  overlay one or more saved probes (e.g. base vs
                      RL-trained checkpoints) and played logs on the
                      fixed-opponent CRLD flow in the reciprocity plane
  demo                offline end-to-end run with mock models ("base" =
                      selfish, "trained" = reciprocal) producing every
                      figure type; sanity-checks the pipeline

Typical base-vs-trained comparison against served checkpoints:

  ... donors-probe --api-base http://h:8000 --model Qwen/Qwen3-8B \
        --out results/probe_base.json --b 4 --c-over-b 0.5 --w 0.75 --q 0.75
  ... donors-probe --api-base http://h:8001 --model qwen3-donor-grpo \
        --out results/probe_trained.json --b 4 --c-over-b 0.5 --w 0.75 --q 0.75
  ... reciprocity-figure --probes results/probe_base.json \
        results/probe_trained.json --opponent tit_for_tat
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import donors_game as dg
from . import matrix_games as mg
from . import policy_probe as pp
from .llm_client import make_client

RESULTS = Path(__file__).resolve().parent / "results"


def _add_client_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-base", default=None,
                   help="OpenAI-compatible endpoint (e.g. vLLM); omit for mock")
    p.add_argument("--model", default=None)
    p.add_argument("--api-key", default="dummy")
    p.add_argument("--mock", default="soft-tft",
                   help="mock policy when no --api-base "
                        "(base-selfish, trained-reciprocal, soft-tft, ...)")
    p.add_argument("--thinking", action="store_true",
                   help="enable model thinking mode (Qwen3)")
    p.add_argument("--no-think-suffix", dest="no_think", action="store_true",
                   default=True)
    p.add_argument("--with-think", dest="no_think", action="store_false",
                   help="drop the /no_think prompt suffix")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=512)


def _client(args, seed: int = 0):
    return make_client(args.api_base, args.model, args.mock,
                       api_key=args.api_key, enable_thinking=args.thinking,
                       seed=seed)


def _model_tag(args) -> str:
    if args.api_base:
        return args.model.replace("/", "_")
    return f"mock-{args.mock}"


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",")]


# ── donors-sweep ──────────────────────────────────────────────────────────

def cmd_donors_sweep(args) -> None:
    from . import donors_crld as dc
    from . import plots

    b = args.b
    c = args.b * args.c_over_b
    qs, ws = _floats(args.q_values), _floats(args.w_values)
    strategies = args.strategies.split(",")
    out_dir = RESULTS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _model_tag(args)

    summaries = []
    all_rows: dict[tuple, dict[int, list[dict]]] = {}
    for strat in strategies:
        for q in qs:
            for w in ws:
                cell: dict[int, list[dict]] = {}
                for seed in range(args.seeds):
                    client = _client(args, seed=seed)
                    res = dg.run_donors_game(
                        client, b=b, c=c, w=w, q=q,
                        num_rounds=args.rounds, opponent_strategy=strat,
                        seed=seed, swap_mode=args.swap_mode,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens, no_think=args.no_think)
                    cell[seed] = res["rows"]
                    summaries.append(res["summary"])
                    dg.write_jsonl(
                        out_dir / "rounds" /
                        f"{tag}_{strat}_q{q:g}_w{w:g}_s{seed}.jsonl",
                        res["rows"])
                all_rows[(strat, q, w)] = cell
                print(f"[donors-sweep] {tag} vs {strat} q={q:g} w={w:g}: "
                      f"coop={np.mean([s['cooperation_rate'] or 0 for s in summaries[-args.seeds:]]):.2f}")

    with open(out_dir / f"summary_{tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        wr.writeheader()
        wr.writerows(summaries)

    # Heatmaps: cooperation rate over q x w, Nowak threshold marked.
    for strat in strategies:
        vals = np.full((len(ws), len(qs)), np.nan)
        for iq, q in enumerate(qs):
            for iw, w in enumerate(ws):
                cs = [s["cooperation_rate"] for s in summaries
                      if s["opponent_strategy"] == strat
                      and s["q"] == q and s["w"] == w
                      and s["cooperation_rate"] is not None]
                if cs:
                    vals[iw, iq] = float(np.mean(cs))
        thr = None
        cb = c / b
        if min(qs) <= cb <= max(qs) and len(qs) > 1:
            thr = float(np.interp(cb, qs, np.arange(len(qs))))
        plots.sweep_heatmap(
            vals, qs, ws, "q (reputation availability)",
            "w (re-encounter probability)",
            f"donors game: {tag} vs {strat}  (b={b:g}, c/b={cb:g})",
            out_dir / f"heatmap_{tag}_{strat}.png", threshold_line=thr)

    # Cooperation-plane portraits over the CRLD flow, one per cell.
    if not args.skip_portraits:
        for (strat, q, w), cell in all_rows.items():
            memo = dc.donors_memo_env(b, c, memory=args.memory, q=q)
            mae = dc.build_mae(memo, w=w, algo=args.algo, q=q)
            si = dc.allc_state(memo)
            trajs = {f"seed {s}": rows for s, rows in cell.items()}
            plots.cooperation_portrait(
                mae, si, trajs,
                f"{tag} vs {strat} | b={b:g} c/b={c/b:g} w={w:g} q={q:g} | "
                f"CRLD {args.algo} flow (memory {args.memory})",
                out_dir / f"portrait_{tag}_{strat}_q{q:g}_w{w:g}.png",
                partner_label=strat, window=args.window)
    print(f"[donors-sweep] results in {out_dir}")


# ── matrix-sweep ──────────────────────────────────────────────────────────

def cmd_matrix_sweep(args) -> None:
    from . import donors_crld as dc
    from . import plots

    games = args.games.split(",")
    memories = [int(m) for m in args.memories.split(",")]
    strategies = args.strategies.split(",")
    out_dir = RESULTS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _model_tag(args)

    summaries = []
    for game in games:
        for memory in memories:
            for strat in strategies:
                cell = {}
                for seed in range(args.seeds):
                    client = _client(args, seed=seed)
                    res = mg.run_matrix_game(
                        client, game_key=game, memory=memory,
                        num_rounds=args.rounds, opponent_strategy=strat,
                        seed=seed, temperature=args.temperature,
                        max_tokens=args.max_tokens, no_think=args.no_think)
                    cell[seed] = res["rows"]
                    summaries.append(res["summary"])
                    mg.write_jsonl(
                        out_dir / "rounds" /
                        f"{tag}_{game}_m{memory}_{strat}_s{seed}.jsonl",
                        res["rows"])
                pay = mg.GAMES[game]
                memo = dc.build_memo_env(pay["R"], pay["T"], pay["S"],
                                         pay["P"], memory=memory)
                mae = dc.build_mae(memo, w=args.gamma, algo=args.algo)
                si = dc.allc_state(memo)
                trajs = {f"seed {s}": rows for s, rows in cell.items()}
                plots.cooperation_portrait(
                    mae, si, trajs,
                    f"{pay['label']} | {tag} vs {strat} | memory {memory} | "
                    f"CRLD {args.algo} flow",
                    out_dir / f"portrait_{tag}_{game}_m{memory}_{strat}.png",
                    partner_label=strat, window=args.window)
                coop = np.mean([s["cooperation_rate"] or 0
                                for s in summaries[-args.seeds:]])
                print(f"[matrix-sweep] {tag} {game} m={memory} vs {strat}: "
                      f"coop={coop:.2f}")

    with open(out_dir / f"summary_{tag}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        wr.writeheader()
        wr.writerows(summaries)
    print(f"[matrix-sweep] results in {out_dir}")


# ── probes ────────────────────────────────────────────────────────────────

def cmd_donors_probe(args) -> None:
    client = _client(args)
    probe = pp.probe_donors_policy(
        client, b=args.b, c=args.b * args.c_over_b, w=args.w, q=args.q,
        memory=args.memory, temperature=args.probe_temperature,
        samples=args.samples, no_think=args.no_think)
    out = Path(args.out) if args.out else \
        RESULTS / "probes" / f"donors_{_model_tag(args)}_m{args.memory}.json"
    pp.save_probe(out, probe)
    _print_probe(probe)
    print(f"[donors-probe] saved {out}")


def cmd_matrix_probe(args) -> None:
    client = _client(args)
    probe = pp.probe_matrix_policy(
        client, game_key=args.game, memory=args.memory,
        temperature=args.probe_temperature, samples=args.samples,
        no_think=args.no_think)
    out = Path(args.out) if args.out else \
        RESULTS / "probes" / f"{args.game}_{_model_tag(args)}_m{args.memory}.json"
    pp.save_probe(out, probe)
    _print_probe(probe)
    print(f"[matrix-probe] saved {out}")


def _print_probe(probe: dict) -> None:
    for label, entry in sorted(probe["states"].items()):
        p = entry["p_cooperate"]
        print(f"  P(C | {label}) = {'--' if p is None else f'{p:.3f}'}")


# ── reciprocity-figure ────────────────────────────────────────────────────

def cmd_reciprocity_figure(args) -> None:
    from . import donors_crld as dc
    from . import plots

    probes = [pp.load_probe(f) for f in args.probes]
    first = probes[0]
    memory = first["memory"]

    if first["kind"] == "donors":
        b, c, q = first["b"], first["c"], first["q"]
        w = args.w if args.w is not None else first["w"]
        memo = dc.donors_memo_env(b, c, memory=memory, q=q)
        subtitle = f"donors b={b:g} c/b={c/b:g} w={w:g} q={q:g}"
    else:
        pay = mg.GAMES[first["game"]]
        w = args.w if args.w is not None else 0.9
        memo = dc.build_memo_env(pay["R"], pay["T"], pay["S"], pay["P"],
                                 memory=memory)
        subtitle = f"{pay['label']} gamma={w:g}"
        q = 1.0
    mae = dc.build_mae(memo, w=w, algo=args.algo, q=q)

    X_opp = dc.strategy_policy(args.opponent, memo, agent=1)
    sx = dc.uniform_state(memo, "c", "c", memory)
    sy = dc.uniform_state(memo, "c", "d", memory)
    flow = dc.fixed_opponent_flow(mae, memo, X_opp, sx, sy,
                                  NrRandom=args.flow_samples)

    points = {}
    for path, probe in zip(args.probes, probes):
        label = Path(path).stem
        px = pp.uniform_state_p(probe, "c", "c")
        py = pp.uniform_state_p(probe, "c", "d")
        if px is not None and py is not None:
            points[label] = (px, py)

    trajs = {}
    for path in args.played or []:
        rows = dg.read_jsonl(path)
        trajs[Path(path).stem] = plots.conditional_coop_series(rows)

    out = Path(args.out) if args.out else \
        RESULTS / f"reciprocity_{args.opponent}.png"
    plots.reciprocity_portrait(
        flow, points, trajs,
        f"reciprocity plane vs {args.opponent} | {subtitle} | "
        f"CRLD {args.algo} flow", out,
        xlabel="P(C | sustained mutual cooperation)",
        ylabel="P(C | own C, partner D)")
    print(f"[reciprocity-figure] saved {out}")


# ── demo ──────────────────────────────────────────────────────────────────

def cmd_demo(args) -> None:
    """Offline mock pipeline: 'base' (selfish) vs 'trained' (reciprocal)."""
    import types

    base = types.SimpleNamespace(**vars(args))
    base.api_base = None

    print("== donors sweep: mock base model ==")
    b_args = argparse.Namespace(**{**vars(base), "mock": "base-selfish",
                                   "out": "demo_donors_base"})
    cmd_donors_sweep(b_args)
    print("== donors sweep: mock trained model ==")
    t_args = argparse.Namespace(**{**vars(base), "mock": "trained-reciprocal",
                                   "out": "demo_donors_trained"})
    cmd_donors_sweep(t_args)

    print("== probes ==")
    probe_files = []
    for mock in ("base-selfish", "trained-reciprocal"):
        p_args = argparse.Namespace(**{**vars(base), "mock": mock,
                                       "w": 0.75, "q": 0.75,
                                       "out": str(RESULTS / "demo_probes" /
                                                  f"{mock}.json")})
        cmd_donors_probe(p_args)
        probe_files.append(p_args.out)

    print("== reciprocity figure (base vs trained) ==")
    played = sorted((RESULTS / "demo_donors_base" / "rounds").glob(
        "*tit_for_tat_q0.75_w0.75_s0.jsonl"))
    played += sorted((RESULTS / "demo_donors_trained" / "rounds").glob(
        "*tit_for_tat_q0.75_w0.75_s0.jsonl"))
    r_args = argparse.Namespace(**{**vars(base), "probes": probe_files,
                                   "played": [str(p) for p in played],
                                   "opponent": "tit_for_tat", "w": 0.75,
                                   "flow_samples": 6,
                                   "out": str(RESULTS / "demo_reciprocity.png")})
    cmd_reciprocity_figure(r_args)

    print("== matrix sweep (mock trained model) ==")
    m_args = argparse.Namespace(**{**vars(base), "mock": "trained-reciprocal",
                                   "games": args.games,
                                   "memories": args.memories,
                                   "out": "demo_matrix"})
    cmd_matrix_sweep(m_args)
    print("[demo] done — see llm_dynamics/results/demo_*")


# ── argument wiring ───────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(prog="llm_dynamics",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def game_params(p):
        p.add_argument("--b", type=float, default=4.0)
        p.add_argument("--c-over-b", type=float, default=0.5)

    def crld_params(p):
        p.add_argument("--memory", type=int, default=1)
        p.add_argument("--algo", choices=["ac", "sarsa"], default="ac")

    p = sub.add_parser("donors-sweep")
    _add_client_args(p)
    game_params(p)
    crld_params(p)
    p.add_argument("--q-values", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--w-values", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--strategies", default="tit_for_tat,always_defect,always_cooperate")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--swap-mode", choices=["same", "pool"], default="same")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--skip-portraits", action="store_true")
    p.add_argument("--out", default="donors_sweep")
    p.set_defaults(func=cmd_donors_sweep)

    p = sub.add_parser("matrix-sweep")
    _add_client_args(p)
    crld_params(p)
    p.add_argument("--games", default="ipd,chicken,staghunt,harmony")
    p.add_argument("--memories", default="1,2")
    p.add_argument("--strategies", default="tit_for_tat")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.9,
                   help="CRLD discount for the flow background")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--out", default="matrix_sweep")
    p.set_defaults(func=cmd_matrix_sweep)

    p = sub.add_parser("donors-probe")
    _add_client_args(p)
    game_params(p)
    p.add_argument("--w", type=float, default=0.75)
    p.add_argument("--q", type=float, default=0.75)
    p.add_argument("--memory", type=int, default=1)
    p.add_argument("--probe-temperature", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_donors_probe)

    p = sub.add_parser("matrix-probe")
    _add_client_args(p)
    p.add_argument("--game", choices=list(mg.GAMES), default="ipd")
    p.add_argument("--memory", type=int, default=1)
    p.add_argument("--probe-temperature", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_matrix_probe)

    p = sub.add_parser("reciprocity-figure")
    p.add_argument("--probes", nargs="+", required=True)
    p.add_argument("--played", nargs="*", default=[])
    p.add_argument("--opponent", default="tit_for_tat")
    p.add_argument("--algo", choices=["ac", "sarsa"], default="ac")
    p.add_argument("--w", type=float, default=None,
                   help="override discount for the flow (default: probe's w)")
    p.add_argument("--flow-samples", type=int, default=8)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_reciprocity_figure)

    p = sub.add_parser("demo")
    _add_client_args(p)
    game_params(p)
    crld_params(p)
    p.add_argument("--q-values", default="0.25,0.75")
    p.add_argument("--w-values", default="0.25,0.75")
    p.add_argument("--strategies", default="tit_for_tat,always_defect")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--swap-mode", choices=["same", "pool"], default="same")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--skip-portraits", action="store_true")
    p.add_argument("--games", default="ipd,chicken")
    p.add_argument("--memories", default="1")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--probe-temperature", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=1)
    p.set_defaults(func=cmd_demo)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
