#!/usr/bin/env python3
"""Run the DiG-bench public game set across both model arms, N repetitions each.

One subprocess per (arm, game, rep) invoking the stock baseline harness, with a
bounded concurrency pool pointed at a single vLLM server. Resumable: a run whose
JSONL already carries a `summary` row is skipped, so the sweep can be restarted
after a server restart or a Slurm timeout without redoing finished work.

Failed runs are kept, not deleted -- the report's protocol counts them as data
(with their stop_reason), so the notebook can account for them explicitly.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HARNESS = Path("/nas/ucb/marimeireles/MARL-mixed/external_benchmarks/dig-bench/baseline-harness")
PYTHON = Path("/nas/ucb/marimeireles/MARL-mixed/external_benchmarks/../.venv-bench/bin/python")
RUNS = Path(os.environ.get("DIG_RUNS", "/nas/ucb/marimeireles/MARL-mixed/external_benchmarks/runs/digbench/runs"))
API = "https://api.digbench.ai"

# arm label -> served model id. Override with DIG_ARMS='{"base":"qwen3-8b-bench"}'.
ARMS = json.loads(os.environ.get("DIG_ARMS", '{"base": "qwen3-8b-bench"}'))

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_games(token: str) -> list[str]:
    req = urllib.request.Request(
        f"{API}/api/agent/games",
        headers={"Authorization": f"Bearer {token}",
                 "User-Agent": "digbench-baseline-harness/1.0 (+https://digbench.ai)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    if isinstance(data, dict):
        data = data.get("games", data.get("data", []))
    return [g if isinstance(g, str) else g.get("name") for g in data]


def is_complete(run_dir: Path) -> bool:
    """True iff a finished run (summary row) already exists in this dir."""
    for path in run_dir.glob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if '"type": "summary"' in line or '"type":"summary"' in line:
                        return True
        except OSError:
            continue
    return False


def run_one(arm: str, model: str, game: str, rep: int, args) -> dict:
    run_dir = RUNS / arm / game / f"rep{rep}"
    if is_complete(run_dir):
        log(f"SKIP  {arm} {game} rep{rep} (already complete)")
        return {"arm": arm, "game": game, "rep": rep, "status": "skipped"}

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON), "-m", "core.cli", game,
        "--provider", "sglang",
        "--base-url", args.base_url,
        "--model", model,
        "--move-channel", args.move_channel,
        "--max-tokens", str(args.max_tokens),
        "--max-steps", str(args.max_steps),
        "--run-dir", str(run_dir),
        "--run-label", f"{arm}-{game}-rep{rep}",
    ]
    started = time.time()
    log(f"START {arm} {game} rep{rep}")
    with (run_dir / "console.log").open("w", encoding="utf-8") as out:
        proc = subprocess.run(
            cmd, cwd=HARNESS, stdout=out, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            timeout=args.run_timeout,
        )
    dur = time.time() - started

    summary = None
    for path in run_dir.glob("*.jsonl"):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("type") == "summary":
                    summary = row
    beaten = summary.get("levels_beaten") if summary else None
    stop = summary.get("stop_reason") if summary else f"no-summary(rc={proc.returncode})"
    log(f"DONE  {arm} {game} rep{rep} · levels_beaten={beaten} · {stop} · {dur/60:.1f}m")
    return {"arm": arm, "game": game, "rep": rep, "rc": proc.returncode,
            "levels_beaten": beaten, "stop_reason": stop, "wall_s": round(dur, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="vLLM OpenAI endpoint, e.g. http://nid006657:8000/v1")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    # Qwen3-32B's window is 40960. The harness reserves ~1.5x max_tokens from it, so the
    # harness default of 32000 would leave a NEGATIVE history budget and silently disable
    # truncation (-> the server hard-errors once the window fills). 8192 leaves ~28k of
    # rolling history while still fitting a long thinking turn.
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--move-channel", default="guided-json",
                    help="Must be identical across arms -- the channel is a fairness confound.")
    ap.add_argument("--run-timeout", type=int, default=10800, help="per-run wall ceiling (s)")
    ap.add_argument("--games", nargs="*", help="override the game list (default: all public games)")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()

    token = os.environ.get("DIGBENCH_API_TOKEN")
    if not token:
        print("DIGBENCH_API_TOKEN is not set", file=sys.stderr)
        return 2

    games = args.games or list_games(token)
    log(f"{len(games)} games · arms={args.arms} · reps={args.reps} "
        f"· {len(games) * len(args.arms) * args.reps} runs total")

    # Interleave arms and reps so that a sweep cut short by a Slurm timeout still
    # yields balanced coverage across both arms rather than a complete arm A and
    # an empty arm B.
    jobs = [(arm, ARMS[arm], game, rep)
            for rep in range(1, args.reps + 1)
            for game in games
            for arm in args.arms]

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, a, m, g, r, args) for a, m, g, r in jobs]
        for fut in cf.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # a crashed subprocess must not kill the sweep
                log(f"ERROR {type(exc).__name__}: {exc}")
                results.append({"status": "error", "error": str(exc)})

    out = RUNS / "sweep_index.json"
    out.write_text(json.dumps(results, indent=2))
    log(f"wrote {out}")
    done = [r for r in results if r.get("levels_beaten") is not None]
    log(f"{len(done)}/{len(results)} runs produced a summary row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
