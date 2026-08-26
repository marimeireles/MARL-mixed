#!/usr/bin/env python3
"""Flatten the DiG-bench sweep's per-run JSONL into two tidy CSVs.

The notebooks read only these CSVs, never the raw run directories, so the
published artifact is self-contained and carries no verbatim game text: the
turn table keeps observation LENGTHS and the model's own action string, not the
observations or the model's reasoning prose.

Outputs
  data/digbench_runs.csv   one row per (arm, game, rep)  -- 126 rows
  data/digbench_turns.csv  one row per turn              -- level trace per run
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/capstor/scratch/cscs/mmeireles/digbench-run/runs")
OUT = Path(__file__).parent / "data"
TIERS = {g["name"]: g["tier"]
         for g in json.loads((ROOT / "games.json").read_text())["game_details"]}


def read_run(path: Path) -> tuple[dict | None, dict | None, list[dict], int]:
    """Return (session row, summary row, turn rows, count of transport warnings)."""
    session, summary, turns, warns = None, None, [], 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            kind = row.get("type")
            if kind == "summary":
                summary = row
            elif kind == "session":
                session = row
            elif kind == "turn":
                turns.append(row)
            elif kind == "warn":
                warns += 1
    return session, summary, turns, warns


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_rows, turn_rows = [], []

    for arm_dir in sorted(ROOT.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        for game_dir in sorted(arm_dir.iterdir()):
            game = game_dir.name
            for rep_dir in sorted(game_dir.glob("rep*")):
                rep = int(rep_dir.name.removeprefix("rep"))
                # A run retried across Slurm allocations leaves one JSONL per attempt.
                # Take the newest attempt that actually finished; never just the first,
                # which may be a truncated run killed when the server hit its walltime.
                session = summary = None
                for path in sorted(rep_dir.glob("*.jsonl"), reverse=True):
                    session, summary, turns, warns = read_run(path)
                    if summary is not None:
                        break
                if summary is None:
                    continue

                tokens = summary.get("tokens") or {}
                # The harness reports thoughts=0 for our provider (vLLM returns
                # reasoning out-of-band), so measure thinking from the turn rows.
                summ_chars = [len(t.get("output", {}).get("reasoning_summary") or "")
                              for t in turns]
                nonzero = [c for c in summ_chars if c > 0]
                creative = sum(1 for t in turns
                               if (t.get("input") or {}).get("mode") == "creative")

                # PRE-REGISTERED PRIMARY ENDPOINT for the 10-rep sweep.
                # `levels_beaten` is coarse and zero-inflated (46/63 base runs scored 0
                # in the 3-rep pilot), which caps its resolution at ~14% of base even
                # with infinite reps on the 21-game public set. The mean level over the
                # trajectory scores every run, and on the pilot data resolves ~12% --
                # better than levels_beaten can ever reach. levels_beaten is retained as
                # the secondary/official endpoint.
                levels = [t["input"].get("level") for t in turns
                          if (t.get("input") or {}).get("level") is not None]

                run_rows.append({
                    "arm": arm, "game": game, "tier": TIERS.get(game), "rep": rep,
                    "auc_level": round(sum(levels) / len(levels), 4) if levels else None,
                    "max_level_seen": max(levels) if levels else None,
                    "levels_beaten": summary.get("levels_beaten"),
                    "level_reached": summary.get("level"),
                    "max_level": summary.get("max_level"),
                    "stop_reason": summary.get("stop_reason"),
                    "result": summary.get("result"),
                    "turns": summary.get("turns"),
                    "llm_calls": summary.get("llm_calls"),
                    "prompt_tokens": tokens.get("prompt"),
                    "output_tokens": tokens.get("output"),
                    "total_tokens": tokens.get("total"),
                    "wall_s": summary.get("wall_s"),
                    "llm_s": summary.get("llm_s"),
                    "creative_turns": creative,
                    "mean_reasoning_chars": round(sum(summ_chars) / len(summ_chars), 1) if summ_chars else 0.0,
                    "zero_reasoning_turns": len(summ_chars) - len(nonzero),
                    "transport_warnings": warns,
                    "move_channel": summary.get("move_channel"),
                    "model": summary.get("reported_model"),
                    # Provenance: the session id is the server-side run, and the
                    # playback URL replays it move-for-move on digbench.ai.
                    "session_id": (session or {}).get("session_id", ""),
                    "game_seed": (session or {}).get("game_seed"),
                    "framework_version": (session or {}).get("framework_version"),
                    "playback": summary.get("playback") or "",
                })

                for t in turns:
                    inp = t.get("input") or {}
                    outp = t.get("output") or {}
                    turn_rows.append({
                        "arm": arm, "game": game, "tier": TIERS.get(game), "rep": rep,
                        "turn": t.get("turn"),
                        "level": inp.get("level"),
                        "lives_left": inp.get("lives_left"),
                        "steps_remaining": inp.get("steps_remaining"),
                        "mode": inp.get("mode"),
                        "status": inp.get("status"),
                        "n_legal_actions": len(inp.get("legal_actions") or []),
                        "obs_chars": len(inp.get("observation") or ""),
                        "action": outp.get("action"),
                        "reasoning_chars": len(outp.get("reasoning_summary") or ""),
                        "elapsed_s": t.get("elapsed_s"),
                    })

    for name, rows in (("digbench_runs.csv", run_rows), ("digbench_turns.csv", turn_rows)):
        path = OUT / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
