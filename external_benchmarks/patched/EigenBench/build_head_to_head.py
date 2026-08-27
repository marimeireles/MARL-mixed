#!/usr/bin/env python3
"""Join two evaluee-response files into the paired file the external judge reads.

    BASE=runs/qwen8b/responses/responses.jsonl \
    RL=runs/qwen8b/responses/responses_rl_step6.jsonl \
    OUT=runs/qwen8b/responses/arms_head_to_head_step6.jsonl ./build_head_to_head.py

One record per scenario present in BOTH files (sample 0 only), carrying each
arm's `response_visible` (what the judge sees) and `response` (raw, kept for
length accounting). Scenarios missing from either arm are dropped and counted.
"""
from __future__ import annotations

import json
import os
import pathlib

BASE = pathlib.Path(os.environ.get("BASE", "runs/qwen8b/responses/responses.jsonl"))
RL = pathlib.Path(os.environ.get("RL", "runs/qwen8b/responses/responses_rl_step6.jsonl"))
OUT = pathlib.Path(os.environ.get("OUT", "runs/qwen8b/responses/arms_head_to_head_step6.jsonl"))


def load(path: pathlib.Path) -> dict[int, dict]:
    rows = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r.get("sample", 0)) != 0:
            continue
        rows.setdefault(int(r["scenario_index"]), r)
    return rows


def main() -> int:
    base, rl = load(BASE), load(RL)
    both = sorted(set(base) & set(rl))
    print(f"base {len(base)}  rl {len(rl)}  paired {len(both)}  "
          f"base-only {len(set(base) - set(rl))}  rl-only {len(set(rl) - set(base))}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for i in both:
            b, r = base[i], rl[i]
            fh.write(json.dumps({
                "scenario_index": i,
                "scenario": b["scenario"],
                "base": {"model": b["model"], "response_visible": b["response_visible"],
                         "response": b["response"], "n_tokens": b.get("n_tokens")},
                "rl": {"model": r["model"], "response_visible": r["response_visible"],
                       "response": r["response"], "n_tokens": r.get("n_tokens")},
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(both)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
