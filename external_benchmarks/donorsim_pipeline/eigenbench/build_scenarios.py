#!/usr/bin/env python3
"""Convert AIRiskDilemmas into an EigenBench scenario file.

AIRiskDilemmas ships one row per (dilemma, action) pair, so every dilemma
appears once per candidate action. EigenBench judges free-text responses, not
forced choices, so we keep the dilemma text only and drop the action variants --
the models answer the dilemma openly and the constitution decides who answered
better.

The dataset's own `risky_behaviors` and `context` labels are carried through as
extra keys. EigenBench's `_normalize_scenarios` reads the `dilemma` field and
ignores everything else, so they cost nothing at run time but let us slice
results by risk category afterwards (does the adapter help on Deception but not
Power-Seeking?).

    python build_scenarios.py [--count 400] [--out ../../data/scenarios/airiskdilemmas.json]
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://huggingface.co/datasets/kellycyy/AIRiskDilemmas/resolve/main/full.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data/scenarios/airiskdilemmas.json")
    ap.add_argument("--count", type=int, default=400,
                    help="how many unique dilemmas to keep (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    with urllib.request.urlopen(URL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("dilemma") or "").strip()
            if not text or text in seen:
                continue
            seen[text] = {
                "dilemma": text,
                "risky_behaviors": row.get("risky_behaviors") or [],
                "context": row.get("context") or "",
            }

    scenarios = list(seen.values())
    print(f"{len(scenarios)} unique dilemmas")

    # Stratify by risk category so a smaller slice keeps the same mix as the full
    # set -- a random head would over-represent whatever the file happens to list
    # first, and the risk categories are exactly what we want to compare across.
    if args.count and args.count < len(scenarios):
        rng = random.Random(args.seed)
        by_cat: dict[str, list[dict]] = {}
        for s in scenarios:
            key = s["risky_behaviors"][0] if s["risky_behaviors"] else "unlabelled"
            by_cat.setdefault(key, []).append(s)
        for group in by_cat.values():
            rng.shuffle(group)
        picked, i = [], 0
        while len(picked) < args.count:
            added = False
            for group in by_cat.values():          # round-robin across categories
                if i < len(group) and len(picked) < args.count:
                    picked.append(group[i])
                    added = True
            if not added:
                break
            i += 1
        scenarios = picked

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scenarios, indent=1), encoding="utf-8")

    cats = Counter(s["risky_behaviors"][0] if s["risky_behaviors"] else "unlabelled"
                   for s in scenarios)
    ctxs = Counter(s["context"] or "unlabelled" for s in scenarios)
    print(f"wrote {args.out}  ({len(scenarios)} scenarios)")
    print("  risk categories:", dict(cats.most_common()))
    print("  contexts:", dict(ctxs.most_common(8)))


if __name__ == "__main__":
    main()
