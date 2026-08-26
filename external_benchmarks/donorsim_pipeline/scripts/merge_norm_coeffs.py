#!/usr/bin/env python3
"""Merge our generated random-agent coefficients into the shipped ones, and report
how much the two provenances disagree.

Policy: the 30 games MACHIAVELLI ships keep the PAPER's coefficients untouched, so
every number we have already published stays exactly comparable. The other 106 get
ours. That leaves the final file with mixed provenance, which is only defensible if
the two agree -- so this script quantifies the disagreement on the 30-game overlap
rather than asserting it is small.

Their baseline and ours differ in at least one known way (we cap episodes at 400
steps to match `run_arms.py`; the paper's random baseline reports no cap), so the
overlap comparison is the honest way to size that choice.

Writes:
  game_data/normalization_coeffs.json          <- merged, used by every scorer
  game_data/normalization_coeffs.provenance.md <- the disagreement report
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

MACH = Path(__file__).resolve().parent
os.chdir(MACH)
sys.path.insert(0, str(MACH))

from machiavelli.config import MachiavelliConfig  # noqa: E402

GD = MACH / "game_data"
SHIPPED = GD / "normalization_coeffs.json.orig"
OURS = GD / "normalization_coeffs.ours.json"
OUT = GD / "normalization_coeffs.json"
REPORT = GD / "normalization_coeffs.provenance.md"

KEYS = ["game.score", "violations.Σ", "power.Σ", "utility.Σ"]


def main() -> int:
    if not SHIPPED.exists():
        print(f"missing {SHIPPED} -- refusing to merge without the pristine original")
        return 1
    shipped = json.loads(SHIPPED.read_text())
    ours = json.loads(OURS.read_text()) if OURS.exists() else {}
    working = list(MachiavelliConfig().working_games)

    merged = dict(shipped)                       # paper's 30 win by policy
    added = [g for g in working if g not in merged and g in ours]
    for g in added:
        merged[g] = ours[g]

    still_missing = [g for g in working if g not in merged]
    OUT.write_text(json.dumps(merged, indent=4, sort_keys=True, ensure_ascii=False))

    # --- provenance check on the overlap ---
    overlap = [g for g in shipped if g in ours]
    lines = ["# Normalization coefficient provenance", "",
             f"Merged file covers **{len(merged)}** games "
             f"({len(shipped)} from the MACHIAVELLI release, {len(added)} generated here).",
             ""]
    if still_missing:
        lines += [f"⚠️ {len(still_missing)} working game(s) still have no coefficients: "
                  f"{', '.join(still_missing)}", ""]

    if overlap:
        lines += [f"## Agreement on the {len(overlap)} overlapping games", "",
                  "Ratio of our random-agent mean to the paper's. 1.00 = identical; "
                  "our rollouts cap at 400 steps to match `run_arms.py`, theirs do not, "
                  "so ratios below 1 on cumulative counts are the expected direction.", "",
                  "| count | median ratio | min | max | games within ±10% |",
                  "|---|---|---|---|---|"]
        for k in KEYS:
            ratios = []
            for g in overlap:
                a = shipped[g].get(k, {}).get("mean")
                b = ours[g].get(k, {}).get("mean")
                if a and b is not None and a != 0:
                    ratios.append(b / a)
            if not ratios:
                continue
            within = sum(1 for r in ratios if 0.9 <= r <= 1.1)
            lines.append(f"| `{k}` | {statistics.median(ratios):.3f} | {min(ratios):.3f} | "
                         f"{max(ratios):.3f} | {within}/{len(ratios)} |")
        lines += ["", "If these ratios sit near 1.00, mixing provenances is safe and the "
                  "30 shipped games stay on the paper's own scale. If they do not, rerun "
                  "the analysis against `normalization_coeffs.ours.json` for all 136 games "
                  "instead -- that file is self-consistent by construction.", ""]

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"merged -> {OUT}  ({len(merged)} games: {len(shipped)} shipped + {len(added)} ours)")
    if still_missing:
        print(f"WARNING: still missing {len(still_missing)}: {still_missing}")
    print(f"provenance report -> {REPORT}")
    for line in lines:
        if line.startswith("|") and "`" in line:
            print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
