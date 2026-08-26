#!/usr/bin/env python3
"""Score the MACHIAVELLI trajectory pickles into one tidy CSV.

Uses the benchmark's own `Trajectory.get_mach_scores()`, i.e. the published
normalisation: every harm/power count is divided by the random agent's mean
count on the SAME game and expressed as a percentage, so 100 = "as harmful as
random" and lower is better. `game.score` is normalised the same way, but there
higher is better.

We merge in our own per-arm sidecar stats (reasoning length, tokens, parse
fallbacks) so the notebook can check that a reward difference is not just one
arm thinking less than the other.

Output: data/machiavelli_trajectories.csv -- one row per (arm, game, episode).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

MACH = "/capstor/scratch/cscs/mmeireles/machiavelli"
sys.path.insert(0, MACH)
# MachiavelliConfig resolves its data paths (incl. the random-agent normalization
# coefficients) relative to the CWD, so run from the repo root.
os.chdir(MACH)

from machiavelli.game.machiavelli_env import Trajectory  # noqa: E402

TRAJ = Path("/capstor/scratch/cscs/mmeireles/machiavelli-run/trajectories")
OUT = Path(__file__).resolve().parent / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Prefer the per-episode .done.json markers over index.json: the index is only
    # written when run_arms.py finishes the whole sweep, so mid-sweep it is stale or
    # absent and every new trajectory would lose its sidecar stats. The markers are
    # written per trajectory, so this aggregator works on a partial sweep.
    index = {}
    if (TRAJ / "index.json").exists():
        for r in json.loads((TRAJ / "index.json").read_text()):
            if "error" not in r:
                index[(r["arm"], r["game"], r["episode"])] = r
    for marker in TRAJ.glob("*/*/ep*.done.json"):
        try:
            r = json.loads(marker.read_text())
        except (OSError, ValueError):
            continue
        index[(r["arm"], r["game"], r["episode"])] = r

    rows = []
    missing_coeffs: set[str] = set()
    for path in sorted(TRAJ.glob("*/*/ep*.pkl")):
        arm = path.parent.parent.name
        episode = int(path.stem.removeprefix("ep"))
        traj = Trajectory.load(path)
        side = index.get((arm, traj.game_name, episode), {})

        # Normalised scores need this game's random-agent coefficients. The sweep
        # widened to 136 games before `gen_norm_coeffs.py` had finished generating
        # the missing 106, so a trajectory can legitimately arrive before its
        # denominator exists -- record it unnormalised rather than crashing the
        # whole aggregation (and with it the autopilot's checkpoint).
        try:
            scores = traj.get_mach_scores()
        except KeyError:
            scores = {}
            missing_coeffs.add(traj.game_name)

        rows.append({
            "arm": arm,
            "game": traj.game_name,
            "episode": episode,
            "steps": len(traj),
            # Reward side: raw achievement count, and points as % of the game max.
            "achievements": len(traj.get_achievements()),
            "points_pct_of_max": traj.get_points(normalize=True),
            # Raw counts are kept alongside the normalised scores so the whole set
            # can be re-normalised later without replaying a single trajectory.
            **{f"raw.{k}": v for k, v in traj.get_mach_counts().items()},
            # Behaviour side: MACHIAVELLI's random-normalised harm/power scores.
            **scores,
            # Our own instrumentation.
            "reached_end": side.get("reached_end"),
            "mean_reasoning_chars": side.get("mean_reasoning_chars"),
            "total_completion_tokens": side.get("total_completion_tokens"),
            "parse_failures": side.get("parse_failures"),
            "random_actions": side.get("random_actions"),
            "wall_s": side.get("wall_s"),
        })
        print(f"  {arm:11s} {traj.game_name:45s} steps={len(traj):4d} "
              f"ach={len(traj.get_achievements()):3d}", flush=True)

    df = pd.DataFrame(rows).sort_values(["game", "arm"])
    path = OUT / "machiavelli_trajectories.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(df)} rows, {df['game'].nunique()} games, "
          f"{df['arm'].nunique()} arms)")
    if missing_coeffs:
        print(f"NOTE: {len(missing_coeffs)} game(s) have no random-agent coefficients yet, so "
              f"their normalised scores are blank (raw.* columns are still populated): "
              f"{', '.join(sorted(missing_coeffs)[:8])}"
              f"{' ...' if len(missing_coeffs) > 8 else ''}")


if __name__ == "__main__":
    main()
