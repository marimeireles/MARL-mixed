#!/usr/bin/env python3
"""Generate random-agent normalization coefficients for the un-shipped games.

MACHIAVELLI ships `normalization_coeffs.json` for the 30 test games only, but the
game data itself covers all 136 `working_games`. Every harm/power score is a
percentage of the random agent's mean count on the SAME game, so without these
coefficients the extra 106 games can be played but not scored -- which is what
capped our sweep at 30 games.

The random agent needs no LLM, so this is pure CPU: ~0.1-0.6 s per episode.

Differs from upstream `machiavelli/data/get_normalization_coeffs.py` in two ways
that matter here:

  * Upstream writes 1000 pickled trajectories per game to disk and re-reads them.
    That would be ~106k pickles for no benefit, so counts are accumulated in
    memory instead.
  * Upstream rebuilds the whole JSON from whatever trajectories it finds, which
    would silently DROP the shipped 30 games we have no rollouts for. This merges
    instead, leaving existing entries untouched so previously computed results
    stay exactly comparable.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

MACH = Path(__file__).resolve().parent
os.chdir(MACH)
sys.path.insert(0, str(MACH))

from machiavelli.config import MachiavelliConfig  # noqa: E402
from machiavelli.game.machiavelli_env import MachiavelliEnv  # noqa: E402

N_EPISODES = int(os.environ.get("N_EPISODES", 1000))   # matches the paper's random baseline
WORKERS = int(os.environ.get("WORKERS", 24))           # shared login node: stay modest
# Must match `run_arms.py --max-traj-length`. The coefficient is the DENOMINATOR of
# every score, so it has to be measured under the same stopping rule as the agents
# it normalises. It is also load-bearing for termination: a random agent loops in
# `rebels` indefinitely and never reaches an ending.
MAX_STEPS = int(os.environ.get("MAX_STEPS", 400))
FORCE_ALL = os.environ.get("FORCE_ALL", "0") == "1"
COEFFS = MACH / "game_data" / os.environ.get("OUT", "normalization_coeffs.json")


def rollout_game(game: str) -> tuple[str, dict | None, str]:
    """Run N_EPISODES random episodes and return mean/stddev per count key."""
    t0 = time.time()
    rows: list[dict] = []
    capped = 0
    for ep in range(N_EPISODES):
        try:
            env = MachiavelliEnv(game=game)
            _, info = env.reset(seed=ep)
            rng = random.Random(ep)
            done = False
            steps = 0
            while not done:
                choices = info["game_state"]["choice_texts"]
                if not choices:
                    break
                _, _, done, info = env.step(rng.randrange(len(choices)))
                steps += 1
                if steps >= MAX_STEPS:
                    capped += 1
                    break
            rows.append(env.trajectory.get_mach_counts())
            env.close()
        except Exception as exc:                      # a single bad episode must not lose the game
            if ep == 0:
                return game, None, f"{type(exc).__name__}: {exc}"
            continue

    if not rows:
        return game, None, "no episodes completed"

    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [float(r.get(k, 0)) for r in rows]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        out[k] = {"mean": mean, "stddev": var ** 0.5, "n_samples": n}
    return game, out, (f"{len(rows)} eps in {time.time() - t0:.0f}s, "
                       f"{100 * capped / max(len(rows), 1):.0f}% hit the {MAX_STEPS}-step cap")


def main() -> int:
    cfg = MachiavelliConfig()
    existing = json.loads(COEFFS.read_text()) if COEFFS.exists() else {}
    todo = list(cfg.working_games) if FORCE_ALL else [g for g in cfg.working_games
                                                      if g not in existing]
    print(f"working games: {len(cfg.working_games)} | already have coeffs: {len(existing)} | "
          f"to generate: {len(todo)} @ {N_EPISODES} eps, cap {MAX_STEPS} steps, "
          f"{WORKERS} workers -> {COEFFS.name}", flush=True)
    if not todo:
        print("nothing to do")
        return 0

    # Back up before the first write; the shipped file is not reproducible.
    backup = COEFFS.with_suffix(".json.orig")
    if COEFFS.exists() and not backup.exists():
        backup.write_text(COEFFS.read_text())
        print(f"backed up shipped coeffs -> {backup}", flush=True)

    done = 0
    failed = []
    with Pool(WORKERS) as pool:
        for game, coeffs, note in pool.imap_unordered(rollout_game, todo):
            done += 1
            if coeffs is None:
                failed.append(game)
                print(f"[{done}/{len(todo)}] FAIL {game}: {note}", flush=True)
                continue
            existing[game] = coeffs
            print(f"[{done}/{len(todo)}] ok   {game:45s} {note}", flush=True)
            # Write through after every game: this runs for ~30 min on a shared
            # login node, and a kill must not throw away completed rollouts.
            tmp = COEFFS.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(existing, indent=4, sort_keys=True, ensure_ascii=False))
            tmp.replace(COEFFS)

    print(f"\ndone: {len(existing)} games have coeffs; {len(failed)} failed", flush=True)
    if failed:
        print("failed:", failed, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
