#!/usr/bin/env python3
"""Generate MACHIAVELLI trajectories for both model arms against one vLLM server.

Mirrors generate_trajectories.py's loop, but adds the things a two-arm
comparison needs: a bounded thread pool (one env per worker), resumability
(a finished trajectory pickle means skip), per-arm sidecar logs of the model's
own reasoning, and the arm/model recorded alongside every artifact.

Both arms hit the SAME server and the SAME prompts; only the served model id
differs, so the arm is the only variable.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from machiavelli.agent.vllm_agent import VLLMAgent  # noqa: E402
from machiavelli.config import MachiavelliConfig  # noqa: E402
from machiavelli.game.machiavelli_env import MachiavelliEnv  # noqa: E402

ARMS = {
    "rl-step175": "qwen3-32b-donorsim",
    "base": "qwen3-32b-base",
}

_lock = threading.Lock()


def log(msg: str) -> None:
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(arm: str, model: str, game: str, episode: int, args) -> dict:
    outdir = Path(args.traj_dir) / arm / game
    done_marker = outdir / f"ep{episode}.done.json"
    if done_marker.exists():
        log(f"SKIP  {arm} {game} ep{episode}")
        return json.loads(done_marker.read_text())

    outdir.mkdir(parents=True, exist_ok=True)
    # Truncate the sidecar: a retry after a server walltime kill would otherwise append
    # to the dead attempt's rows and double-count this episode's steps.
    (outdir / f"ep{episode}.steps.jsonl").unlink(missing_ok=True)
    agent = VLLMAgent(
        model=model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        log_path=str(outdir / f"ep{episode}.steps.jsonl"),
    )
    env = MachiavelliEnv(game=game)

    started = time.time()
    log(f"START {arm} {game} ep{episode}")
    try:
        obs, info = env.reset(seed=episode)
        rew, done, steps = 0, False, 0
        while not done:
            action = agent.get_action(obs, rew, done, info)
            obs, rew, done, info = env.step(action)
            steps += 1
            if steps >= args.max_traj_length:
                log(f"  {arm} {game} ep{episode}: hit max_traj_length {args.max_traj_length}")
                break
        traj_path = outdir / f"ep{episode}.pkl"
        env.trajectory.save(traj_path)
        result = {
            "arm": arm, "model": model, "game": game, "episode": episode,
            "steps": steps, "reached_end": bool(done),
            "traj": str(traj_path),
            "total_calls": agent.total_calls,
            "mean_reasoning_chars": round(
                agent.total_reasoning_chars / max(agent.total_calls, 1), 1),
            "total_completion_tokens": agent.total_completion_tokens,
            "parse_failures": agent.parse_failures,
            "random_actions": agent.random_actions,
            "wall_s": round(time.time() - started, 1),
        }
        done_marker.write_text(json.dumps(result, indent=2))
        log(f"DONE  {arm} {game} ep{episode} · steps={steps} · "
            f"reasoning={result['mean_reasoning_chars']:.0f}ch · {result['wall_s']/60:.1f}m")
        return result
    except Exception as exc:  # noqa: BLE001 - one bad game must not kill the sweep
        log(f"ERROR {arm} {game} ep{episode}: {type(exc).__name__}: {exc}")
        (outdir / f"ep{episode}.error.txt").write_text(traceback.format_exc())
        return {"arm": arm, "game": game, "episode": episode, "error": str(exc)}
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--traj-dir", default="/capstor/scratch/cscs/mmeireles/machiavelli-run/trajectories")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-traj-length", type=int, default=400)
    ap.add_argument("--games", nargs="*")
    # `working` = all 136 non-buggy games. Widening the game set is the only lever that
    # reduces the paired MDE regardless of whether the noise is within- or between-game,
    # so it beats buying more episodes on the 30-game test split.
    ap.add_argument("--game-set", choices=["test", "working"], default="test")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "dummy")  # openai_helpers builds a client on import
    cfg = MachiavelliConfig()
    games = args.games or (cfg.working_games if args.game_set == "working"
                           else cfg.games_test)
    log(f"{len(games)} games · arms={args.arms} · episodes={args.episodes} "
        f"· {len(games) * len(args.arms) * args.episodes} trajectories")

    # Interleave arms so an interrupted sweep still has balanced coverage.
    jobs = [(arm, ARMS[arm], game, ep)
            for ep in range(args.episodes)
            for game in games
            for arm in args.arms]

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(run_one, a, m, g, e, args) for a, m, g, e in jobs]
        for fut in cf.as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                log(f"POOL ERROR {exc}")

    index = Path(args.traj_dir) / "index.json"
    index.write_text(json.dumps(results, indent=2))
    ok = [r for r in results if "error" not in r]
    log(f"wrote {index} · {len(ok)}/{len(results)} trajectories completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
