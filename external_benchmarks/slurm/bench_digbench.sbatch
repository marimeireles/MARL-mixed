#!/bin/bash
#SBATCH --job-name=bench-digbench
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-digbench-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-digbench-%j.err
#SBATCH --export=ALL
# DiG-bench: 21 public games x REPS per arm via the baseline harness against our vLLM
# server (guided-json move channel, needs --reasoning-parser qwen3). Token from
# DIGBENCH_API_TOKEN or ~/.digbench_token.
#   DIG_ARMS='{"base":"qwen3-8b-bench"}' REPS=10 sbatch external_benchmarks/slurm/bench_digbench.sbatch
set -uo pipefail
ROOT=/nas/ucb/marimeireles/MARL-mixed; cd $ROOT/external_benchmarks/dig-bench
TOKEN_FILE="${DIGBENCH_TOKEN_FILE:-$HOME/.digbench_token}"
if [ -z "${DIGBENCH_API_TOKEN:-}" ] && [ -r "$TOKEN_FILE" ]; then export DIGBENCH_API_TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"; fi
: "${DIGBENCH_API_TOKEN:?no token: export DIGBENCH_API_TOKEN or write it to $TOKEN_FILE (chmod 600)}"
ENDPOINT_FILE="${ENDPOINT_FILE:-$ROOT/llm_dynamics/logs/vllm_endpoint_qwen3-8b-bench.txt}"
for _ in $(seq 1 360); do [ -f "${ENDPOINT_FILE}.ready" ] && break; sleep 20; done
API="$(cat $ENDPOINT_FILE)/v1"; echo "endpoint $API ($(date -Iseconds))"
DEFAULT_ARMS='{"base":"qwen3-8b-bench"}'
export DIG_ARMS="${DIG_ARMS:-$DEFAULT_ARMS}" DIG_RUNS="${DIG_RUNS:-$ROOT/external_benchmarks/runs/digbench/runs}"
# optional sharding: SHARD=i NSHARD=n plays games i, i+n, i+2n, ... (disjoint writers)
GAMES_ARG=""
if [ -n "${NSHARD:-}" ]; then
  GAMES=$(SHARD="${SHARD:-0}" NSHARD="$NSHARD" $ROOT/.venv-bench/bin/python -c "
import os, json
g = json.load(open(os.environ.get('DIG_GAMES_JSON', '$ROOT/external_benchmarks/runs/digbench/runs/games.json')))
ids = [x['id'] if isinstance(x, dict) else x for x in (g if isinstance(g, list) else g.get('games', []))]
print(' '.join(ids[int(os.environ['SHARD'])::int(os.environ['NSHARD'])]))")
  GAMES_ARG="--games $GAMES"; echo "shard ${SHARD:-0}/${NSHARD}: $(echo $GAMES | wc -w) games -> $GAMES"
fi
$ROOT/.venv-bench/bin/python orchestrate.py --base-url "$API" --reps "${REPS:-10}" --concurrency "${CONCURRENCY:-21}" --max-steps 200 --move-channel guided-json --max-tokens 8192 $GAMES_ARG
if [ "${AGGREGATE:-1}" = "1" ]; then
echo "== aggregate =="; $ROOT/.venv-bench/bin/python aggregate_digbench.py 2>&1 | tail -20; cp data/digbench_runs.csv "$ROOT/llm_dynamics/results/external/digbench_runs_${SLURM_JOB_ID}.csv"
fi
echo "== DONE $(date -Iseconds) =="
