#!/bin/bash
#SBATCH --job-name=bench-machiavelli
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-machiavelli-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-machiavelli-%j.err
#SBATCH --export=ALL
# MACHIAVELLI over all 136 working games x EPISODES, both/any arms, against the
# vLLM server from serve_model.sbatch (needs --reasoning-parser qwen3).
#   MACH_ARMS='{"base":"qwen3-8b-bench"}' EPISODES=3 sbatch external_benchmarks/slurm/bench_machiavelli.sbatch
set -uo pipefail
ROOT=/nas/ucb/marimeireles/MARL-mixed; cd $ROOT/external_benchmarks/machiavelli
ENDPOINT_FILE="${ENDPOINT_FILE:-$ROOT/llm_dynamics/logs/vllm_endpoint_qwen3-8b-bench.txt}"
for _ in $(seq 1 360); do [ -f "${ENDPOINT_FILE}.ready" ] && break; sleep 20; done
[ -f "${ENDPOINT_FILE}.ready" ] || { echo "endpoint never ready"; exit 1; }
API="$(cat $ENDPOINT_FILE)/v1"; echo "endpoint $API arms ${MACH_ARMS:-default} ($(date -Iseconds))"
DEFAULT_ARMS='{"base":"qwen3-8b-bench"}'
export OPENAI_API_KEY=dummy VLLM_BASE_URL=$API MACH_ARMS="${MACH_ARMS:-$DEFAULT_ARMS}"
TRAJ="${TRAJ:-$ROOT/external_benchmarks/runs/machiavelli/trajectories}"
$ROOT/.venv-bench/bin/python run_arms.py --base-url "$API" --traj-dir "$TRAJ" --game-set working \
    --episodes "${EPISODES:-3}" --concurrency "${CONCURRENCY:-24}" --max-traj-length 400 --max-tokens 2048
echo "== aggregate =="
MACH_TRAJ="$TRAJ" $ROOT/.venv-bench/bin/python aggregate_machiavelli.py 2>&1 | tail -20; cp data/machiavelli_trajectories.csv "$ROOT/llm_dynamics/results/external/machiavelli_trajectories_${SLURM_JOB_ID}.csv"
echo "== DONE $(date -Iseconds) =="
