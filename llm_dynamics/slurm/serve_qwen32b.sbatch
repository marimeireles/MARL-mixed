#!/bin/bash
#SBATCH --job-name=vllm-qwen3-32b
#SBATCH --partition=main
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm-qwen3-32b-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm-qwen3-32b-%j.err
#SBATCH --export=ALL
#
# Serves base Qwen/Qwen3-32B (bf16) on one A100-80GB for the llm_dynamics
# experiments. Pattern copied from zoo-eval/scripts/sbatch_vllm_gemma4.sh:
# writes the reachable endpoint to ENDPOINT_FILE and a `.ready` sentinel
# once /health answers, so client jobs can wait on it.
#
#   sbatch llm_dynamics/slurm/serve_qwen32b.sbatch
#   PORT=8011 sbatch --export=ALL,PORT llm_dynamics/slurm/serve_qwen32b.sbatch
set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
export HF_HOME=/nas/ucb/marimeireles/cache
PORT="${PORT:-8010}"
ENDPOINT_FILE=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm_endpoint.txt

echo "job ${SLURM_JOB_ID:-} node $(hostname -f) gpus ${CUDA_VISIBLE_DEVICES:-unset} port $PORT start $(date -Iseconds)"
PRIV_IP=$(hostname -I | tr ' ' '\n' | awk '/^192\.168\./{print; exit}')
[ -z "$PRIV_IP" ] && PRIV_IP=$(hostname -f)
echo "http://${PRIV_IP}:${PORT}" > "$ENDPOINT_FILE"
rm -f "${ENDPOINT_FILE}.ready"
echo "endpoint: $(cat "$ENDPOINT_FILE")"

( for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      touch "${ENDPOINT_FILE}.ready"; echo "[$(date -Iseconds)] vLLM READY"; break
    fi; sleep 30; done ) &

exec /nas/ucb/marimeireles/vllm-env/bin/vllm serve Qwen/Qwen3-32B \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --served-model-name qwen3-32b-base
