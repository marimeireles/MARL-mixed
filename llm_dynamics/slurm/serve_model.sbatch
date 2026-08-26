#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=main
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm-%j.err
#SBATCH --export=ALL
# Generic vLLM server for llm_dynamics. Env: MODEL_PATH (HF id or local dir),
# SERVED_NAME, PORT, ENDPOINT_FILE, MAX_LEN, LORA (optional adapter dir ->
# served as SERVED_NAME via --lora-modules), EXTRA_ARGS (e.g. reasoning/tool
# parsers for the external benchmarks: "--reasoning-parser qwen3
# --enable-auto-tool-choice --tool-call-parser hermes").
#   MODEL_PATH=Qwen/Qwen3-8B SERVED_NAME=qwen3-8b-base sbatch llm_dynamics/slurm/serve_model.sbatch
set -euo pipefail
cd /nas/ucb/marimeireles/MARL-mixed
export HF_HOME=/nas/ucb/marimeireles/cache
PORT="${PORT:-8010}"; MODEL_PATH="${MODEL_PATH:?}"; SERVED_NAME="${SERVED_NAME:?}"
ENDPOINT_FILE="${ENDPOINT_FILE:-/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/vllm_endpoint_${SERVED_NAME}.txt}"
MAX_LEN="${MAX_LEN:-16384}"
echo "job ${SLURM_JOB_ID:-} node $(hostname -f) model $MODEL_PATH as $SERVED_NAME port $PORT $(date -Iseconds)"
PRIV_IP=$(hostname -I | tr ' ' '\n' | awk '/^192\.168\./{print; exit}'); [ -z "$PRIV_IP" ] && PRIV_IP=$(hostname -f)
echo "http://${PRIV_IP}:${PORT}" > "$ENDPOINT_FILE"; rm -f "${ENDPOINT_FILE}.ready"
( for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { touch "${ENDPOINT_FILE}.ready"; echo "[$(date -Iseconds)] READY"; break; }; sleep 30; done ) &
EXTRA="${EXTRA_ARGS:-}"
if [ -n "${LORA:-}" ]; then EXTRA="$EXTRA --enable-lora --max-lora-rank 64 --lora-modules ${SERVED_NAME}=${LORA}"; fi
exec /nas/ucb/marimeireles/vllm-env/bin/vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 --max-model-len "$MAX_LEN" --dtype bfloat16 \
    --served-model-name "$SERVED_NAME" $EXTRA
