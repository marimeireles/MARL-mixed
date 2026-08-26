#!/bin/bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --job-name=digbench-vllm
#SBATCH --output=/capstor/scratch/cscs/mmeireles/digbench-run/logs/vllm-%j.out
#SBATCH --environment=/capstor/scratch/cscs/mmeireles/digbench-run/edf/vllm.toml

# Serves BOTH benchmark arms from ONE vLLM process so that every serving
# condition except the LoRA weights is identical between them:
#   qwen3-32b-base      -> stock Qwen/Qwen3-32B
#   qwen3-32b-donorsim  -> same base + GRPO LoRA (step_175, rank 16)

set -euo pipefail

BASE=/capstor/scratch/cscs/mmeireles/models/Qwen3-32B-base
LORA=/capstor/scratch/cscs/mmeireles/models/qwen3-32b-donorsim-loras/step_175

echo "=== node: $(hostname) ==="
echo "=== serving on $(hostname):8000 ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Egress probe: the harness must reach api.digbench.ai. If a compute node cannot,
# the harness runs on the login node instead and only vLLM stays here.
curl -s -m 15 -o /dev/null -w "EGRESS digbench: HTTP %{http_code}\n" \
    https://api.digbench.ai/api/agent/games || echo "EGRESS digbench: UNREACHABLE"

exec vllm serve "$BASE" \
    --served-model-name qwen3-32b-base \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    --enable-lora \
    --max-lora-rank 16 \
    --max-loras 1 \
    --lora-modules "qwen3-32b-donorsim=$LORA" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
