#!/bin/bash
#SBATCH --job-name=bench-eigen-judge
#SBATCH --partition=main
#SBATCH --gpus=2
#SBATCH --exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-eigen-judge-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-eigen-judge-%j.err
#SBATCH --export=ALL
# EigenBench external judging: Gemma 4 31B-it (TP=2, fits 2x A100-80GB or 2x A6000)
# judges base vs donorSim-RL responses, both orders, over a constitution battery.
# Builds the paired file from the two response files first.
#   sbatch --dependency=afterok:<responses job> external_benchmarks/slurm/bench_eigen_judge_gemma4.sbatch
set -uo pipefail
ROOT=/nas/ucb/marimeireles/MARL-mixed; cd $ROOT/external_benchmarks/EigenBench
export HF_HOME=/nas/ucb/marimeireles/cache HF_HUB_OFFLINE=1 OPENAI_API_KEY=dummy TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export XDG_CACHE_HOME=/nas/ucb/marimeireles/cache/xdg VLLM_CACHE_ROOT=/nas/ucb/marimeireles/cache/vllm \
       TORCHINDUCTOR_CACHE_DIR=/nas/ucb/marimeireles/cache/inductor TRITON_CACHE_DIR=/nas/ucb/marimeireles/cache/triton
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR" $XDG_CACHE_HOME $VLLM_CACHE_ROOT $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
export BASE="${BASE:-runs/qwen8b/responses/responses.jsonl}" RL="${RL:-runs/qwen8b/responses/responses_rl_step6.jsonl}"
export SRC="${SRC:-runs/qwen8b/responses/arms_head_to_head_step6.jsonl}" OUT_DIR="${OUT_DIR:-runs/qwen8b/judgments_gemma4_step6}"
export CONSTS="${CONSTS:-kindness oct_goodness oct_misalignment oct_sycophancy oct_humor oct_poeticism}"
export SCENARIOS="${SCENARIOS:-0}" TP="${TP:-2}" CHUNK="${CHUNK:-1000}"
echo "judge gemma4  base=$BASE rl=$RL -> $SRC -> $OUT_DIR  consts=[$CONSTS] scenarios=$SCENARIOS ($(date -Iseconds))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OUT="$SRC" $ROOT/.venv-eigen/bin/python runs/qwen8b/build_head_to_head.py || { echo "pairing failed"; exit 1; }
$ROOT/.venv-eigen/bin/python runs/qwen8b/judge_gemma4.py
echo "== DONE $(date -Iseconds) =="
