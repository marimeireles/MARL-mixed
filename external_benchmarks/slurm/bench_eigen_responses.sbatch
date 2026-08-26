#!/bin/bash
#SBATCH --job-name=bench-eigen-resp
#SBATCH --partition=main
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-eigen-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/bench-eigen-%j.err
#SBATCH --export=ALL
# EigenBench evaluee responses (no judging) for the models in SPEC, via in-process vLLM.
#   SPEC=runs/qwen8b/spec.py SCENARIOS=10398 sbatch external_benchmarks/slurm/bench_eigen_responses.sbatch
set -uo pipefail
ROOT=/nas/ucb/marimeireles/MARL-mixed; cd $ROOT/external_benchmarks/EigenBench
export HF_HOME=/nas/ucb/marimeireles/cache OPENAI_API_KEY=dummy TOKENIZERS_PARALLELISM=false
export SPEC="${SPEC:-runs/qwen8b/spec.py}" OUT="${OUT:-runs/qwen8b/responses/responses.jsonl}"
export SCENARIOS="${SCENARIOS:-10398}" SAMPLES="${SAMPLES:-1}" MAX_TOKENS="${MAX_TOKENS:-4096}" TEMPERATURE="${TEMPERATURE:-0.7}"
echo "spec $SPEC -> $OUT scenarios $SCENARIOS samples $SAMPLES ($(date -Iseconds))"
$ROOT/.venv-eigen/bin/python runs/qwen8b/gen_responses.py
echo "== DONE $(date -Iseconds) =="
