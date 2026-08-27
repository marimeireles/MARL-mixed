#!/bin/bash
#SBATCH --job-name=llmdyn-extras
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-extras-%j.log
#SBATCH --error=/nas/ucb/marimeireles/MARL-mixed/llm_dynamics/logs/suite-extras-%j.err
#SBATCH --export=ALL
# Run only the v3 stages whose outputs are missing (perturb / self-play / group).
set -uo pipefail; cd /nas/ucb/marimeireles/MARL-mixed
TAG="${TAG:?}"; MODEL="${MODEL:?}"; ENDPOINT_FILE="${ENDPOINT_FILE:?}"
for _ in $(seq 1 180); do [ -f "${ENDPOINT_FILE}.ready" ] && break; sleep 20; done
API=$(cat "$ENDPOINT_FILE"); export JAX_PLATFORMS=cpu
RUN="/nas/ucb/marimeireles/MARL-mixed/.venv-jaxmarl/bin/python -m llm_dynamics.run_experiments"; COMMON="--api-base $API --model $MODEL"; R=llm_dynamics/results
n=$(find $R/${TAG}_donors_v3_perturb -name '*.jsonl' 2>/dev/null | wc -l); [ "$n" -ge 20 ] || { echo "== perturb =="; $RUN donors-perturb $COMMON --seeds 4 --rounds 20 --perturb-round 8 --out ${TAG}_donors_v3_perturb; }
n=$(find $R/${TAG}_selfplay_v3 -name '*.jsonl' 2>/dev/null | wc -l); [ "$n" -ge 16 ] || { echo "== self-play =="; $RUN donors-selfplay $COMMON --q-values 0,1.0 --w-values 0,1.0 --seeds 4 --rounds 20 --out ${TAG}_selfplay_v3; }
n=$(find $R/${TAG}_group_v3 -name '*.jsonl' 2>/dev/null | wc -l); [ "$n" -ge 100 ] || { echo "== group =="; $RUN group-eval $COMMON --thinking --with-think --max-tokens 4096 --temperature 0.8 --scenarios 50 --seeds 2 --out ${TAG}_group_v3; }
echo "== DONE $(date -Iseconds) =="
