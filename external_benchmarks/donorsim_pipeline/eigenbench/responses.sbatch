#!/bin/bash
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --job-name=eigen-responses
#SBATCH --output=/capstor/scratch/cscs/mmeireles/EigenBench/runs/donorsim/responses-%j.out
#SBATCH --environment=/capstor/scratch/cscs/mmeireles/EigenBench/runs/donorsim/edf-eigenbench.toml

# Responses only -- the judging half is being done externally by an OSS GPT model.
#
# 400 scenarios x 4 models = 1,600 generations against the ~27,200 completions a
# single constitution's full 3-phase run needed. Four engine loads instead of nine
# per constitution, and none of the pipeline's broken checkpointing is in the path:
# gen_responses.py appends per (model, sample) block and skips triples already on
# disk, so this job is resumable by simply resubmitting it.
#
# 6h rather than 12h: Phase 1 for 6 scenarios took ~3 min per 100-prompt batch under
# TP=1, and this now runs TP=4 with CUDA graphs. 6h is generous; if it somehow is
# not, the run resumes where it stopped.
set -uo pipefail

ROOT=/capstor/scratch/cscs/mmeireles/EigenBench
cd "$ROOT" || exit 1

echo "=== responses on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VENV=$ROOT/.venv-container
SYSPY=$(command -v python3 || command -v python) || { echo "FATAL: no python"; exit 1; }
if [ ! -x "$VENV/bin/python" ]; then
    "$SYSPY" -m venv --system-site-packages "$VENV" || exit 1
    "$VENV/bin/pip" install -q --no-cache-dir \
        scikit-learn matplotlib tqdm python-dotenv scipy pandas gradio_client \
        || echo "WARN: some deps failed to install"
fi
PY="$VENV/bin/python"

$PY -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)" \
    || { echo "FATAL: vllm not importable"; exit 1; }

echo "=== generating ==="
SCENARIOS=${SCENARIOS:-400} SAMPLES=${SAMPLES:-1} START=${START:-0} \
  "$PY" runs/donorsim/gen_responses.py
RC=$?
echo "=== exited rc=$RC at $(date) ==="

F=$ROOT/runs/donorsim/responses/responses.jsonl
[ -f "$F" ] && echo "responses.jsonl: $(wc -l < "$F") records, $(stat -c%s "$F") bytes"
exit $RC
