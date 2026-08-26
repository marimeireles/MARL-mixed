#!/bin/bash
# Reproduce the external-benchmark setup (clones + data + our adaptations).
set -euo pipefail
cd "$(dirname "$0")"; ROOT=$(cd .. && pwd)
for r in jchang153/EigenBench discos-research/dig-bench aypan17/machiavelli; do d=$(basename $r); [ -d $d ] || git clone -q --depth 1 https://github.com/$r.git $d; done
# venvs: .venv-bench (py3.11, machiavelli + dig clients), .venv-eigen (on top of vllm-env)
[ -x $ROOT/.venv-bench/bin/python ] || { $ROOT/.venv-jaxmarl/bin/python -m venv $ROOT/.venv-bench; $ROOT/.venv-bench/bin/pip install -q gdown "openai>=1.0" "gym==0.25.1" numpy pandas pyrallis scikit-learn scipy networkx wandb "transformers>=4.23.1" asteval tqdm matplotlib seaborn plotly; }
[ -x $ROOT/.venv-eigen/bin/python ] || { /nas/ucb/marimeireles/vllm-env/bin/python -m venv --system-site-packages $ROOT/.venv-eigen; /nas/ucb/marimeireles/vllm-env/bin/python -c "import site;print(site.getsitepackages()[0])" > $($ROOT/.venv-eigen/bin/python -c "import site;print(site.getsitepackages()[0])")/vllmenv.pth; $ROOT/.venv-eigen/bin/pip install -q scikit-learn scipy pandas datasets matplotlib python-dotenv gradio_client; }
# MACHIAVELLI: game data (Google Drive, password 'machiavelli'), donorSim VLLMAgent + runner + coeffs
( cd machiavelli
  [ -d game_data ] || { $ROOT/.venv-bench/bin/gdown 19PXa2bgjkfFfTTI3EZIT3-IJ_vxrV0Rz -O game_data.zip && 7z x -pmachiavelli -y game_data.zip >/dev/null; }
  cp ../patched/machiavelli/vllm_agent.py machiavelli/agent/vllm_agent.py
  cp ../patched/machiavelli/run_arms.py ../patched/machiavelli/aggregate_machiavelli.py .
  git apply --check ../donorsim_pipeline/scripts/machiavelli_load_agent.patch 2>/dev/null && git apply ../donorsim_pipeline/scripts/machiavelli_load_agent.patch || true
  cp ../donorsim_pipeline/data/normalization_coeffs.json game_data/normalization_coeffs.json )
# DiG-bench: our orchestrator/aggregator (needs DIGBENCH_API_TOKEN at run time)
cp patched/dig-bench/orchestrate.py patched/dig-bench/aggregate_digbench.py dig-bench/
# EigenBench: scenarios + 8B run dir (responses only)
mkdir -p EigenBench/data/scenarios EigenBench/runs/qwen8b
cp donorsim_pipeline/eigenbench/airiskdilemmas.json EigenBench/data/scenarios/
cp patched/EigenBench/spec.py patched/EigenBench/gen_responses.py EigenBench/runs/qwen8b/
echo "external benchmarks ready"
