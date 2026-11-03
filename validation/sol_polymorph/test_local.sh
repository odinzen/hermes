#!/bin/bash
# Rehearse the Sol setup and both job bodies in WSL before shipping: a stand-in for Sol's `module` command and the
# SLURM variables, the hermes-mace env built from env.yml, another conda env active in the submitting shell, and no
# network for the jobs. Runs the check, then stishovite with the smoke lengths. Outputs go beside the package.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
export PATH="$HOME/miniforge3/bin:$PATH"   # stands in for 'module load mamba/latest', which puts conda on PATH
module() { :; }
export -f module
bash setup_env.sh

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate base
export HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9
export SLURM_SUBMIT_DIR="$here" SLURM_CPUS_PER_TASK=4 RESULTS="$(dirname "$here")/hermes-polymorph-sol-localtest"
rm -rf "$RESULTS"
mkdir -p "$RESULTS"
bash run_check.slurm
printf 'mpa-0 stishovite
' > "$RESULTS/cases_smoke.txt"
for line in 1; do CASES="$RESULTS/cases_smoke.txt" SLURM_ARRAY_TASK_ID=$line SMOKE=1 bash run_case.slurm; done
ls -la "$RESULTS"/*/runs/stishovite
