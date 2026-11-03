#!/bin/bash
# Run on a Sol LOGIN node (it has internet; compute nodes do not): builds the conda env once, records its interpreter
# for the batch jobs, and checks run_oxides.py and the weights are the frozen files. Safe to rerun after every upload.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/sol_config.sh"
module load "$MAMBA_MODULE"
if ! conda env list | grep -q "^$ENV_NAME "; then
    mamba env create -y -f "$here/env.yml"
fi
eval "$(conda shell.bash hook 2>/dev/null)" 2>/dev/null || true
conda activate "$ENV_NAME" 2>/dev/null || source activate "$ENV_NAME" 2>/dev/null || true
# Sol's /lib64/libstdc++ is older than the one numpy was built against; the env's own copy must be found first.
export LD_LIBRARY_PATH="$(python -c 'import sys; print(sys.prefix)')/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Light imports only: a login node kills a process that loads the GPU libraries (memory limit). The check job loads
# everything on a GPU node before any case starts.
if ! python -c "import numpy, ase, mace"; then
    echo "ERROR: environment $ENV_NAME is not active or incomplete (python: $(command -v python || echo none))"; exit 2
fi
# The jobs call this interpreter directly, so an env active in the submitting shell cannot leak in.
python -c "import sys; print(sys.executable)" > "$here/env_python.txt"
cd "$here"
python - <<'PY'
import importlib.metadata as md
import sol_run
sol_run.check_frozen()  # run_oxides.py and the weights are the frozen files
print(" | ".join(f"{p} {md.version(p)}" for p in ("torch", "ase", "mace-torch", "cuequivariance-torch")))
PY
echo "interpreter: $(cat "$here/env_python.txt")"
echo "SETUP OK"
