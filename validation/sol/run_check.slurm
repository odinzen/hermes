#!/bin/bash
# SLURM body of the check job (one GPU): the cuEquivariance parity gate and the force check against the local GPU.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
unset PYTHONPATH PYTHONHOME
PY="$(cat env_python.txt 2>/dev/null || true)"
# Sol's /lib64/libstdc++ is older than the one numpy was built against; the env's own copy must be found first.
[ -n "$PY" ] && export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -z "$PY" ] || ! "$PY" -c "import mace" 2>/dev/null; then
    echo "ERROR: no working interpreter in env_python.txt ('$PY'); rerun setup_env.sh on a login node"; exit 2
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "== check | host=$(hostname) start=$(date -Is) | python=$PY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PY" sol_run.py check "$RESULTS"
rc=$?
echo "end=$(date -Is) rc=$rc"
[ $rc -eq 0 ] && echo "STATUS check: JOB DONE" || echo "STATUS check: FAILED rc=$rc"
exit $rc
