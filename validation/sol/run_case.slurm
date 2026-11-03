#!/bin/bash
# SLURM array body: one case per task, line SLURM_ARRAY_TASK_ID of cases.txt (or $CASES). Submit with submit.sh.
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
tag="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CASES:-cases.txt}" | tr -d '\r ')"
if [ -z "$tag" ]; then echo "no case at line $SLURM_ARRAY_TASK_ID"; exit 1; fi
echo "== $tag | host=$(hostname) start=$(date -Is) | python=$PY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PY" sol_run.py case "$tag" "$RESULTS" ${SMOKE:+--smoke}
rc=$?
echo "end=$(date -Is) rc=$rc"
[ $rc -eq 0 ] && echo "STATUS $tag: JOB DONE" || echo "STATUS $tag: FAILED rc=$rc"
exit $rc
