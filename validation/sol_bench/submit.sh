#!/bin/bash
# Submit from the package folder on Sol. Runs the check job first unless a passing check.json is already in the
# results folder; the case array waits for it (afterok) and only cases without CASE_DONE are submitted.
#   bash submit.sh          # check if needed, then every unfinished model and case
#   bash submit.sh 3,7      # lines 3 and 7 of cases.txt regardless
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
source sol_config.sh
# Results and logs live beside this folder, so replacing the whole package folder never deletes finished runs.
results="${HERMES_RESULTS:-$(dirname "$here")/hermes-bench-sol-results}"
mkdir -p "$results/logs"
if command -v squeue > /dev/null && squeue -u "$USER" -h -o %j | grep -q '^hermes-bench'; then
    echo "hermes-bench jobs are still queued or running (squeue -u \$USER); submit again once they have finished"; exit 1
fi
lines="${1:-}"
if [ -z "$lines" ]; then
    lines=$(awk '{ gsub(/\r/, "") } NF { print NR, $1, $2 }' cases.txt | while read -r i model tag; do
        grep -qs CASE_DONE "$results/$model/runs/$tag/progress.log" || echo "$i"; done | paste -sd, -)
fi
if [ -z "$lines" ]; then echo "every case has finished; results in $results"; exit 0; fi

common=(--parsable --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" --gres="$GPU_GRES"
        --cpus-per-task="$CPUS" --mem="$MEM" --export=ALL,RESULTS="$results")
dependency=()
if ! grep -qs '^ "verdict": "PASS"' "$results/check.json"; then
    check=$(sbatch "${common[@]}" --time="$CHECK_TIME" --job-name=hermes-bench-check \
        --output="$results/logs/check_%j.out" run_check.slurm | cut -d';' -f1)  # --parsable prints id;cluster
    echo "Submitted check job $check"
    dependency=(--dependency=afterok:"$check")
fi
cases=$(sbatch "${common[@]}" "${dependency[@]}" --time="$TIME" --array="$lines" --job-name=hermes-bench \
    --output="$results/logs/case_%A_%a.out" run_case.slurm | cut -d';' -f1)
echo "Submitted case array $cases (lines $lines of cases.txt) -> $results/<model>/runs/"
