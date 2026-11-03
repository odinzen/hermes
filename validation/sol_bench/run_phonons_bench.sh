#!/usr/bin/env bash
# Phonon check of all three models on the local GPU, one after another. Run from WSL.
cd "$(dirname "$0")"
PY=~/scratch/mace-gpu/bin/python
ARCHIVE=/mnt/c/Users/busta/Code/juno/assets/local-archive/hermes-validation/data/bench-sol-2026-10-27/results
[ "${1:-}" = "--skip-mp0" ] || $PY phonons_bench.py mp-0 ../oxide_vibration/runs mgo al2o3 tio2 nio mno coo fe2o3 cr2o3
for m in mpa-0 mp-0b3; do
  $PY phonons_bench.py $m "$ARCHIVE/$m/runs" mgo tio2 nio mno coo fe2o3 cr2o3
done
echo PHONONS_DONE
