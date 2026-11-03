#!/bin/bash
# Second held-out S_cc test, six new Cu-Zr liquids at 2300 K (Mendelev EAM). Resumable: finished steps are skipped.
# 1. 864-atom liquid per composition (mean energy for the exact reference, start structure for calphy)
# 2. calphy free energy per composition
# 3. two 6912-atom runs per composition from the calphy-equilibrated liquid, JOBS at a time, 2 MPI ranks each
set -u
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
POT=~/opt/lammps-mace/potentials/CuZr_mm.eam.fs
MPI="--oversubscribe --mca pml ob1 --mca btl self,sm"
TAGS=(010 028 043 060 080 090)
NZR=(86 242 372 518 691 778)
JOBS=${JOBS:-3}
export OMP_NUM_THREADS=1
done_log() { grep -q "Total wall time" "$1" 2>/dev/null; }

conda activate lammps-full
for i in "${!TAGS[@]}"; do
  t=${TAGS[$i]}
  done_log ref_x$t.log && [ -s liq_x$t.data ] && continue
  mpirun -np 4 $MPI lmp -log ref_x$t.log -screen none -var tag x$t -var nzr ${NZR[$i]} -var lseed $((7100 + i)) \
    -var T 2300 -var pot $POT -in liquid_864.lmp > ref_x$t.out 2>&1 && echo "ref x$t done" || { echo "ref x$t FAILED"; exit 1; }
done

conda activate ti
for i in "${!TAGS[@]}"; do
  t=${TAGS[$i]}
  ls x$t-fe-*/report.yaml > /dev/null 2>&1 && continue
  calphy_kernel -i calphy_new.yaml -k $i > calphy_k$i.out 2>&1 && echo "calphy x$t done" || { echo "calphy x$t FAILED"; exit 1; }
done

conda activate lammps-full
# mpirun, prterun and every rank all carry the -log argument, so count distinct runs
running() { ps -eo args | grep -o "lmp -log big_x[0-9]*_s[0-9]*" | sort -u | wc -l; }
for t in "${TAGS[@]}"; do
  for seed in 71031 90217; do
    tag=big_x${t}_s$seed
    done_log $tag.log && continue
    while [ "$(running)" -ge "$JOBS" ]; do sleep 60; done
    f=$(ls x${t}-fe-liq_x${t}.data-liquid-2300-0.*.data)
    setsid nohup mpirun -np 2 $MPI lmp -log $tag.log -screen none -var data $f -var tag $tag -var T 2300 \
      -var seed $seed -var pot $POT -in md_liquid_rep.lmp > $tag.out 2>&1 < /dev/null &
    echo "launched $tag $(date -Iseconds)"
    sleep 30
  done
done
wait
while [ "$(running)" -gt 0 ]; do sleep 60; done
echo ALL_DONE $(date -Iseconds)
