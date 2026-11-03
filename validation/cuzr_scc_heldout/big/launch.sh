#!/bin/bash
# Larger-box (6912-atom) liquid Cu-Zr at 2300 K, 3 MPI ranks per run; x000 doubles as the random-label control.
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lammps-full
R=/mnt/c/Users/busta/Code/_wt_station2_llzo/entropy/binary_cuzr_eam
for t in 000 020 036 050 070; do
  f=$(ls $R/x${t}-fe-liq_x${t}.data-liquid-2300-0.*.data)
  OMP_NUM_THREADS=1 setsid nohup mpirun -np 3 --oversubscribe --mca pml ob1 --mca btl self,sm lmp -log big_x$t.log -screen none \
    -var data $f -var tag big_x$t -var T 2300 -var pot ~/opt/lammps-mace/potentials/CuZr_mm.eam.fs \
    -in md_liquid_big.lmp > big_x$t.out 2>&1 < /dev/null &
done
sleep 30
ps -eo args | grep -c '[l]mp -log big_'
