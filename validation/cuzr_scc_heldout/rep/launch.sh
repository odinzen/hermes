#!/bin/bash
# Independent replicas of the 6912-atom Cu-Zr liquids at 2300 K (new velocity seeds), 2 MPI ranks each.
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lammps-full
R=/mnt/c/Users/busta/Code/_wt_station2_llzo/entropy/binary_cuzr_eam
for t in 020 036 050 070; do
  for seed in 71031 90217; do
    f=$(ls $R/x${t}-fe-liq_x${t}.data-liquid-2300-0.*.data)
    OMP_NUM_THREADS=1 setsid nohup mpirun -np 2 --oversubscribe --mca pml ob1 --mca btl self,sm lmp -log rep_x${t}_s$seed.log -screen none \
      -var data $f -var tag rep_x${t}_s$seed -var T 2300 -var seed $seed -var pot ~/opt/lammps-mace/potentials/CuZr_mm.eam.fs \
      -in md_liquid_rep.lmp > rep_x${t}_s$seed.out 2>&1 < /dev/null &
  done
done
sleep 20
