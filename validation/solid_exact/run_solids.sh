#!/bin/bash
# Twelve EAM crystals for the solid-entropy test: MD frames and mean energy, then calphy Frenkel-Ladd free energy.
# Four cases at a time, 4 MPI ranks each. Resumable: finished steps are skipped.
set -u
cd "$(dirname "$0")"
P=$HOME/opt/lammps-mace/potentials
MPI="--oversubscribe --mca pml ob1 --mca btl self,sm"
export OMP_NUM_THREADS=1
CASES="fe_0700 fe_1400 w_1200 w_2400 mg_0370 mg_0740 zr_0450 zr_0900 al_0370 al_0740 ni_0700 ni_1400"

params() {
  case $1 in
    fe_*) lat=bcc a=2.855 n="8 8 8" style=eam/fs coeff="* * $P/Fe_mm.eam.fs Fe" el=Fe mass=55.845 ;;
    w_*)  lat=bcc a=3.165 n="8 8 8" style=eam/alloy coeff="* * $P/W_zhou.eam.alloy W" el=W mass=183.84 ;;
    mg_*) lat=hcp a=3.184 n="8 5 5" style=eam/fs coeff="* * $P/Mg_mm.eam.fs Mg" el=Mg mass=24.305 ;;
    zr_*) lat=hcp a=3.220 n="8 5 5" style=eam/fs coeff="* * $P/Zr_mm.eam.fs Zr" el=Zr mass=91.224 ;;
    al_*) lat=fcc a=4.045 n="6 6 6" style=eam/fs coeff="* * $P/Al_mm.eam.fs Al" el=Al mass=26.98154 ;;
    ni_*) lat=fcc a=3.520 n="6 6 6" style=eam coeff="* * $P/Ni_u3.eam" el=Ni mass=58.71 ;;
  esac
  t=$((10#${1#*_}))
  read -r nx ny nz <<< "$n"
}

one() {
  local tag=$1
  params "$tag"
  source ~/miniforge3/etc/profile.d/conda.sh
  if ! grep -q "Total wall time" "$tag.log" 2>/dev/null; then
    conda activate lammps-full
    mpirun -np 4 $MPI lmp -log "$tag.log" -screen none -var tag "$tag" -var lat $lat -var a $a -var nx $nx -var ny $ny \
      -var nz $nz -var T $t -var pstyle $style -var pcoeff "$coeff" -in solid.lmp > "$tag.out" 2>&1 \
      || { echo "$tag md FAILED"; return 1; }
  fi
  if ! ls "$tag"-fe-*/report.yaml > /dev/null 2>&1; then
    cat > "calphy_$tag.yaml" <<EOF
calculations:
- mode: fe
  lattice: start_$tag.data
  reference_phase: solid
  element: [$el]
  mass: [$mass]
  temperature: $t
  pressure: 0
  npt: true
  pair_style: [$style]
  pair_coeff: ["$coeff"]
  n_equilibration_steps: 25000
  n_switching_steps: ${SWITCH:-50000}
  n_iterations: 1
  lammps_executable: $HOME/miniforge3/envs/lammps-full/bin/lmp_mpi
  mpi_executable: $HOME/miniforge3/envs/lammps-full/bin/mpirun
  folder_prefix: $tag
  queue:
    scheduler: local
    cores: 4
EOF
    conda activate ti
    calphy_kernel -i "calphy_$tag.yaml" -k 0 > "calphy_$tag.out" 2>&1 || { echo "$tag calphy FAILED"; return 1; }
  fi
  echo "$tag done $(date -Iseconds)"
}

if [ "${1:-}" = one ]; then one "$2"; exit; fi
printf '%s\n' $CASES | xargs -P "${LANES:-4}" -I{} bash "$0" one {}
echo ALL_DONE "$(date -Iseconds)"
