#!/bin/bash
# Twelve EAM crystals for the second fresh solid-entropy test (200 ps, 401 frames), then calphy Frenkel-Ladd free energy.
# Four cases at a time, 4 MPI ranks each. Resumable: finished steps are skipped.
set -u
cd "$(dirname "$0")"
P=$HOME/opt/lammps-mace/potentials
MPI="--oversubscribe --mca pml ob1 --mca btl self,sm"
export OMP_NUM_THREADS=1
CASES="cum_0900_2048 zr_0600_2352 cum_0900_0864 cum_0900_0256 zr_0600_0384 fe_0500_1024 v_0800_1024 au_0800_0864 cuf_1200_0864 al_0800_0864 ni_1500_0864 pd_1300_0864"

params() {
  local size=${1##*_}
  case $1 in
    cum_*) lat=fcc a=3.615 style=eam/alloy coeff="* * $P/Cu_mishin1.eam.alloy Cu" el=Cu mass=63.55 ;;
    cuf_*) lat=fcc a=3.615 style=eam coeff="* * $P/Cu_u3.eam" el=Cu mass=63.55 ;;
    zr_*)  lat=hcp a=3.220 style=eam/fs coeff="* * $P/Zr_mm.eam.fs Zr" el=Zr mass=91.224 ;;
    au_*)  lat=fcc a=4.080 style=eam coeff="* * $P/Au_u3.eam" el=Au mass=196.97 ;;
    fe_*)  lat=bcc a=2.855 style=eam/fs coeff="* * $P/Fe_mm.eam.fs Fe" el=Fe mass=55.845 ;;
    v_*)   lat=bcc a=3.030 style=eam/fs coeff="* * $P/VFe_mm.eam.fs V" el=V mass=50.9415 ;;
    al_*)  lat=fcc a=4.045 style=eam/fs coeff="* * $P/Al_mm.eam.fs Al" el=Al mass=26.98154 ;;
    ni_*)  lat=fcc a=3.520 style=eam coeff="* * $P/Ni_u3.eam" el=Ni mass=58.71 ;;
    pd_*)  lat=fcc a=3.890 style=eam coeff="* * $P/Pd_u3.eam" el=Pd mass=106.40 ;;
  esac
  case $lat:$size in
    fcc:0256) n="4 4 4" ;; fcc:0864) n="6 6 6" ;; fcc:2048) n="8 8 8" ;;
    bcc:1024) n="8 8 8" ;; hcp:0384) n="6 4 4" ;; hcp:2352) n="12 7 7" ;;
  esac
  local rest=${1#*_}
  t=$((10#${rest%%_*}))
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
