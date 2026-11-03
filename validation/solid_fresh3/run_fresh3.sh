#!/bin/bash
# Third fresh solid-entropy test: per metal an anchor run at T0, a run at T, a seven-step NPT enthalpy ladder from T0 to
# T, and calphy at T and T0. Four metals at a time, 4 MPI ranks per run. Resumable: finished steps are skipped.
set -u
cd "$(dirname "$0")"
P=$HOME/opt/lammps-mace/potentials
MPI="--oversubscribe --mca pml ob1 --mca btl self,sm"
export OMP_NUM_THREADS=1
CASES="fe v pt ni ag au cum mg al"

params() {
  case $1 in
    ag)  lat=fcc a=4.090 n="6 6 6" style=eam coeff="* * $P/Ag_u3.eam" el=Ag mass=107.87 t0=430 t=1000 ;;
    pt)  lat=fcc a=3.920 n="6 6 6" style=eam coeff="* * $P/Pt_u3.eam" el=Pt mass=195.09 t0=710 t=1600 ;;
    au)  lat=fcc a=4.080 n="6 6 6" style=eam coeff="* * $P/Au_u3.eam" el=Au mass=196.97 t0=470 t=1080 ;;
    cum) lat=fcc a=3.615 n="6 6 6" style=eam/alloy coeff="* * $P/Cu_mishin1.eam.alloy Cu" el=Cu mass=63.55 t0=480 t=1150 ;;
    mg)  lat=hcp a=3.184 n="8 5 5" style=eam/fs coeff="* * $P/Mg_mm.eam.fs Mg" el=Mg mass=24.305 t0=320 t=780 ;;
    al)  lat=fcc a=4.045 n="6 6 6" style=eam/fs coeff="* * $P/Al_mm.eam.fs Al" el=Al mass=26.98154 t0=330 t=780 ;;
    fe)  lat=bcc a=2.855 n="8 8 8" style=eam/fs coeff="* * $P/Fe_mm.eam.fs Fe" el=Fe mass=55.845 t0=630 t=1550 ;;
    v)   lat=bcc a=3.030 n="8 8 8" style=eam/fs coeff="* * $P/VFe_mm.eam.fs V" el=V mass=50.9415 t0=760 t=1850 ;;
    ni)  lat=fcc a=3.520 n="6 6 6" style=eam coeff="* * $P/Ni_u3.eam" el=Ni mass=58.71 t0=600 t=1450 ;;
  esac
  read -r nx ny nz <<< "$n"
}

lmp_run() {  # tag input temperature
  grep -q "Total wall time" "$1.log" 2>/dev/null && return 0
  mpirun -np 4 $MPI lmp -log "$1.log" -screen none -var tag "$1" -var lat $lat -var a $a -var nx $nx -var ny $ny \
    -var nz $nz -var T $3 -var pstyle $style -var pcoeff "$coeff" -in "$2" > "$1.out" 2>&1
}

calphy_run() {  # run temperature
  ls "$1"-fe-*/report.yaml > /dev/null 2>&1 && return 0
  cat > "calphy_$1.yaml" <<EOF
calculations:
- mode: fe
  lattice: start_$1.data
  reference_phase: solid
  element: [$el]
  mass: [$mass]
  temperature: $2
  pressure: 0
  npt: true
  pair_style: [$style]
  pair_coeff: ["$coeff"]
  n_equilibration_steps: 25000
  n_switching_steps: ${SWITCH:-50000}
  n_iterations: 1
  lammps_executable: $HOME/miniforge3/envs/lammps-full/bin/lmp_mpi
  mpi_executable: $HOME/miniforge3/envs/lammps-full/bin/mpirun
  folder_prefix: $1
  queue:
    scheduler: local
    cores: 4
EOF
  conda activate ti
  calphy_kernel -i "calphy_$1.yaml" -k 0 > "calphy_$1.out" 2>&1
  local status=$?
  conda activate lammps-full
  return $status
}

one() {
  local c=$1
  params "$c"
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate lammps-full
  lmp_run "${c}_a" solid.lmp $t0 || { echo "$c anchor FAILED"; return 1; }
  lmp_run "${c}_t" solid.lmp $t || { echo "$c target FAILED"; return 1; }
  local k=0
  # seven temperatures evenly from T0 to T, rounded as analyze_fresh3.ladder rounds them (no step falls on a half)
  for tk in $(awk -v a=$t0 -v b=$t 'BEGIN { for (k = 0; k < 7; k++) printf "%d ", int(a + k * (b - a) / 6 + 0.5) }'); do
    lmp_run "${c}_h$k" npt.lmp $tk || { echo "$c ladder $k FAILED"; return 1; }
    k=$((k + 1))
  done
  calphy_run "${c}_t" $t || { echo "$c calphy T FAILED"; return 1; }
  calphy_run "${c}_a" $t0 || { echo "$c calphy T0 FAILED"; return 1; }
  echo "$c done $(date -Iseconds)"
}

if [ "${1:-}" = one ]; then one "$2"; exit; fi
printf '%s\n' $CASES | xargs -P "${LANES:-4}" -I{} bash "$0" one {}
echo ALL_DONE "$(date -Iseconds)"
