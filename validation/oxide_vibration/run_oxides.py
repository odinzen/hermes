"""MD for the closed-shell oxide vibration test, with the settings RULE.md fixes. Runs in the WSL MACE venv.

    python run_oxides.py TAG [TAG ...]     production runs, resumable stage by stage
    python run_oxides.py --parity          cuEquivariance force parity on one CaO frame (gate before production)
    python run_oxides.py --smoke           CaO with every stage shortened, plumbing only

Per case, runs/<tag>/ gets ladder.json (NPT averages at seven temperatures), npt_<T>.traj (last frame of each
NPT run), and nvt_t0.npz / nvt_t1.npz (positions every 0.25 ps, velocities every 4 fs, the cell, masses, species).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.build import make_supercell
from ase.io import read, write
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

HERE = Path(__file__).resolve().parent
MODEL = Path.home() / ".cache/mace/20231203mace128L1_epoch199model"
T0, T1 = 298.15, 1000.0
LADDER = np.linspace(T0, T1, 7)
SEED = 4928
DT = 1.0 * units.fs
MIN_WIDTH, MIN_ATOMS = 12.0, 200

FULL = {"npt_settle": 10_000, "npt_average": 30_000, "nvt_settle": 10_000, "nvt_t0": 100_000, "nvt_t1": 30_000,
        "sample_every": 10, "position_every": 250, "velocity_every": 4}
SMOKE = {**FULL, "npt_settle": 200, "npt_average": 400, "nvt_settle": 200, "nvt_t0": 2_000, "nvt_t1": 1_000,
         "position_every": 50}


def calculator(cueq=True):
    from mace.calculators import mace_mp
    return mace_mp(model=str(MODEL), device="cuda", default_dtype="float32", enable_cueq=cueq)


def repeats(cell):
    """Smallest repeats giving every perpendicular width >= 12 A, then more along the narrowest until >= 200 atoms."""
    cell = np.asarray(cell)
    volume = abs(np.linalg.det(cell))
    widths = np.array([volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)])
    return widths, np.ceil(MIN_WIDTH / widths).astype(int)


def build_box(tag):
    unit = read(HERE / "structures" / f"{tag}.cif")
    widths, n = repeats(unit.cell)
    while len(unit) * n.prod() < MIN_ATOMS:
        n[np.argmin(n * widths)] += 1
    return make_supercell(unit, np.diag(n))


def pressure_volume(atoms):
    pressure = -np.trace(atoms.get_stress(voigt=False, include_ideal_gas=True)) / 3
    return pressure * atoms.get_volume() / len(atoms)


def run_npt(atoms, temperature, steps, out):
    dyn = MTKNPT(atoms, timestep=DT, temperature_K=temperature, pressure_au=0.0, tdamp=100 * units.fs,
                 pdamp=1000 * units.fs)
    dyn.run(steps["npt_settle"])
    pe, pv, cells, kin = [], [], [], []
    for _ in range(steps["npt_average"] // steps["sample_every"]):
        dyn.run(steps["sample_every"])
        pe.append(atoms.get_potential_energy() / len(atoms))
        pv.append(pressure_volume(atoms))
        kin.append(atoms.get_temperature())
        cells.append(atoms.cell.array.copy())
    write(out, atoms)
    return {"T": float(temperature), "pe_eV_atom": float(np.mean(pe)), "pv_eV_atom": float(np.mean(pv)),
            "pe_sem": float(np.std(pe) / np.sqrt(len(pe))), "T_kinetic_mean": float(np.mean(kin)),
            "mean_cell": np.mean(cells, axis=0).tolist(), "samples": len(pe)}


def run_nvt(atoms, temperature, mean_cell, production, steps, out):
    atoms.set_cell(mean_cell, scale_atoms=True)
    dyn = Bussi(atoms, timestep=DT, temperature_K=temperature, taut=100 * units.fs,
                rng=np.random.default_rng(SEED))
    dyn.run(steps["nvt_settle"])
    positions, velocities = [atoms.get_positions().astype(np.float32)], []
    for step in range(1, production + 1):
        dyn.run(1)
        if step % steps["velocity_every"] == 0:
            velocities.append(atoms.get_velocities().astype(np.float32))
        if step % steps["position_every"] == 0:
            positions.append(atoms.get_positions().astype(np.float32))
    np.savez(out, positions=np.array(positions), velocities=np.array(velocities), cell=atoms.cell.array,
             masses=atoms.get_masses(), numbers=atoms.get_atomic_numbers(), temperature=temperature,
             position_dt_ps=steps["position_every"] * 1e-3, velocity_dt_ps=steps["velocity_every"] * 1e-3)


def run_case(tag, steps, cueq=True, root=HERE / "runs"):
    folder = root / tag
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / "progress.log").open("a")
    stamp = lambda msg: (log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n"), log.flush())  # noqa: E731
    atoms = build_box(tag)
    atoms.calc = calculator(cueq)
    stamp(f"start {tag}: {len(atoms)} atoms")
    ladder_file = folder / "ladder.json"
    ladder = json.loads(ladder_file.read_text()) if ladder_file.exists() else []
    if ladder:
        atoms = read(folder / f"npt_{ladder[-1]['T']:.2f}.traj")
        atoms.calc = calculator(cueq)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=np.random.default_rng(SEED))
        Stationary(atoms)
    for temperature in LADDER[len(ladder):]:
        ladder.append(run_npt(atoms, temperature, steps, folder / f"npt_{temperature:.2f}.traj"))
        ladder_file.write_text(json.dumps(ladder, indent=1))
        stamp(f"npt {temperature:.2f} K done, <T> {ladder[-1]['T_kinetic_mean']:.1f}")
    for name, temperature, entry, production in (("nvt_t0", T0, ladder[0], steps["nvt_t0"]),
                                                  ("nvt_t1", T1, ladder[-1], steps["nvt_t1"])):
        out = folder / f"{name}.npz"
        if out.exists():
            continue
        start = read(folder / f"npt_{temperature:.2f}.traj")
        start.calc = calculator(cueq)
        run_nvt(start, temperature, np.array(entry["mean_cell"]), production, steps, out)
        stamp(f"{name} done")
    stamp("CASE_DONE")


def parity():
    atoms = build_box("cao")
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0, rng=np.random.default_rng(SEED))
    atoms.rattle(0.05, seed=SEED)
    forces = []
    for cueq in (False, True):
        atoms.calc = calculator(cueq)
        forces.append(atoms.get_forces())
    worst = float(np.abs(forces[0] - forces[1]).max())
    verdict = "PASS" if worst <= 1e-3 else "FAIL"
    result = {"max_force_difference_eV_A": worst, "tolerance": 1e-3, "verdict": verdict, "atoms": len(atoms)}
    (HERE / "parity.json").write_text(json.dumps(result, indent=1))
    print(result)
    return verdict == "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="*")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-cueq", action="store_true", help="only if the parity gate failed")
    args = parser.parse_args()
    if args.parity:
        sys.exit(0 if parity() else 1)
    if args.smoke:
        run_case("cao", SMOKE, cueq=not args.no_cueq, root=HERE / "smoke")
        return
    for tag in args.tags:
        run_case(tag, FULL, cueq=not args.no_cueq)


if __name__ == "__main__":
    main()
