"""Exploration, outside every verdict: harmonic phonons of the MACE-MP-0 crystals the frozen tests ran, to split the
error into the potential's part and the entropy route's part.

At 298.15 K these oxides are close to harmonic, so the harmonic entropy of the potential's own phonons at the MD cell is
what an exact route would give on that potential. Phonons at the measured volume show how much of the remaining error
is the potential's over-expansion. Runs in the WSL MACE venv:  python phonons.py TAG [TAG ...]  -> phonons_<tag>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixSymmetry
from ase.optimize import BFGS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

HERE = Path(__file__).resolve().parent
VIB = HERE.parent / "oxide_vibration"
sys.path.insert(0, str(VIB))
import run_oxides as ro  # noqa: E402

KB, HBAR, NA = 1.380649e-23, 1.054571817e-34, 6.02214076e23
R = KB * NA
T0 = 298.15
NA_CM3 = 1e-24 * NA


def calculator():
    from mace.calculators import mace_mp
    return mace_mp(model=str(ro.MODEL), device="cuda", default_dtype="float64", enable_cueq=False)


def md_unit_cell(tag):
    """The conventional cell with the shape and size of the 298.15 K NPT average of the production run."""
    unit = ro.read(VIB / "structures" / f"{tag}.cif")
    widths, n = ro.repeats(unit.cell)
    while len(unit) * n.prod() < ro.MIN_ATOMS:
        n[np.argmin(n * widths)] += 1
    ladder = json.loads((VIB / "runs" / tag / "ladder.json").read_text())
    cell = np.array(ladder[0]["mean_cell"]) / n[:, None]
    unit.set_cell(cell, scale_atoms=True)
    return unit, n


def relax_positions(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    atoms.set_constraint(FixSymmetry(atoms, symprec=0.01))
    BFGS(atoms, logfile=None).run(fmax=1e-4, steps=500)
    atoms.set_constraint()
    return atoms


def harmonic(atoms, n, calc, mesh=(16, 16, 16)):
    cell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())
    ph = Phonopy(cell, supercell_matrix=np.diag(n), primitive_matrix="auto")
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    ph.forces = forces
    ph.produce_force_constants()
    ph.run_mesh(mesh, is_gamma_center=False)
    m = ph.get_mesh_dict()
    nu = m["frequencies"] * 1e12  # THz -> Hz
    w = m["weights"][:, None] * np.ones_like(nu)
    good = nu > 1e9
    x = 2 * np.pi * HBAR * nu[good] / (KB * T0)
    wq = w[good] / w.sum() * nu.shape[1]  # modes per primitive cell, the cell phonopy samples
    s_q = np.sum(wq * (x / np.expm1(x) - np.log1p(-np.exp(-x)))) * R
    s_cl = np.sum(wq * (1 - np.log(x))) * R
    natoms = len(ph.primitive)
    return {"S_q": s_q / natoms, "S_cl": s_cl / natoms, "min_THz": float(m["frequencies"].min()),
            "displacements": len(forces)}


def main():
    calc = calculator()
    ox = json.loads((VIB / "oxide_results.json").read_text())["cases"]
    arm_a = json.loads((HERE.parent / "extra_terms" / "arm_a_results.json").read_text())["cases"]
    sys.path.insert(0, str(HERE.parent / "sol_bench"))
    import volumes  # matches the phase too; TiO2 has rutile and anatase
    for tag in sys.argv[1:]:
        r = ox.get(tag) or arm_a.get(tag)
        p = r["prediction"]
        unit, n = md_unit_cell(tag)
        apf = volumes.OXIDES[tag][1]
        out = {"tag": tag, "hermes_S_cl_T0": p["S_cl_T0"], "hermes_S_vib_T0": p["S_vib_T0"],
               "vdos_harmonic_T0": p["harmonic_quantum_T0"], "measured_S298": (r.get("measured") or {}).get("S298")}
        md = relax_positions(unit, calc)
        out["md_volume"] = md.get_volume() / len(md) * apf * NA_CM3
        out["at_md_cell"] = harmonic(md, n, calc)
        out["exp_volume"] = volumes.measured_volume(tag)
        scaled = unit.copy()
        scaled.set_cell(unit.cell.array * (out["exp_volume"] / out["md_volume"]) ** (1 / 3), scale_atoms=True)
        out["at_exp_volume"] = harmonic(relax_positions(scaled, calc), n, calc)
        (HERE / f"phonons_{tag}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out))


if __name__ == "__main__":
    main()
