"""Harmonic quantum entropy of MPA-0 by finite-displacement phonons, no Hermes code: at the relaxed volume, the MD
volume at 298.15 K and the measured volume. Diagnostic run 2026-11-03 after the test was scored; not part of the rule.

    python phonon_check.py WORKDIR     (WSL, hermes-mace env, GPU)"""
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.dft.kpoints import monkhorst_pack
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import BFGS
from ase.phonons import Phonons
from mace.calculators import mace_mp

STRUCT = Path("/mnt/c/Users/busta/Code/hermes-private/validation/oxide_vibration/structures")
WORK = Path(sys.argv[1])
NA_CM3 = 1e-24 * 6.02214076e23
R = 8.314462618
T = 298.15
CASES = {  # tag: (atoms per formula unit, supercell, V md cm3/mol, V measured cm3/mol)
    "calcite": (5, (3, 3, 1), 38.37, 36.93),
    "aragonite": (5, (3, 2, 3), 36.42, 34.15),
    "cao": (2, (3, 3, 3), None, 16.76),
}
calc = mace_mp(model=str(Path.home() / ".cache/mace/macempa0mediummodel"), device="cuda", default_dtype="float64")


def molar_volume(atoms, apf):
    return atoms.get_volume() / len(atoms) * apf * NA_CM3


def entropy(atoms, tag, label, supercell):
    ph = Phonons(atoms, calc, supercell=supercell, delta=0.01, name=str(WORK / f"{tag}_{label}"))
    ph.run()
    ph.read(acoustic=True)
    q = monkhorst_pack((8, 8, 8))  # even grid, no Gamma point, so no zero acoustic modes
    e = ph.band_structure(q).ravel()  # eV
    imaginary = int(np.sum(e < 0))
    x = np.abs(e) / (units.kB * T)
    s = x / np.expm1(x) - np.log(-np.expm1(-x))
    return R * s.sum() / len(q) / len(atoms), imaginary


for tag, (apf, supercell, v_md, v_meas) in CASES.items():
    atoms = read(STRUCT / f"{tag}.cif")
    atoms.calc = calc
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-3, steps=500)
    ref = atoms.copy()
    for label, v in (("relaxed", None), ("md", v_md), ("measured", v_meas)):
        if v is None and label != "relaxed":
            continue
        a = ref.copy()
        if v is not None:
            a.set_cell(a.cell * (v / molar_volume(ref, apf)) ** (1 / 3), scale_atoms=True)
        a.calc = calc
        BFGS(a, logfile=None).run(fmax=1e-4, steps=500)
        s, imag = entropy(a, tag, label, supercell)
        print(f"{tag:10s} {label:9s} V {molar_volume(a, apf):6.2f} cm3/mol  S_harm(298) {s:6.2f} J/(K mol atoms)"
              f"  imaginary {imag}", flush=True)
