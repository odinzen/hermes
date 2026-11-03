"""Exploration, outside every verdict: the classical harmonic entropy of exactly the vibrations an MD box holds (the
q-points commensurate with the box, three zero modes dropped), the like-for-like exact reference for Hermes's classical
Gaussian entropy at 298.15 K on the same potential and cell.   python box_modes.py TAG [TAG ...]
"""
import json
import sys

import numpy as np
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms

import phonons as ph


def main():
    calc = ph.calculator()
    ox = json.loads((ph.VIB / "oxide_results.json").read_text())["cases"]
    arm_a = json.loads((ph.HERE.parent / "extra_terms" / "arm_a_results.json").read_text())["cases"]
    for tag in sys.argv[1:]:
        p = (ox.get(tag) or arm_a.get(tag))["prediction"]
        unit, n = ph.md_unit_cell(tag)
        md = ph.relax_positions(unit, calc)
        cell = PhonopyAtoms(symbols=md.get_chemical_symbols(), cell=md.cell.array,
                            scaled_positions=md.get_scaled_positions())
        phon = Phonopy(cell, supercell_matrix=np.diag(n), primitive_matrix="auto")
        phon.generate_displacements(distance=0.01)
        forces = []
        for sc in phon.supercells_with_displacements:
            a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
            a.calc = calc
            forces.append(a.get_forces())
        phon.forces = forces
        phon.produce_force_constants()
        smat_prim = np.rint(np.linalg.inv(phon.primitive_matrix) @ np.diag(n)).astype(int)
        q = get_commensurate_points(smat_prim)
        phon.run_qpoints(q)
        nu = phon.get_qpoints_dict()["frequencies"].ravel() * 1e12
        nu = np.sort(nu)[3:]  # the box's three translations
        natoms = len(md) * int(np.prod(n))
        x = 2 * np.pi * ph.HBAR * nu / (ph.KB * ph.T0)
        s_cl = ph.R * np.sum(1 - np.log(x)) / natoms
        s_q = ph.R * np.sum(x / np.expm1(x) - np.log1p(-np.exp(-x))) / natoms
        print(f"{tag:7s} box {natoms} atoms, {len(q)} q-points, min {nu.min() / 1e12:.2f} THz | classical: Hermes "
              f"{p['S_cl_T0']:.2f} box-harmonic {s_cl:.2f} diff {p['S_cl_T0'] - s_cl:+.2f} | quantum box-harmonic "
              f"{s_q:.2f}", flush=True)


if __name__ == "__main__":
    main()
