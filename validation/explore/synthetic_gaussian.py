"""Exploration, outside every verdict: feed Hermes's gaussian_entropy exact Gaussian samples of a harmonic crystal
(the MACE-MP-0 force constants of the MD box, classical, 298.15 K) and compare with the exact classical entropy of the
same box modes. Isolates the estimator from the potential and from anharmonicity.  python synthetic_gaussian.py TAG ...
"""
import json
import sys

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

import phonons as ph

sys.path.insert(0, str(ph.HERE.parents[1]))
from hermes.compression_entropy import gaussian_entropy  # noqa: E402

KT_EV = ph.KB * ph.T0 / 1.602176634e-19


def main():
    calc = ph.calculator()
    rng = np.random.default_rng(1)
    for tag in sys.argv[1:]:
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
        box = phon.supercell
        N = len(box.masses)
        fc = phon.force_constants.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)  # eV/A^2
        fc = 0.5 * (fc + fc.T)
        m = np.repeat(box.masses, 3)
        dyn = fc / np.sqrt(np.outer(m, m))
        w2, vec = np.linalg.eigh(dyn)
        keep = np.argsort(w2)[3:]  # drop the three translations
        w2, vec = w2[keep], vec[:, keep]
        # exact classical entropy of these modes, in the same units as box_modes.py
        nu = np.sqrt(w2 * 1.602176634e-19 / 1.66053906660e-27 * 1e20) / (2 * np.pi)
        x = 2 * np.pi * ph.HBAR * nu / (ph.KB * ph.T0)
        exact = ph.R * np.sum(1 - np.log(x)) / N
        # classical Gaussian samples: mass-weighted normal-mode amplitudes with variance kT / w^2
        positions0 = box.positions
        for frames in (401,):
            q = rng.standard_normal((frames, len(w2))) * np.sqrt(KT_EV / w2)
            u = (q @ vec.T) / np.sqrt(m)
            pos = positions0[None] + u.reshape(frames, N, 3)
            g = gaussian_entropy(pos.astype(np.float32), box.cell, box.masses, ph.T0, species=box.numbers, limit="box")
            gb = gaussian_entropy(pos.astype(np.float32), box.cell, box.masses, ph.T0, species=box.numbers,
                                  limit="bulk")
            bulk = json.loads((ph.HERE / f"phonons_{tag}.json").read_text())["at_md_cell"]["S_cl"]
            print(f"{tag:7s} {N} atoms {frames:5d} frames: box {g['entropy']:.3f} (exact box {exact:.3f}) | "
                  f"bulk {gb['entropy']:.3f} (exact bulk {bulk:.3f}, error {gb['entropy'] - bulk:+.3f})", flush=True)


if __name__ == "__main__":
    main()
