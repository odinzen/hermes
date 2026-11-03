"""Exploration, outside every verdict: Hermes's gaussian_entropy on exact classical harmonic samples of MgO (MACE-MP-0
force constants at the 298.15 K MD cell) in boxes of 2^3 to 6^3 conventional cells, against the exact entropy of the
box's own modes and the exact bulk value (dense mesh). Shows whether the centre-of-mass term is a finite-box effect.
"""
import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

import phonons as ph
from synthetic_gaussian import KT_EV, gaussian_entropy


def phonopy_for(md, n, calc):
    cell = PhonopyAtoms(symbols=md.get_chemical_symbols(), cell=md.cell.array, scaled_positions=md.get_scaled_positions())
    phon = Phonopy(cell, supercell_matrix=np.diag([n] * 3), primitive_matrix="auto")
    phon.generate_displacements(distance=0.01)
    forces = []
    for sc in phon.supercells_with_displacements:
        a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    phon.forces = forces
    phon.produce_force_constants()
    return phon


def main():
    calc = ph.calculator()
    rng = np.random.default_rng(2)
    unit, _ = ph.md_unit_cell("mgo")
    md = ph.relax_positions(unit, calc)
    bulk = None
    for n in (3, 2, 4, 5, 6):  # 3 first: its force constants give the bulk value
        phon = phonopy_for(md, n, calc)
        if bulk is None and n == 3:
            phon.run_mesh((24, 24, 24), is_gamma_center=False)
            nu = phon.get_mesh_dict()["frequencies"].ravel() * 1e12
            w = np.repeat(phon.get_mesh_dict()["weights"], 6)
            x = 2 * np.pi * ph.HBAR * nu / (ph.KB * ph.T0)
            bulk = ph.R * np.sum(w * (1 - np.log(x))) / w.sum() * 3  # 3 modes per atom
        box = phon.supercell
        N = len(box.masses)
        fc = phon.force_constants.transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
        fc = 0.5 * (fc + fc.T)
        m = np.repeat(box.masses, 3)
        w2, vec = np.linalg.eigh(fc / np.sqrt(np.outer(m, m)))
        keep = np.argsort(w2)[3:]
        w2, vec = w2[keep], vec[:, keep]
        nu = np.sqrt(w2 * 1.602176634e-19 / 1.66053906660e-27 * 1e20) / (2 * np.pi)
        x = 2 * np.pi * ph.HBAR * nu / (ph.KB * ph.T0)
        exact_box = ph.R * np.sum(1 - np.log(x)) / N
        q = rng.standard_normal((401, len(w2))) * np.sqrt(KT_EV / w2)
        pos = box.positions[None] + ((q @ vec.T) / np.sqrt(m)).reshape(401, N, 3)
        g = gaussian_entropy(pos.astype(np.float32), box.cell, box.masses, ph.T0, species=box.numbers, limit="box")
        gb = gaussian_entropy(pos.astype(np.float32), box.cell, box.masses, ph.T0, species=box.numbers, limit="bulk")
        from hermes.compression_entropy import crystal_correlation_entropy
        c = crystal_correlation_entropy(pos.astype(np.float32), box.cell, species=box.numbers, limit="box")
        print(f"N={N:5d}  Hermes {g['entropy']:.3f}  exact box {exact_box:.3f}  bulk {bulk:.3f}  "
              f"Hermes-bulk {g['entropy'] - bulk:+.3f}  box-bulk {exact_box - bulk:+.3f}  Hermes CoM term "
              f"{c['centre_of_mass']:.3f}  bulk-mode {gb['entropy']:.3f} bulk-mode error {gb['entropy'] - bulk:+.3f}",
              flush=True)


if __name__ == "__main__":
    main()
