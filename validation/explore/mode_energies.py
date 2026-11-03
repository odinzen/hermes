"""Exploration, outside every verdict: why the Gaussian entropy of a real 298.15 K trajectory falls below the classical
harmonic entropy of the same potential at the same cell. Projects the MD displacements on the harmonic normal modes of
the box and compares each mode's variance with kT / w^2.

If the modes share energy only slowly (the Bussi thermostat rescales all velocities by one factor, so only the
anharmonic coupling moves energy between modes), each mode keeps an energy drawn once from the thermal distribution: the
ratios r = var w^2 / kT then average 1 but scatter, a mode low in one half of the run stays low in the other, and
mean ln r falls below 0 (by Euler's constant 0.577 in the frozen limit). A potential that stiffens with amplitude instead
gives every mode r < 1 with little scatter. (R/2) mean ln r over the 3N-3 modes, per atom, is the classical gap this
explains. Runs in the WSL MACE venv:  python mode_energies.py MODEL RUNS TAG [TAG ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

import phonons as ph

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sol_bench"))
import phonons_bench as pb  # noqa: E402

AMU_A2_PER_EV_S2 = 1.602176634e-19 / 1.66053906660e-27 * 1e20  # (eV/A^2)/amu -> s^-2
KT_EV = ph.KB * ph.T0 / 1.602176634e-19


def box_modes(md, n, calc):
    cell = PhonopyAtoms(symbols=md.get_chemical_symbols(), cell=md.cell.array, scaled_positions=md.get_scaled_positions())
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
    big = len(box.masses)
    fc = phon.force_constants.transpose(0, 2, 1, 3).reshape(3 * big, 3 * big)
    m = np.repeat(box.masses, 3)
    w2, vec = np.linalg.eigh(0.5 * (fc + fc.T) / np.sqrt(np.outer(m, m)))
    keep = np.argsort(w2)[3:]
    return box, w2[keep], vec[:, keep], m


def match(box, frame0, cell):
    """Index of the MD atom on each phonopy supercell site, by minimum-image distance."""
    frac_md = np.linalg.solve(cell.T, frame0.T).T
    frac_box = np.linalg.solve(cell.T, box.positions.T).T
    d = frac_box[:, None] - frac_md[None]
    d -= np.round(d)
    order = np.argmin(np.linalg.norm(d @ cell, axis=2), axis=1)
    if len(set(order)) != len(order):
        raise ValueError("supercell sites do not map one to one onto the MD atoms")
    return order


def main():
    model, runs, tags = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    from mace.calculators import mace_mp
    weights, _ = pb.WEIGHTS[model]
    calc = mace_mp(model=str(weights), device="cuda", default_dtype="float64", enable_cueq=False)
    out_file = HERE / "mode_energies.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for tag in tags:
        d = np.load(runs / tag / "nvt_t0.npz")
        cell = d["cell"]
        unit, n = pb.md_unit_cell(tag, runs)
        unit.set_cell(cell / n[:, None], scale_atoms=True)  # the NVT cell itself
        md = ph.relax_positions(unit, calc)
        box, w2, vec, m = box_modes(md, n, calc)
        pos = d["positions"].astype(float)
        frac = np.linalg.solve(cell.T, pos.reshape(-1, 3).T).T.reshape(pos.shape)
        step = frac - frac[0]
        step -= np.round(step)
        cart = (frac[0] + step) @ cell
        disp = cart - cart.mean(axis=0)
        disp = disp[:, match(box, cart.mean(axis=0), cell)]
        q = disp.reshape(len(disp), -1) * np.sqrt(m) @ vec  # mass-weighted normal-mode coordinates
        r = q.var(axis=0) * w2 / KT_EV
        halves = [q[i * len(q) // 2:(i + 1) * len(q) // 2].var(axis=0) * w2 / KT_EV for i in (0, 1)]
        big = len(box.masses)
        res = {"modes": int(len(r)), "mean_r": float(r.mean()), "mean_ln_r": float(np.log(r).mean()),
               "sd_ln_r": float(np.log(r).std()),
               "half_correlation_ln_r": float(np.corrcoef(np.log(halves[0]), np.log(halves[1]))[0, 1]),
               "gap_explained": float(ph.R / 2 * np.log(r).sum() / big)}
        results[f"{model} {tag}"] = res
        print(f"{model} {tag}: " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                             for k, v in res.items()), flush=True)
        out_file.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
