"""Materials Project inputs for Arm A (magnetic oxides) of RULE.md.

The rule says "the Materials Project structure of each oxide" without naming an entry. Filled here as the oxide
vibration rule selects: the entry in the space group of the polymorph stable at 298.15 K with the lowest energy above
the hull. That is not always MP's lowest entry: MP's GGA+U puts MnO lowest in a distorted Pmmn cell and CoO in zinc
blende. Writes the conventional cell to ../oxide_vibration/structures/<tag>.cif (so the vibration route runs it
unchanged), the GGA+U density of states to dos/<tag>.npz, and magnetic.json.
"""
import json
from pathlib import Path

import numpy as np
from mp_api.client import MPRester
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
STRUCTURES = HERE.parent / "oxide_vibration" / "structures"
KEY = (Path.home() / "Code/juno/entropy/_work/MP-API-key.txt").read_text().strip()

CASES = {  # tag: (formula, space group, polymorph)
    "nio": ("NiO", "Fm-3m", "bunsenite (rock salt)"),
    "mno": ("MnO", "Fm-3m", "manganosite (rock salt)"),
    "coo": ("CoO", "Fm-3m", "rock salt"),
    "fe2o3": ("Fe2O3", "R-3c", "hematite (corundum)"),
    "cr2o3": ("Cr2O3", "R-3c", "eskolaite (corundum)"),
}


def band_gap(energies, total, efermi, floor=1e-3):
    """Gap around the Fermi level of a summed DOS, where it falls below a small floor."""
    occupied = energies[(energies <= efermi) & (total > floor)]
    empty = energies[(energies > efermi) & (total > floor)]
    return float(empty.min() - occupied.max()) if len(occupied) and len(empty) else 0.0


def main():
    (HERE / "dos").mkdir(exist_ok=True)
    out = {}
    with MPRester(KEY) as mpr:
        for tag, (formula, sg, polymorph) in CASES.items():
            docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "energy_above_hull",
                                                                         "symmetry", "structure", "band_gap"])
            matching = sorted((d for d in docs if d.symmetry.symbol == sg), key=lambda d: d.energy_above_hull)
            best = matching[0]
            conv = SpacegroupAnalyzer(best.structure, symprec=0.01).get_conventional_standard_structure()
            conv.to(filename=str(STRUCTURES / f"{tag}.cif"))
            dos = mpr.get_dos_by_material_id(best.material_id)
            # the DOS comes from its own calculation, whose cell (and magnetic order) need not be the summary's
            summary = mpr.materials.electronic_structure.search(material_ids=[str(best.material_id)])[0].dos
            task = mpr.materials.tasks.search(task_ids=[str(summary.task_id)], fields=["output", "calc_type"])[0]
            dos_atoms = len(task.output.structure)
            energies = np.asarray(dos.energies, dtype=float)
            up = np.asarray(dos.densities[Spin.up], dtype=float)
            down = np.asarray(dos.densities.get(Spin.down, np.zeros_like(up)), dtype=float)
            np.savez(HERE / "dos" / f"{tag}.npz", energies=energies, up=up, down=down, efermi=dos.efermi,
                     atoms=dos_atoms)
            out[tag] = {"formula": formula, "polymorph": polymorph, "space_group": sg,
                        "material_id": str(best.material_id), "energy_above_hull_eV_atom": best.energy_above_hull,
                        "lowest_entry_any_space_group": str(min(docs, key=lambda d: d.energy_above_hull).material_id),
                        "summary_band_gap_eV": best.band_gap, "dos_gap_eV": band_gap(energies, up + down, dos.efermi),
                        "dos_atoms": dos_atoms, "dos_task": str(summary.task_id), "dos_calc_type": str(task.calc_type),
                        "dos_magnetic_ordering": str(summary.magnetic_ordering),
                        "dos_spin_gaps_eV": {str(k): v.band_gap for k, v in summary.total.items()},
                        "conventional_atoms": len(conv)}
            print(tag, out[tag])
    (HERE / "magnetic.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
