"""Materials Project inputs for B2 of RULE.md (the phosphates' full prediction, reported outside the verdict).

Selected as for Arm A: the entry in the space group of the measured polymorph with the lowest energy above the hull.
For GdPO4 that is monazite (mp-3735), not MP's lowest entry, which is xenotime. Writes the conventional cell to
../oxide_vibration/structures/<tag>.cif, the GGA+U density of states to dos/<tag>.npz, and phosphates.json.
"""
import json

import numpy as np
from mp_api.client import MPRester
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from fetch_magnetic import HERE, KEY, STRUCTURES, band_gap

CASES = {  # tag: (formula, space group, polymorph)
    "ndpo4": ("NdPO4", "P2_1/c", "monazite"),
    "eupo4": ("EuPO4", "P2_1/c", "monazite"),
    "gdpo4": ("GdPO4", "P2_1/c", "monazite"),
    "hopo4": ("HoPO4", "I4_1/amd", "xenotime"),
    "erpo4": ("ErPO4", "I4_1/amd", "xenotime"),
    "tmpo4": ("TmPO4", "I4_1/amd", "xenotime"),
    "ybpo4": ("YbPO4", "I4_1/amd", "xenotime"),
}


def main():
    out = {}
    with MPRester(KEY) as mpr:
        for tag, (formula, sg, polymorph) in CASES.items():
            docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "energy_above_hull",
                                                                         "symmetry", "structure", "band_gap"])
            best = min((d for d in docs if d.symmetry.symbol == sg), key=lambda d: d.energy_above_hull)
            conv = SpacegroupAnalyzer(best.structure, symprec=0.01).get_conventional_standard_structure()
            conv.to(filename=str(STRUCTURES / f"{tag}.cif"))
            dos = mpr.get_dos_by_material_id(best.material_id)
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
                        "summary_band_gap_eV": best.band_gap, "dos_gap_eV": band_gap(energies, up + down, dos.efermi),
                        "dos_atoms": dos_atoms, "dos_task": str(summary.task_id), "dos_calc_type": str(task.calc_type),
                        "dos_magnetic_ordering": str(summary.magnetic_ordering), "conventional_atoms": len(conv)}
            print(tag, out[tag])
    (HERE / "phosphates.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
