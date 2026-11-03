"""Materials Project structures for the held-out oxides of the second oxide test, by the rule of the first: the entry
in the measured polymorph's space group with the lowest energy above the hull, written as its conventional standard
cell. Writes ../oxide_vibration/structures/<tag>.cif (CaO is already there) and structures.json here.
"""
import json
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
STRUCTURES = HERE.parent / "oxide_vibration" / "structures"
KEY = (Path.home() / "Code/juno/entropy/_work/MP-API-key.txt").read_text().strip()

CASES = {  # tag: (formula, space group, polymorph)
    "beo": ("BeO", "P6_3mc", "bromellite (wurtzite)"),
    "cao": ("CaO", "Fm-3m", "lime"),
    "sro": ("SrO", "Fm-3m", "rock salt"),
    "bao": ("BaO", "Fm-3m", "rock salt"),
    "li2o": ("Li2O", "Fm-3m", "antifluorite"),
    "zro2": ("ZrO2", "P2_1/c", "baddeleyite"),
    "mg2sio4": ("Mg2SiO4", "Pnma", "forsterite"),
    "zrsio4": ("ZrSiO4", "I4_1/amd", "zircon"),
    "andalusite": ("Al2SiO5", "Pnnm", "andalusite"),
    "kyanite": ("Al2SiO5", "P-1", "kyanite"),
    "sillimanite": ("Al2SiO5", "Pnma", "sillimanite (Pbnm setting)"),
}


def main():
    out = {}
    with MPRester(KEY) as mpr:
        for tag, (formula, sg, polymorph) in CASES.items():
            docs = mpr.materials.summary.search(formula=formula, fields=["material_id", "energy_above_hull", "symmetry",
                                                                         "structure"])
            matching = sorted((d for d in docs if d.symmetry.symbol == sg), key=lambda d: d.energy_above_hull)
            print(f"{tag:12s} {sg:9s} {len(docs)} entries, {len(matching)} in {sg}: "
                  + ", ".join(f"{d.material_id} {d.energy_above_hull * 1000:.1f}" for d in matching[:3]))
            if not matching:
                raise SystemExit(f"no {sg} entry for {formula}")
            best = matching[0]
            conv = SpacegroupAnalyzer(best.structure, symprec=0.01).get_conventional_standard_structure()
            if tag != "cao":
                conv.to(filename=str(STRUCTURES / f"{tag}.cif"))
            out[tag] = {"formula": formula, "polymorph": polymorph, "space_group": sg,
                        "material_id": str(best.material_id), "energy_above_hull_eV_atom": best.energy_above_hull,
                        "conventional_atoms": len(conv), "abc": list(conv.lattice.abc),
                        "angles": list(conv.lattice.angles)}
    (HERE / "structures.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
