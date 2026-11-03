"""Materials Project structures for the oxide vibration test, as RULE.md fixes them.

For each oxide, the entry with the stated space group and the lowest energy above the hull, written as its
conventional standard cell. Writes structures/<tag>.cif and structures.json.
"""
import json
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
KEY = (Path.home() / "Code/juno/entropy/_work/MP-API-key.txt").read_text().strip()

CASES = {  # tag: (formula, space group, polymorph)
    "mgo": ("MgO", "Fm-3m", "periclase"),
    "al2o3": ("Al2O3", "R-3c", "corundum"),
    "tio2": ("TiO2", "P4_2/mnm", "rutile"),
    "y2o3": ("Y2O3", "Ia-3", "bixbyite"),
    "sc2o3": ("Sc2O3", "Ia-3", "bixbyite"),
    "lu2o3": ("Lu2O3", "Ia-3", "bixbyite"),
    "la2o3": ("La2O3", "P-3m1", "A-type"),
    "srtio3": ("SrTiO3", "Pm-3m", "cubic perovskite"),
    "mgtio3": ("MgTiO3", "R-3", "geikielite"),
    "mg3po42": ("Mg3(PO4)2", "P2_1/c", "farringtonite"),  # MP and spglib report P2_1/n as its standard P2_1/c
    "cao": ("CaO", "Fm-3m", "lime (smoke test only)"),
}


def main():
    out = {}
    (HERE / "structures").mkdir(exist_ok=True)
    with MPRester(KEY) as mpr:
        for tag, (formula, sg, polymorph) in CASES.items():
            docs = mpr.materials.summary.search(
                formula=formula.replace("(PO4)2", "P2O8"),
                fields=["material_id", "energy_above_hull", "symmetry", "structure", "database_IDs"])
            matching = sorted((d for d in docs if d.symmetry.symbol == sg), key=lambda d: d.energy_above_hull)
            print(f"{formula:10s} {sg:9s} {len(docs)} entries, {len(matching)} in {sg}: "
                  + ", ".join(f"{d.material_id} {d.energy_above_hull * 1000:.1f}" for d in matching[:3]))
            if not matching:
                raise SystemExit(f"no {sg} entry for {formula}")
            best = matching[0]
            conv = SpacegroupAnalyzer(best.structure, symprec=0.01).get_conventional_standard_structure()
            conv.to(filename=str(HERE / "structures" / f"{tag}.cif"))
            out[tag] = {"formula": formula, "polymorph": polymorph, "space_group": sg,
                        "material_id": str(best.material_id), "energy_above_hull_eV_atom": best.energy_above_hull,
                        "conventional_atoms": len(conv), "abc": list(conv.lattice.abc),
                        "angles": list(conv.lattice.angles)}
    (HERE / "structures.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
