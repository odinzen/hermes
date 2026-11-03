"""Materials Project structures for the polymorph test, the entries named in RULE.md, written as the conventional
standard cell to ../oxide_vibration/structures/<tag>.cif, with structures.json here. Refuses an entry whose space group
or formula is not the one the rule names.
"""
import json
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).resolve().parent
STRUCTURES = HERE.parent / "oxide_vibration" / "structures"
KEY = (Path.home() / "Code/juno/entropy/_work/MP-API-key.txt").read_text().strip()

CASES = {  # tag: (formula, space group, mineral, Materials Project id)
    "quartz": ("SiO2", "P3_121", "alpha-quartz", "mp-7000"),
    "coesite": ("SiO2", "C2/c", "coesite", "mp-11684"),
    "stishovite": ("SiO2", "P4_2/mnm", "stishovite", "mp-6947"),
    "calcite": ("CaCO3", "R-3c", "calcite", "mp-3953"),
    "aragonite": ("CaCO3", "Pnma", "aragonite", "mp-4626"),
    "wollastonite": ("CaSiO3", "P-1", "wollastonite", "mp-4429"),
    "pseudowollastonite": ("CaSiO3", "C2/c", "pseudowollastonite", "mp-561086"),
    "enstatite": ("MgSiO3", "Pbca", "orthoenstatite", "mp-5834"),
    "clinoenstatite": ("MgSiO3", "P2_1/c", "low clinoenstatite", "mp-3470"),
    "perovskite": ("CaTiO3", "Pnma", "perovskite", "mp-4019"),
    "chrysoberyl": ("BeAl2O4", "Pnma", "chrysoberyl", "mp-3081"),
    "diopside": ("CaMgSi2O6", "C2/c", "diopside", "mp-6766"),
    "jadeite": ("NaAlSi2O6", "C2/c", "jadeite", "mp-6124"),
}


def main():
    out = {}
    with MPRester(KEY) as mpr:
        for tag, (formula, sg, mineral, mpid) in CASES.items():
            doc = mpr.materials.summary.search(material_ids=[mpid], fields=["material_id", "energy_above_hull",
                                                                            "symmetry", "structure"])[0]
            if doc.symmetry.symbol != sg or doc.structure.composition.reduced_formula != Composition(formula).reduced_formula:
                raise SystemExit(f"{tag}: {mpid} is {doc.structure.composition.reduced_formula} {doc.symmetry.symbol}")
            conv = SpacegroupAnalyzer(doc.structure, symprec=0.01).get_conventional_standard_structure()
            conv.to(filename=str(STRUCTURES / f"{tag}.cif"))
            out[tag] = {"formula": formula, "mineral": mineral, "space_group": sg, "material_id": mpid,
                        "energy_above_hull_eV_atom": doc.energy_above_hull, "conventional_atoms": len(conv),
                        "abc": list(conv.lattice.abc), "angles": list(conv.lattice.angles)}
            print(f"{tag:19s} {mpid:10s} {sg:9s} {len(conv)} atoms")
    (HERE / "structures.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
