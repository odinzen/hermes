"""Measured and simulated molar volumes of the development oxides at 298.15 K, cm3 per mole of formula units.

The measured volume is matched on phase as well as composition: Juno holds both rutile and anatase for TiO2.
"""
import json
from pathlib import Path

import numpy as np

# Juno sits beside hermes-private under Code/, which resolves from both Windows and WSL.
MANIFEST = Path(__file__).resolve().parents[3] / "juno/assessments/Refractory-Oxides/manifest.json"
NA_CM3 = 1e-24 * 6.02214076e23

OXIDES = {  # tag: (composition, atoms per formula, word in the Robie and Hemingway phase label)
    "mgo": ("MgO", 2, "periclase"),
    "al2o3": ("Al2O3", 5, "corundum"),
    "tio2": ("TiO2", 3, "rutile"),
    "nio": ("NiO", 2, "bunsenite"),
    "mno": ("MnO", 2, "manganosite"),
    "coo": ("CoO", 2, "cobalt monoxide"),
    "fe2o3": ("Fe2O3", 5, "hematite"),
    "cr2o3": ("Cr2O3", 5, "eskolaite"),
}


def measured_volume(tag, manifest=None):
    composition, _, phase = OXIDES[tag]
    manifest = manifest or json.loads(MANIFEST.read_text(encoding="utf-8"))
    hits = [a["value"] for a in manifest["artifacts"] if a["quantity"] == "molar_volume"
            and a["paper"] == "paper:robie_hemingway_1995"
            and (a.get("conditions") or {}).get("composition") == composition and phase in str(a.get("phase")).lower()]
    if len(hits) != 1:
        raise ValueError(f"{tag}: {len(hits)} measured volumes match {composition} {phase}")
    return hits[0]


def md_volume(run, atoms_per_formula):
    """Volume of the 298.15 K NPT average cell of a returned run, per mole of formula units."""
    ladder = json.loads((Path(run) / "ladder.json").read_text())
    if abs(ladder[0]["T"] - 298.15) > 1e-6:
        raise ValueError(f"{run}: first ladder step is {ladder[0]['T']} K")
    n_atoms = len(np.load(Path(run) / "nvt_t0.npz")["masses"])
    return abs(np.linalg.det(np.array(ladder[0]["mean_cell"]))) / n_atoms * atoms_per_formula * NA_CM3
