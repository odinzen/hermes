"""Write levels.json, the crystal-field levels Arm B of RULE.md scores with, and check each multiplet is complete.

One source per ion, chosen 2026-10-24 before any LnPO4 thermal data were read (Michael's approval of the defaults):
the paper's observed level where it gives one, its own crystal-field fit where it does not; Yb as the neutron peaks
measure it (phonon-renormalized). Ce, Pr and Sm have no measured level set in a phosphate and are left out. Values
extracted from rendered table images by an agent that reported no thermal content; Ho and Er spot-checked by hand.
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from hermes.crystal_field import FOUR_F_ELECTRONS, hund_ground_j  # noqa: E402

OBS, FIT = "observed", "fitted"

LEVELS = {
    "Nd": {
        "source": "Antic-Fidancev, Holsa, Lemaitre-Blaise and Porcher, J. Phys.: Condens. Matter 3, 6829 (1991), "
                  "doi 10.1088/0953-8984/3/35/012",
        "locator": "Table 2, p. 6836 (pdf p. 9), NdPO4 experimental column", "method": "optical absorption",
        "unit": "cm-1", "note": "All five 4I9/2 doublets observed in pure NdPO4. Added 2026-10-24, still before any "
                                "LnPO4 thermal data were read.",
        "multiplets": {"4I9/2": [["", 0, 2, OBS], ["", 64, 2, OBS], ["", 210, 2, OBS], ["", 314, 2, OBS],
                                 ["", 456, 2, OBS]]},
    },
    "Eu": {
        "source": "Dexpert-Ghys, Mauricot and Faucher, J. Lumin. 69, 203 (1996), doi 10.1016/S0022-2313(96)00094-4",
        "locator": "Table 2, p. 210 (pdf p. 8), EuPO4 column", "method": "luminescence, 77 K", "unit": "cm-1",
        "note": "Site C1, all singlets. 7F1 is the excited multiplet the rule requires; 7F2 is included as measured. "
                "Which lines are electronic was decided with the authors' crystal-field analysis (p. 211).",
        "multiplets": {
            "7F0": [["", 0, 1, OBS]],
            "7F1": [["I", 257, 1, OBS], ["II", 418, 1, OBS], ["III", 457, 1, OBS]],
            "7F2": [["I", 939, 1, OBS], ["II", 970, 1, OBS], ["III", 1025, 1, OBS], ["IV", 1099, 1, OBS],
                    ["V", 1176, 1, OBS]],
        },
    },
    "Gd": {
        "source": "none needed: 8S7/2 splits by under 1 cm-1, one eight-fold level (the rule's built-in check)",
        "locator": "-", "method": "-", "unit": "cm-1", "note": "",
        "multiplets": {"8S7/2": [["", 0, 8, OBS]]},
    },
    "Tb": {
        "source": "Loong, Soderholm, Goodman, Abraham and Boatner, Phys. Rev. B 48, 6124 (1993), "
                  "doi 10.1103/PhysRevB.48.6124",
        "locator": "Table II, p. 6129 (pdf p. 6)", "method": "inelastic neutron scattering, 4.2 K and 15 K",
        "unit": "cm-1", "note": "Three levels not observed, taken from the paper's fit.",
        "multiplets": {"7F6": [
            ["G5", 0.0, 2, OBS], ["G3", 3.71, 1, OBS], ["G4", 7.82, 1, OBS], ["G1", 21.0, 1, OBS],
            ["G2", 74.2, 1, OBS], ["G5", 76.6, 2, FIT], ["G3", 79.9, 1, FIT], ["G1", 237.1, 1, OBS],
            ["G5", 251.6, 2, OBS], ["G4", 260.7, 1, FIT]]},
    },
    "Dy": {
        "source": "Loong, Soderholm, Xue, Abraham and Boatner, J. Alloys Compd. 207-208, 165 (1994), "
                  "doi 10.1016/0925-8388(94)90198-8",
        "locator": "Fig. 2, p. 167 (pdf p. 3); observed peaks in the text, p. 166", "method":
            "inelastic neutron scattering, 15 K", "unit": "cm-1",
        "note": "The figure is the fitted scheme; its lower five levels equal the observed peaks (8.5 meV inferred "
                "from an asymmetric peak, 16 meV weak). 37.5 and 38.9 meV not observed.",
        "multiplets": {"6H15/2": [
            ["G6", 0, 2, OBS], ["G6", 68.6, 2, OBS], ["G7", 76.6, 2, OBS], ["G6", 129, 2, OBS], ["G7", 147, 2, OBS],
            ["G7", 180, 2, OBS], ["G7", 302, 2, FIT], ["G6", 313, 2, FIT]]},
    },
    "Ho": {
        "source": "Loong, Soderholm, Hammonds, Abraham, Boatner and Edelstein, J. Phys.: Condens. Matter 5, 5121 "
                  "(1993), doi 10.1088/0953-8984/5/29/009",
        "locator": "Table 2, p. 5134 (pdf p. 15), HoPO4 observed column c; fit column b (neutron plus optical)",
        "method": "inelastic neutron scattering, 15 K", "unit": "cm-1",
        "note": "G2 and G1 near 184 cm-1 are one unresolved peak, both given 184.3. Five levels from fit b.",
        "multiplets": {"5I8": [
            ["G5", 0.0, 2, OBS], ["G4", 66.1, 1, OBS], ["G1", 73.5, 1, FIT], ["G3", 80.7, 1, OBS], ["G5", 94.0, 2, OBS],
            ["G4", 162.1, 1, OBS], ["G2", 184.3, 1, OBS], ["G1", 184.3, 1, OBS], ["G5", 222.7, 2, FIT],
            ["G3", 244.4, 1, OBS], ["G2", 273.8, 1, FIT], ["G5", 270.7, 2, FIT], ["G1", 294.2, 1, FIT]]},
    },
    "Er": {
        "source": "Loong, Soderholm, Hammonds, Abraham, Boatner and Edelstein, J. Phys.: Condens. Matter 5, 5121 "
                  "(1993), doi 10.1088/0953-8984/5/29/009",
        "locator": "Table 4, p. 5135 (pdf p. 16), neutron column; calculated column where not observed",
        "method": "inelastic neutron scattering, 15-30 K", "unit": "cm-1",
        "note": "Raman (Becker 1985, 1986) puts the third G7 at 105 and the top G6 at 269; the authors keep the "
                "neutron assignment, followed here. Top two levels from the fit.",
        "multiplets": {"4I15/2": [
            ["G7", 0, 2, OBS], ["G6", 32.9, 2, OBS], ["G7", 52.2, 2, OBS], ["G7", 138.2, 2, OBS],
            ["G6", 144.3, 2, OBS], ["G6", 233.9, 2, OBS], ["G7", 242.3, 2, FIT], ["G6", 322.6, 2, FIT]]},
    },
    "Tm": {
        "source": "Loong, Soderholm, Abraham, Boatner and Edelstein, J. Chem. Phys. 98, 4214 (1993), "
                  "doi 10.1063/1.465028",
        "locator": "Table II, p. 4219 (pdf p. 7), observed column; fit b where not observed",
        "method": "inelastic neutron scattering, 15 K and 100 K", "unit": "cm-1",
        "note": "254.9 and the pair at 338.0 come from weak, unresolved 100 K features. G4 277.9 from the fit.",
        "multiplets": {"3H6": [
            ["G1", 0.0, 1, OBS], ["G5", 29.0, 2, OBS], ["G3", 78.2, 1, OBS], ["G5", 136.3, 2, OBS],
            ["G2", 184.7, 1, OBS], ["G1", 254.9, 1, OBS], ["G4", 277.90, 1, FIT], ["G5", 280.70, 2, OBS],
            ["G3", 338.0, 1, OBS], ["G4", 338.0, 1, OBS]]},
    },
    "Yb": {
        "source": "Loong, Loewenhaupt, Nipko, Braden and Boatner, Phys. Rev. B 60, R12549 (1999), "
                  "doi 10.1103/PhysRevB.60.R12549",
        "locator": "text, p. R12550 (pdf p. 2)", "method": "single-crystal inelastic neutron scattering, 20 K",
        "unit": "meV",
        "note": "Peaks broadened and shifted by coupling to phonons; taken as measured (Michael's choice). The text "
                "gives the top level as ~43 meV on p. R12550 and about 45 meV on p. R12551; 43 used.",
        "multiplets": {"2F7/2": [["G6", 0, 2, OBS], ["G7", 12, 2, OBS], ["G6", 32, 2, OBS], ["G7", 43, 2, OBS]]},
    },
}
LEFT_OUT = {
    "Ce": "no measured 2F5/2 levels in CePO4 or Ce:LaPO4 found",
    "Pr": "only a fit to susceptibility and Raman data (Mandal 2018), not a measured level set",
    "Sm": "no measured 6H5/2 or 6H7/2 levels in SmPO4 or Sm:LaPO4 found",
}


def check():
    for ion, entry in LEVELS.items():
        multiplets = entry["multiplets"]
        ground_name, ground = next(iter(multiplets.items()))
        j = hund_ground_j(FOUR_F_ELECTRONS[ion])
        states = sum(level[2] for level in ground)
        if states != 2 * j + 1:
            raise SystemExit(f"{ion} {ground_name}: degeneracies sum to {states}, 2J+1 = {2 * j + 1}")
        if ion == "Eu":
            for name, want in (("7F1", 3), ("7F2", 5)):
                if sum(level[2] for level in multiplets[name]) != want:
                    raise SystemExit(f"Eu {name} incomplete")
        print(f"{ion}: {ground_name} J = {Fraction(j)}, {len(ground)} levels, {states} states, "
              f"{sum(level[3] == FIT for m in multiplets.values() for level in m)} from the fit")


def main():
    check()
    (HERE / "levels.json").write_text(json.dumps({"levels": LEVELS, "left_out": LEFT_OUT}, indent=1,
                                                 ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
