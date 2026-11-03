"""Held-out test of the solid entropy with the neighbour correction. Rule written 2026-10-15 BEFORE any of these runs existed.

The neighbour correction (hermes.compression_entropy.neighbour_correlation_entropy) was developed on one case,
fcc Cu at 1300 K (Mendelev EAM), where compression + offset - correction gave 72.5 against 72.76. This tests it on
twelve unseen cases, three lattices, six metals, two temperatures each (EAM potentials, about 1000 atoms):
  bcc Fe (Mendelev) 700, 1400 K; bcc W (Zhou) 1200, 2400 K; hcp Mg (Mendelev) 370, 740 K;
  hcp Zr (Mendelev) 450, 900 K; fcc Al (Mendelev) 370, 740 K; fcc Ni (Foiles u3) 700, 1400 K.
Each case is run by run_solids.sh: box relaxed anisotropically at P = 0 and T, then 60 ps NVT at that box, frames every
0.5 ps (121 frames), mean potential energy U; calphy Frenkel-Ladd free energy G from the same relaxed structure.
Exact:   S = (U + 3/2 k T - G) / T, J/(K mol of atoms), as for Cu.
Hermes:  S = trajectory_entropy(frames, n_pairs=60, min_separation=10)["mean"] + CLASSICAL_ENTROPY_OFFSET
             - neighbour_correlation_entropy(frames)["entropy"].
Pass:    |S_Hermes - S_exact| <= 1.0 J/(K mol) in every valid case.
Valid:   the crystal stays on its sites (neighbour_correlation_entropy accepts it) and calphy dissipation <= 0.01 eV/atom
         (a case above that is rerun once with 100000 switching steps before this script is run). Excluded cases are
         reported; with more than two excluded the test is void.
The uncorrected Hermes value is reported alongside for comparison and plays no part in the verdict.
Nothing in this rule may be changed after the solid runs are read.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import yaml

HERMES = r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad\frozen_v1"
sys.path.insert(0, HERMES)
from hermes.compression_entropy import neighbour_correlation_entropy, trajectory_entropy  # noqa: E402
from hermes.md_configuration import CLASSICAL_ENTROPY_OFFSET  # noqa: E402

FROZEN = {
    "hermes/compression_entropy.py": "288fcd6b5d6fa47887499e92e67bf260c54e3525717e018c5520c5a0935da256",
    "hermes/md_configuration.py": "0e10322a636875e3cb8aec19aaeb03d48b66a941000a440bb94a5c6b37be0eb2",
}
HERE = os.path.dirname(os.path.abspath(__file__))
EV = 96485.33212
KB = 8.617333262e-5
TOLERANCE = 1.0
MAX_DISSIPATION = 0.01
CASES = {  # tag: (lattice, mass amu, T)
    "fe_0700": ("bcc", 55.845, 700.0), "fe_1400": ("bcc", 55.845, 1400.0),
    "w_1200": ("bcc", 183.84, 1200.0), "w_2400": ("bcc", 183.84, 2400.0),
    "mg_0370": ("hcp", 24.305, 370.0), "mg_0740": ("hcp", 24.305, 740.0),
    "zr_0450": ("hcp", 91.224, 450.0), "zr_0900": ("hcp", 91.224, 900.0),
    "al_0370": ("fcc", 26.98154, 370.0), "al_0740": ("fcc", 26.98154, 740.0),
    "ni_0700": ("fcc", 58.71, 700.0), "ni_1400": ("fcc", 58.71, 1400.0),
}


def check_frozen():
    for rel, want in FROZEN.items():
        # line endings normalized, so a Windows checkout of the same commit passes
        got = hashlib.sha256(open(os.path.join(HERMES, rel), "rb").read().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            sys.exit(f"{rel} changed since the rule was written ({got}); refusing to run")


def read_dump(path):
    frames, box = [], None
    lines = open(path).read().split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            n = int(lines[i + 1])
            block = lines[i + 7:i + 7 + n]
            if len(block) < n or any(len(b.split()) != 5 for b in block):
                break
            lo_hi = [list(map(float, lines[i + 3 + k].split()[:2])) for k in range(3)]
            box = np.array([h - lo for lo, h in lo_hi])
            data = np.array([b.split() for b in block], float)
            frames.append(data[np.argsort(data[:, 0])][:, 2:5])
            i += 7 + n
        else:
            i += 1
    return np.array(frames), np.diag(box)


def exact_entropy(tag, temperature):
    rows = [ln.split() for ln in open(os.path.join(HERE, f"{tag}_thermo.dat")) if not ln.startswith("#")]
    u = float(rows[-1][1])
    (folder,) = [d for d in glob.glob(os.path.join(HERE, f"{tag}-fe-*")) if os.path.isdir(d)]
    report = yaml.safe_load(open(os.path.join(folder, "report.yaml")))["results"]
    return (u + 1.5 * KB * temperature - report["free_energy"]) / temperature * EV, report["dissipation"]


def main():
    check_frozen()
    out, errors, excluded = {}, [], []
    for tag, (lattice, mass, temperature) in CASES.items():
        exact, dissipation = exact_entropy(tag, temperature)
        frames, cell = read_dump(os.path.join(HERE, f"{tag}.dump"))
        try:
            corr = neighbour_correlation_entropy(frames, cell)
        except ValueError as e:
            excluded.append(tag)
            out[tag] = {"excluded": str(e)}
            print(f"{tag}: EXCLUDED ({e})")
            continue
        if dissipation > MAX_DISSIPATION:
            excluded.append(tag)
            out[tag] = {"excluded": f"calphy dissipation {dissipation:.4f} eV/atom"}
            print(f"{tag}: EXCLUDED (dissipation {dissipation:.4f})")
            continue
        comp = trajectory_entropy(frames, cell, np.full(frames.shape[1], mass), temperature, n_pairs=60, min_separation=10)
        raw = comp["mean"] + CLASSICAL_ENTROPY_OFFSET
        hermes = raw - corr["entropy"]
        errors.append(hermes - exact)
        out[tag] = {"lattice": lattice, "T": temperature, "exact": exact, "hermes": hermes, "uncorrected": raw,
                    "correction": corr["entropy"], "pairs_per_atom": corr["pairs_per_atom"],
                    "compression_sem": float(comp["samples"].std(ddof=1) / np.sqrt(len(comp["samples"]))),
                    "dissipation": dissipation, "pass": abs(hermes - exact) <= TOLERANCE}
        print(f"{tag:8s} {lattice} {temperature:6.0f} K  exact {exact:7.2f}  Hermes {hermes:7.2f} ({hermes - exact:+5.2f})  "
              f"uncorrected {raw:7.2f} ({raw - exact:+5.2f})  correction {corr['entropy']:5.2f}  "
              f"{'PASS' if out[tag]['pass'] else 'FAIL'}", flush=True)
    void = len(excluded) > 2
    verdict = (not void) and all(v.get("pass", True) for v in out.values())
    for lat in ("bcc", "hcp", "fcc"):
        e = [v["hermes"] - v["exact"] for v in out.values() if v.get("lattice") == lat]
        if e:
            print(f"{lat}: mean error {np.mean(e):+.2f}, max |error| {np.max(np.abs(e)):.2f}")
    print(f"excluded {excluded}; verdict {'VOID' if void else ('PASS' if verdict else 'FAIL')}")
    out["verdict"] = "VOID" if void else ("PASS" if verdict else "FAIL")
    json.dump(out, open(os.path.join(HERE, "solid_results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
