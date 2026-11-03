"""Fresh held-out test of the revised solid entropy. Rule written 2026-10-15 BEFORE any of these runs existed.

The revision answers the 2026-10-15 red-team findings on the first solid route: the compression is coded in groups of
256 atoms so its offset no longer depends on the box (calibrated 5.83 +- 0.04 on 60 Einstein solids), and the shared
entropy is the exact Gaussian total correlation from the crystal's translations (crystal_correlation_entropy) instead
of a sum over neighbour pairs. Developed on harmonic test crystals and on the Cu 1300 K case only (73.32 vs 72.76).
None of the cases below, nor the twelve cases of validation/solid_exact, was looked at while developing it.

Cases (EAM potentials), each run by run_fresh.sh: box relaxed anisotropically at P = 0 and T, then 200 ps NVT at that
box with frames every 0.5 ps (401 frames) and the mean potential energy U; calphy Frenkel-Ladd G from the relaxed box.
  fcc Ni (Foiles u3) 1000 K at 256, 864 and 2048 atoms        hcp Mg (Mendelev) 550 K at 800 and 2352 atoms
  bcc Fe (Mendelev) 1000 K, 1024    bcc V (Mendelev, VFe) 1200 K, 1024    hcp Zr (Mendelev) 700 K, 800
  fcc Al (Mendelev) 550 K, 864      fcc Pd (Foiles u3) 900 K, 864         fcc Ag (Foiles u3) 700 K, 864
  fcc Pt (Foiles u3) 1200 K, 864
Exact:   S = (U + 3/2 k T - G) / T, J/(K mol of atoms).
Hermes:  S = trajectory_entropy(frames, n_pairs=60, min_separation=10)["mean"] + CLASSICAL_ENTROPY_OFFSET
             - crystal_correlation_entropy(frames)["entropy"].
Pass, both required:
  A  |S_Hermes - S_exact| <= 1.0 J/(K mol) in every valid case;
  B  box-size independence: the error S_Hermes - S_exact spreads by no more than 0.5 J/(K mol) across the three Ni
     boxes, and likewise across the two Mg boxes.
Valid: the crystal stays on its sites (crystal_correlation_entropy accepts it) and calphy dissipation <= 0.01 eV/atom
(a case above that is rerun once with 100000 switching steps before this script is run). Excluded cases are reported;
with more than two excluded, or any case of a size series excluded, the test is void.
Compression plus offset without the correlation term is reported alongside and plays no part in the verdict.
Nothing in this rule may be changed after the fresh runs are read.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import yaml

HERMES = r"C:\Users\busta\Code\hermes-private"
sys.path.insert(0, HERMES)
from hermes.compression_entropy import crystal_correlation_entropy, trajectory_entropy  # noqa: E402
from hermes.md_configuration import CLASSICAL_ENTROPY_OFFSET  # noqa: E402

FROZEN = {
    "hermes/compression_entropy.py": "6857d825f8ff848a4d74a1f9da0872960e0e410befe3b4782804eebb82cda229",
    "hermes/md_configuration.py": "b7ee8dde8fb0ce6b732256b5ceda139403b806ee7fda0360e6646a5d1493e26e",
}
HERE = os.path.dirname(os.path.abspath(__file__))
EV = 96485.33212
KB = 8.617333262e-5
TOLERANCE, SPREAD, MAX_DISSIPATION = 1.0, 0.5, 0.01
CASES = {  # tag: (lattice, mass amu, T)
    "ni_1000_0256": ("fcc", 58.71, 1000.0), "ni_1000_0864": ("fcc", 58.71, 1000.0), "ni_1000_2048": ("fcc", 58.71, 1000.0),
    "mg_0550_0800": ("hcp", 24.305, 550.0), "mg_0550_2352": ("hcp", 24.305, 550.0),
    "fe_1000_1024": ("bcc", 55.845, 1000.0), "v_1200_1024": ("bcc", 50.9415, 1200.0), "zr_0700_0800": ("hcp", 91.224, 700.0),
    "al_0550_0864": ("fcc", 26.98154, 550.0), "pd_0900_0864": ("fcc", 106.40, 900.0), "ag_0700_0864": ("fcc", 107.87, 700.0),
    "pt_1200_0864": ("fcc", 195.09, 1200.0),
}
SERIES = {"Ni": ["ni_1000_0256", "ni_1000_0864", "ni_1000_2048"], "Mg": ["mg_0550_0800", "mg_0550_2352"]}


def check_frozen():
    for rel, want in FROZEN.items():
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
    out, excluded = {}, []
    for tag, (lattice, mass, temperature) in CASES.items():
        exact, dissipation = exact_entropy(tag, temperature)
        frames, cell = read_dump(os.path.join(HERE, f"{tag}.dump"))
        reason = None
        try:
            corr = crystal_correlation_entropy(frames, cell)
        except ValueError as e:
            reason = str(e)
        if reason is None and dissipation > MAX_DISSIPATION:
            reason = f"calphy dissipation {dissipation:.4f} eV/atom"
        if reason:
            excluded.append(tag)
            out[tag] = {"excluded": reason}
            print(f"{tag}: EXCLUDED ({reason})")
            continue
        comp = trajectory_entropy(frames, cell, np.full(frames.shape[1], mass), temperature, n_pairs=60, min_separation=10)
        raw = comp["mean"] + CLASSICAL_ENTROPY_OFFSET
        hermes = raw - corr["entropy"]
        out[tag] = {"lattice": lattice, "T": temperature, "atoms": frames.shape[1], "exact": exact, "hermes": hermes,
                    "uncorrected": raw, "correlation": corr["entropy"], "cells": corr["cells"], "basis": corr["basis"],
                    "compression_sem": float(comp["samples"].std(ddof=1) / np.sqrt(len(comp["samples"]))),
                    "dissipation": dissipation, "error": hermes - exact, "pass": abs(hermes - exact) <= TOLERANCE}
        print(f"{tag:13s} {lattice} {frames.shape[1]:5d} atoms  exact {exact:7.2f}  Hermes {hermes:7.2f} ({hermes - exact:+5.2f})  "
              f"uncorrected {raw:7.2f}  correlation {corr['entropy']:5.2f}  {'PASS' if out[tag]['pass'] else 'FAIL'}", flush=True)
    series_excluded = any(t in excluded for tags in SERIES.values() for t in tags)
    void = len(excluded) > 2 or series_excluded
    pass_a = all(v.get("pass", True) for v in out.values())
    pass_b = True
    for name, tags in SERIES.items():
        if all(t in out and "error" in out[t] for t in tags):
            errors = [out[t]["error"] for t in tags]
            spread = max(errors) - min(errors)
            pass_b &= spread <= SPREAD
            print(f"{name} size series errors {[round(e, 2) for e in errors]}: spread {spread:.2f} "
                  f"{'PASS' if spread <= SPREAD else 'FAIL'}")
    verdict = "VOID" if void else ("PASS" if pass_a and pass_b else "FAIL")
    errs = [v["error"] for v in out.values() if "error" in v]
    print(f"mean error {np.mean(errs):+.2f}, max |error| {np.max(np.abs(errs)):.2f}; excluded {excluded}")
    print(f"criterion A {'PASS' if pass_a else 'FAIL'}, criterion B {'PASS' if pass_b else 'FAIL'}, verdict {verdict}")
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(HERE, "fresh_results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
