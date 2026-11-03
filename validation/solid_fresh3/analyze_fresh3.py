"""Third fresh held-out test: solid entropy near melting by the heat-capacity route. Rule written 2026-10-16 BEFORE any
of these runs existed.

The second fresh test (validation/solid_fresh2) FAILED on accuracy near melting: Cu 1200 K +1.38, Ni 1500 K +1.29,
Pd 1300 K +1.15 J/(K mol), box-size independence having passed. Diagnosed on the frozen set (validation/anharmonic_
diagnosis): the error of a Gaussian description grows with temperature in all six metals tested, from about zero at a
third of the melting point; each atom's own motion and every neighbour pair are Gaussian to 0.05, so the missing part is
correlation among three or more atoms. The heat-capacity route sidesteps it: anchor the entropy at a moderate T0 with
gaussian_entropy (the motion is Gaussian there; checked on Einstein solids to 0.005) and carry it to T with the
enthalpy of NPT runs, entropy_from_heat_capacity (exact thermodynamics, H(T) fitted by a cubic). Nothing was fitted to
reference values.

Cases (EAM potentials, never run before at these temperatures). T0 = 0.35 of the experimental melting point, rounded.
                       T0      T                        T0      T
  fcc Ag (Foiles u3)   430   1000        fcc Pt (Foiles u3)   710   1600
  fcc Au (Foiles u3)   470   1080        fcc Cu (Mishin)      480   1150
  hcp Mg (Mendelev)    320    780        fcc Al (Mendelev)    330    780
  bcc Fe (Mendelev)    630   1550        bcc V (Mendelev)     760   1850
  fcc Ni (Foiles u3)   600   1450
Runs per case (run_fresh3.sh): solid.lmp at T0 (anchor, 200 ps NVT at the relaxed box, 401 frames) and at T (exact
reference); npt.lmp at seven temperatures evenly from T0 to T (20 ps to settle, 50 ps averaged); calphy at T and at T0.
Exact:   S(T) = (U + 3/2 k T - G) / T as before, U from the NVT run at T, G from calphy (centre of mass included).
Hermes:  S0 = gaussian_entropy(anchor frames, T0)["entropy"];
         S(T) = entropy_from_heat_capacity(ladder T, H = <PE> + 3/2 k T + <P V>, T0, S0, at=T)["entropy"] (cubic fit).
Pass:    |S_Hermes(T) - S_exact(T)| <= 1.0 J/(K mol) in every valid case.
Valid:   gaussian_entropy accepts both the T0 and the T trajectory as a crystal, and both calphy runs have dissipation
         <= 0.01 eV/atom (a run above that is redone once with 100000 switching steps before this script is run).
         Excluded cases are reported; with more than two excluded the test is void.
Reported alongside, outside the verdict: the anchor's error at T0, the compression and Gaussian routes run directly at
T, and the fit residual of H(T).
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
from hermes.compression_entropy import crystal_correlation_entropy, gaussian_entropy, trajectory_entropy  # noqa: E402
from hermes.heat_capacity import entropy_from_heat_capacity  # noqa: E402
from hermes.md_configuration import CLASSICAL_ENTROPY_OFFSET  # noqa: E402

FROZEN = {
    "hermes/compression_entropy.py": "ded8dc2ebc211c6faf5ddacaf1c0785cf1c43809cd3935d6379902f678ba388e",
    "hermes/heat_capacity.py": "541d7a3483bf84adc8949d6a492450e954284d4ff050bf7b6730e11838552b80",
    "hermes/md_configuration.py": "b7ee8dde8fb0ce6b732256b5ceda139403b806ee7fda0360e6646a5d1493e26e",
}
HERE = os.path.dirname(os.path.abspath(__file__))
EV = 96485.33212
KB = 8.617333262e-5
BAR_A3_TO_EV = 1e5 * 1e-30 / 1.602176634e-19
TOLERANCE, MAX_DISSIPATION = 1.0, 0.01
CASES = {  # tag: (lattice, mass amu, T0, T)
    "ag": ("fcc", 107.87, 430, 1000), "pt": ("fcc", 195.09, 710, 1600), "au": ("fcc", 196.97, 470, 1080),
    "cum": ("fcc", 63.55, 480, 1150), "mg": ("hcp", 24.305, 320, 780), "al": ("fcc", 26.98154, 330, 780),
    "fe": ("bcc", 55.845, 630, 1550), "v": ("bcc", 50.9415, 760, 1850), "ni": ("fcc", 58.71, 600, 1450),
}


def ladder(t0, t):
    return [int(round(x)) for x in np.linspace(t0, t, 7)]


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


def last_row(path):
    return [float(x) for x in [ln.split() for ln in open(path) if not ln.startswith("#")][-1]]


def exact_entropy(run, temperature):
    u = last_row(os.path.join(HERE, f"{run}_thermo.dat"))[1]
    (folder,) = [d for d in glob.glob(os.path.join(HERE, f"{run}-fe-*")) if os.path.isdir(d)]
    report = yaml.safe_load(open(os.path.join(folder, "report.yaml")))["results"]
    return (u + 1.5 * KB * temperature - report["free_energy"]) / temperature * EV, report["dissipation"]


def main():
    check_frozen()
    out, excluded = {}, []
    for tag, (lattice, mass, t0, t) in CASES.items():
        exact_t, diss_t = exact_entropy(f"{tag}_t", t)
        exact_0, diss_0 = exact_entropy(f"{tag}_a", t0)
        frames_0, cell_0 = read_dump(os.path.join(HERE, f"{tag}_a.dump"))
        frames_t, cell_t = read_dump(os.path.join(HERE, f"{tag}_t.dump"))
        masses = np.full(frames_0.shape[1], mass)
        reason = None
        try:
            anchor = gaussian_entropy(frames_0, cell_0, masses, t0)["entropy"]
            gauss_t = gaussian_entropy(frames_t, cell_t, masses, t)["entropy"]
        except ValueError as e:
            reason = str(e)
        if reason is None and max(diss_t, diss_0) > MAX_DISSIPATION:
            reason = f"calphy dissipation {max(diss_t, diss_0):.4f} eV/atom"
        if reason:
            excluded.append(tag)
            out[tag] = {"excluded": reason}
            print(f"{tag}: EXCLUDED ({reason})")
            continue
        temps = ladder(t0, t)
        h = []
        for k, tk in enumerate(temps):
            _, pe, pv = last_row(os.path.join(HERE, f"{tag}_h{k}_npt.dat"))
            h.append((pe + 1.5 * KB * tk + pv * BAR_A3_TO_EV) * EV)
        hc = entropy_from_heat_capacity(temps, h, t0, anchor, at=t)
        hermes = float(hc["entropy"][0])
        comp = trajectory_entropy(frames_t, cell_t, masses, t, n_pairs=60, min_separation=10)
        direct = comp["mean"] + CLASSICAL_ENTROPY_OFFSET - crystal_correlation_entropy(frames_t, cell_t)["entropy"]
        out[tag] = {"lattice": lattice, "T0": t0, "T": t, "exact": exact_t, "hermes": hermes, "error": hermes - exact_t,
                    "anchor": anchor, "anchor_error": anchor - exact_0, "direct_compression_error": direct - exact_t,
                    "direct_gaussian_error": gauss_t - exact_t, "fit_residual_J": hc["residual"],
                    "dissipation": [diss_0, diss_t], "pass": abs(hermes - exact_t) <= TOLERANCE}
        print(f"{tag:4s} {lattice} T0 {t0:4d} -> T {t:4d}  exact {exact_t:7.2f}  Hermes {hermes:7.2f} ({hermes - exact_t:+5.2f})  "
              f"anchor error {anchor - exact_0:+5.2f}  direct: compression {direct - exact_t:+5.2f}, Gaussian {gauss_t - exact_t:+5.2f}  "
              f"H fit rms {hc['residual']:.1f} J/mol  {'PASS' if out[tag]['pass'] else 'FAIL'}", flush=True)
    void = len(excluded) > 2
    verdict = "VOID" if void else ("PASS" if all(v.get("pass", True) for v in out.values()) else "FAIL")
    errs = np.array([v["error"] for v in out.values() if "error" in v])
    direct = np.array([v["direct_compression_error"] for v in out.values() if "error" in v])
    print(f"heat-capacity route: mean {errs.mean():+.2f}, max |error| {np.abs(errs).max():.2f}; direct compression route: "
          f"mean {direct.mean():+.2f}, max |error| {np.abs(direct).max():.2f}; excluded {excluded}")
    print(f"verdict {verdict}")
    out["verdict"] = verdict
    json.dump(out, open(os.path.join(HERE, "fresh3_results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
