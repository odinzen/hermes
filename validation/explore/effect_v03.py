"""Exploration, outside every verdict: what the two 0.3 fixes change on the 21 returned trajectories. The bulk limit
replaces the box value of the 298.15 K Gaussian anchor (the carry-up to 1000 K and the quantum correction are
unchanged, so the same shift applies at both temperatures); a measured band gap opens the Materials Project densities
of states. Post-hoc: these compounds were already scored, so nothing here is a test result.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from hermes.compression_entropy import gaussian_entropy  # noqa: E402
from hermes.electronic import electronic_entropy  # noqa: E402

RUNS = HERE.parent / "oxide_vibration" / "runs"
ox = json.loads((HERE.parent / "oxide_vibration" / "oxide_results.json").read_text())["cases"]
arm_a = json.loads((HERE.parent / "extra_terms" / "arm_a_results.json").read_text())["cases"]
b2 = json.loads((HERE.parent / "extra_terms" / "b2_results.json").read_text())
out = {}
for tag in sorted(p.name for p in RUNS.iterdir()):
    d = np.load(RUNS / tag / "nvt_t0.npz")
    args = (d["positions"], d["cell"], d["masses"], 298.15)
    box = gaussian_entropy(*args, species=d["numbers"], limit="box")["entropy"]
    bulk = gaussian_entropy(*args, species=d["numbers"], limit="bulk")["entropy"]
    out[tag] = {"box": box, "bulk": bulk, "shift": bulk - box}
    print(f"{tag:8s} {len(d['masses']):4d} atoms  box {box:7.3f}  bulk {bulk:7.3f}  shift {bulk - box:+.3f}", flush=True)
# electronic entropy with a 2 eV gap, the smallest measured gap among these insulators being larger
for tag in ("coo", "fe2o3", "cr2o3", "eupo4", "ybpo4"):
    dos = np.load(HERE.parent / "extra_terms" / "dos" / f"{tag}.npz")
    g = dos["up"] + dos["down"]
    kw = dict(atoms=int(dos["atoms"]), fermi_energy=float(dos["efermi"]))
    s = electronic_entropy(dos["energies"], g, [298.15, 1000.0], **kw)
    s2 = electronic_entropy(dos["energies"], g, [298.15, 1000.0], band_gap=2.0, **kw)
    out[tag + "_el"] = {"as_published": s.tolist(), "gap_2eV": s2.tolist()}
    print(f"{tag:8s} S_el 298/1000 K: as published {s[0]:.3f}/{s[1]:.3f}  opened to 2 eV {s2[0]:.2e}/{s2[1]:.2e}")
(HERE / "effect_v03.json").write_text(json.dumps(out, indent=1))
