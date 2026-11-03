"""Non-Gaussian neighbour information on the twelve first-fresh MD crystals, against Hermes's error (with the
centre-of-mass term, as in the current code)."""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad")
sys.path.insert(0, r"C:\Users\busta\Code\hermes-private\validation\solid_fresh")
import analyze_fresh as af  # noqa: E402
import nongauss as ng  # noqa: E402

R = 8.314462618
res = json.load(open(r"C:\Users\busta\Code\hermes-private\validation\solid_fresh\fresh_results.json"))
rng = np.random.default_rng(42)
rows = []
for tag in af.CASES:
    frames, cell = af.read_dump(rf"C:\Users\busta\Code\hermes-private\validation\solid_fresh\{tag}.dump")
    u, sites = ng.displacements(frames, cell)
    n = frames.shape[1]
    s2 = float((u**2).sum(axis=2).mean() / 3)
    s_com = R / n * np.log(abs(np.linalg.det(cell)) * (n / (2 * np.pi * np.e * s2)) ** 1.5)
    err = res[tag]["error"] + s_com
    pairs, per_atom = ng.first_shell_pairs(sites, cell, 1500, rng)
    d, se = ng.excess_information(u, pairs, rng)
    excess = R * per_atom / 2 * d
    rows.append((tag, err, excess, R * per_atom / 2 * se))
    print(f"{tag:13s} Hermes error {err:+.2f}   non-Gaussian neighbour information {excess:+.2f} +- {R * per_atom / 2 * se:.2f} "
          f"J/(K mol)  ({per_atom / 2:.0f} pairs/atom)", flush=True)
e = np.array([r[1] for r in rows])
x = np.array([r[2] for r in rows])
print(f"mean Hermes error {e.mean():+.2f}; mean non-Gaussian information {x.mean():+.2f}; correlation across crystals "
      f"{np.corrcoef(e, x)[0, 1]:+.2f}")
json.dump(rows, open(r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad\nongauss_md.json", "w"))
