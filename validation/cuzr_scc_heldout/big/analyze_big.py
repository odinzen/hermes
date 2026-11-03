"""Bigger-box test of the S_cc route: 6912 vs 864 atoms, liquid Cu-Zr at 2300 K.

Rule fixed before the runs: same estimator (S_0 + a k^2 over k <= 0.8 1/A) on both boxes; also the raw
S_cc on the shells below 0.24 1/A that only the big box reaches. Targets: S_cc(0) implied by calphy G_ex
(Redlich-Kister fits of order 2 and 3 to the exact G_ex bracket it). x000 = random-label control (0.25).
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\busta\Code\hermes-private")
from hermes.concentration_fluctuations import concentration_structure_factor, long_wavelength_limit  # noqa: E402

N_BIG = 6912
HERE = r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\e7397efa-dcb0-4ef2-878d-eb03d3cbbcfd\scratchpad"
SMALL = json.load(open(rf"{HERE}\joint_fit.json"))["s_cc"]
CALPHY_IMPLIED = {"020": (0.1565, 0.1850), "036": (0.1611, 0.1875), "050": (0.1419, 0.1743), "070": (0.1322, 0.1324)}
IDEAL_X = {"000": 0.5, "020": 173 / 864, "036": 311 / 864, "050": 0.5, "070": 605 / 864}


def read_dump(path):
    frames, types, box = [], None, None
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
            data = data[np.argsort(data[:, 0])]
            types = data[:, 1].astype(int) - 1
            frames.append(data[:, 2:5])
            i += 7 + n
        else:
            i += 1
    return np.array(frames), types, np.diag(box)


out = {}
for t in ("000", "020", "036", "050", "070"):
    frames, types, cell = read_dump(rf"{HERE}\big\big_x{t}.dump")
    if t == "000":
        types = np.zeros(N_BIG, int)
        types[np.random.default_rng(1).permutation(N_BIG)[: N_BIG // 2]] = 1
    r = concentration_structure_factor(frames, cell, types, k_max=0.8, n_blocks=10)
    s0, err = long_wavelength_limit(r["k"], r["s_cc"], r["blocks"], k_fit=0.8)
    low = r["k"] < 0.24
    shells = {}
    for kk in np.unique(np.round(r["k"][low], 3)):
        sel = np.isclose(np.round(r["k"], 3), kk)
        shells[float(kk)] = (float(r["s_cc"][sel].mean()), int(sel.sum()))
    small = SMALL.get(f"('{t}', 2300)") or SMALL.get(str((t, 2300)))
    xb = IDEAL_X[t]
    out[t] = {"frames": len(frames), "S_cc0_big": s0, "err_big": err, "S_cc0_864": small,
              "calphy_implied": CALPHY_IMPLIED.get(t, (0.25, 0.25)), "ideal": xb * (1 - xb), "low_k_shells": shells}
    print(f"x{t}: frames {len(frames)}  big {s0:.4f} +- {err:.4f}   864-atom {small}   "
          f"calphy-implied {out[t]['calphy_implied']}   ideal {xb * (1 - xb):.4f}", flush=True)
    for kk, (v, m) in shells.items():
        print(f"     k {kk:.3f}: S_cc {v:.4f} ({m} vectors)")
json.dump(out, open(rf"{HERE}\big\big_results.json", "w"), indent=1)
