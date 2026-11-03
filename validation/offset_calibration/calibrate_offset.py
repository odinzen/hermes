"""Calibrate the offset for 256-atom groups on exactly sampled Einstein solids: masses x amplitudes x box sizes."""
import json
import sys

import numpy as np
import scipy.constants

sys.path.insert(0, r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad")
import chunk_offset as co  # noqa: E402

R = scipy.constants.R
rng = np.random.default_rng(20261015)
rows = []
for mass in (12.0, 27.0, 60.0, 180.0):
    temperature = 1000.0
    lam = co.ce.thermal_de_broglie_wavelength([mass], temperature)[0]
    for ratio in (1.0, 1.5, 2.0, 3.0, 4.5):
        sigma = ratio * lam
        omega = np.sqrt(scipy.constants.k * temperature / (mass * scipy.constants.atomic_mass * (sigma * 1e-10) ** 2))
        exact = 3 * R * (1 + np.log(scipy.constants.k * temperature / (scipy.constants.hbar * omega)))
        for n in (7, 10, 13):
            a = max(4.05, 12 * sigma)  # far enough apart that no atom is mapped to a neighbour
            sites = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float) * a
            cell = np.eye(3) * n * a
            vals = [co.chunked_entropy(sites + rng.normal(0, sigma, sites.shape), sites + rng.normal(0, sigma, sites.shape),
                                       cell, lam, rng) for _ in range(max(2, 3000 // n**3))]
            rows.append({"mass": mass, "ratio": ratio, "N": n**3, "offset": exact - float(np.mean(vals)),
                         "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals)))})
            print(json.dumps(rows[-1]), flush=True)
off = np.array([r["offset"] for r in rows])
sem = np.array([r["sem"] for r in rows])
w = 1 / sem**2
mean = np.sum(w * off) / np.sum(w)
print(f"weighted mean offset {mean:.3f} +- {1 / np.sqrt(np.sum(w)):.3f}; chi2_red {np.sum(w * (off - mean) ** 2) / (len(off) - 1):.2f}")
for key in ("mass", "ratio", "N"):
    for val in sorted({r[key] for r in rows}):
        sel = np.array([r[key] == val for r in rows])
        print(f"{key} {val}: {np.sum(w[sel] * off[sel]) / np.sum(w[sel]):.3f}")
json.dump(rows, open(r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad\offset_calibration.json", "w"), indent=1)
