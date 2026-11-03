"""Held-out test of the long-wavelength S_cc estimator. Rule written 2026-10-09 BEFORE the replica data existed.

Estimator (Ornstein-Zernike form, fixed in advance):
  per run: least-squares fit of 1/S_cc(k) = A + B k^2 over every reciprocal vector with k <= 0.30 1/A
           (equal weight per vector); S_cc(0) = 1/A.
  per composition: mean of the run-level S_cc(0) over the three independent 6912-atom runs (the original big
           run, seed 4928, plus replicas 71031 and 90217); error = standard deviation / sqrt(3).
Pass criteria, per composition, against the calphy-implied S_cc(0) range (Redlich-Kister order 2 and 3 fits
to the exact G_ex): the mean lies within the range widened by 2 standard errors. Then S_ex from the ESPEI-style
excess model at 2300 K (H_ex = the calphy-run mixing enthalpies, S_cc errors as weights, AICc selection)
against the exact S_ex: pass if every |S_ex - exact| <= 0.5 J/(K mol).
Nothing in this rule may be changed after the replica data are read.
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\busta\Code\hermes-private")
from hermes.concentration_fluctuations import concentration_structure_factor  # noqa: E402
from hermes.excess_model import MixingData, select_excess_model  # noqa: E402

K_FIT = 0.30
HERE = r"C:\Users\busta\Code\hermes-private\validation\cuzr_scc_heldout"
TRUTH = r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\results\cuzr_liquid_mixing.json"
CALPHY_IMPLIED = {"020": (0.1565, 0.1850), "036": (0.1611, 0.1875), "050": (0.1419, 0.1743), "070": (0.1322, 0.1324)}
RUNS = {t: [rf"{HERE}\big\big_x{t}.dump", rf"{HERE}\rep\rep_x{t}_s71031.dump", rf"{HERE}\rep\rep_x{t}_s90217.dump"]
        for t in CALPHY_IMPLIED}


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


def oz_intercept(k, s):
    use = k <= K_FIT
    design = np.stack([np.ones(use.sum()), k[use] ** 2], axis=1)
    a, b = np.linalg.lstsq(design, 1.0 / s[use], rcond=None)[0]
    return 1.0 / a


truth = json.load(open(TRUTH))
out, s_mean, s_err, xs, h = {}, [], [], [], []
for t, paths in RUNS.items():
    vals = []
    for p in paths:
        frames, types, cell = read_dump(p)
        r = concentration_structure_factor(frames, cell, types, k_max=K_FIT + 1e-9, n_blocks=2)
        vals.append(oz_intercept(r["k"], r["s_cc"]))
        print(f"x{t} {p.split(chr(92))[-1]}: {len(frames)} frames, {int((r['k'] <= K_FIT).sum())} vectors, S_cc(0) {vals[-1]:.4f}",
              flush=True)
    m, e = float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    lo, hi = CALPHY_IMPLIED[t]
    ok = (lo - 2 * e) <= m <= (hi + 2 * e)
    out[t] = {"runs": vals, "mean": m, "err": e, "calphy_implied": [lo, hi], "pass": ok}
    print(f"x{t}: S_cc(0) {m:.4f} +- {e:.4f}   calphy-implied {lo:.4f}-{hi:.4f}   {'PASS' if ok else 'FAIL'}\n", flush=True)
    s_mean.append(m); s_err.append(e)
    xs.append(truth["per_composition"][t]["x_Zr"]); h.append(truth["dS_mix"]["mixing_enthalpy_kJ"][t] * 1000)

xs, h = np.array(xs), np.array(h)
s_exact = np.array([truth["dS_mix"]["truth_calphy"][t] - truth["dS_mix"]["ideal"][t] for t in RUNS])
data = MixingData(enthalpy_x=xs, enthalpy_t=np.full(4, 2300.0), enthalpy=h, enthalpy_error=np.full(4, 50.0),
                  scc_x=xs, scc_t=np.full(4, 2300.0), scc=s_mean, scc_error=np.maximum(s_err, 1e-3))
best, _ = select_excess_model(data, max_order=3, temperature_features=("a", "b"))
s, se = best.entropy(xs, 2300.0)
print(f"excess model: {best.features}, chi2_red {best.chi2_reduced:.2f}")
all_ok = True
for i, t in enumerate(RUNS):
    ok = abs(s[i] - s_exact[i]) <= 0.5
    all_ok &= ok
    print(f"x_Zr {xs[i]:.3f}: S_ex {s[i]:6.2f} +- {se[i]:4.2f}   exact {s_exact[i]:6.2f}   {'PASS' if ok else 'FAIL'}")
print("S_ex verdict:", "PASS" if all_ok else "FAIL")
out["excess_model"] = {"features": best.features, "s_ex": s.tolist(), "s_ex_err": se.tolist(), "s_exact": s_exact.tolist(),
                       "pass": bool(all_ok)}
json.dump(out, open(rf"{HERE}\rep\rep_results.json", "w"), indent=1, default=str)
