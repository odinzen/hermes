"""Second held-out test of the S_cc excess-entropy route. Rule written 2026-10-14 BEFORE any new data existed.

Why a second test: the first (rep/analyze_rep.py) FAILED. Its four compositions at one temperature gave 8 data,
too few for the modified AICc to keep any excess-entropy term. Since then candidate_feature_sets was corrected
to ESPEI's per-order scheme (parity checked against espei 0.9.0). Nothing else in the estimator changed.

Data, 10 compositions of liquid Cu-Zr (Mendelev EAM) at 2300 K, 6912 atoms per run:
  old  x_Zr 0.20 / 0.36 / 0.50 / 0.70, three runs each (run-level values from rep/rep_results.json, made by the
       same code this script hash-checks);
  new  x_Zr = n/864 with n = 86, 242, 372, 518, 691, 778 (0.10 / 0.28 / 0.43 / 0.60 / 0.80 / 0.90), two runs each
       (seeds 71031, 90217), produced by run_campaign.sh with the same LAMMPS input as the replicas.
Estimator, fixed in advance and unchanged from the first test:
  per run   least-squares 1/S_cc(k) = A + B k^2 over every reciprocal vector with k <= 0.30 1/A, S_cc(0) = 1/A;
  per x     mean of the run-level values; error = pooled within-composition standard deviation over all ten
            compositions (two runs alone give a poor SD) divided by sqrt(runs at that x).
Exact references: S_ex and H_mix from calphy + NVT energies, old points from the station2 cuzr_liquid_mixing.json,
new points from new_truth.py (same formula as analyze_cuzr.py, pure ends taken from that same JSON).
Excess model: hermes.excess_model.select_excess_model(max_order=3, temperature_features=("a", "b")) on all ten
compositions, H_mix error 50 J/mol, S_cc errors as above (floor 1e-3).
Pass criteria, both required:
  A  |S_ex(model) - S_ex(exact)| <= 0.5 J/(K mol) at every one of the ten compositions;
  B  at every composition the S_cc(0) mean lies within the exact-implied range widened by 2 errors, where the
     range is the min..max of the S_cc(0) implied by Redlich-Kister L0..L2 and L0..L3 least-squares fits to the
     ten exact G_ex = H_mix - T S_ex values, S_cc(0) = 1 / (1 / (x(1-x)) + G_ex'' / RT).
Nothing in this rule may be changed after the new data are read.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERMES = r"C:\Users\busta\Code\hermes-private"
sys.path.insert(0, HERMES)
from hermes.compression_entropy import GAS_CONSTANT as R  # noqa: E402
from hermes.concentration_fluctuations import concentration_structure_factor  # noqa: E402
from hermes.excess_model import MixingData, _rk_basis, select_excess_model  # noqa: E402

FROZEN = {
    "hermes/excess_model.py": "953553f9e1bcf10768e08e34e16c1deec1d6b1b665bdccf7a4726f498108b0f8",
    "hermes/concentration_fluctuations.py": "717c174d59aae95bec20cdacf8fe250399fceece6856f1ed81b62220234043d6",
}
K_FIT = 0.30
T = 2300.0
HERE = os.path.dirname(os.path.abspath(__file__))
OLD_RUNS = os.path.join(HERE, "..", "rep", "rep_results.json")
OLD_TRUTH = r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\results\cuzr_liquid_mixing.json"
NEW_TRUTH = os.path.join(HERE, "truth.json")
NEW = {"010": 86, "028": 242, "043": 372, "060": 518, "080": 691, "090": 778}
SEEDS = (71031, 90217)


def check_frozen():
    for rel, want in FROZEN.items():
        got = hashlib.sha256(open(os.path.join(HERMES, rel), "rb").read()).hexdigest()
        if got != want:
            sys.exit(f"{rel} changed since the rule was written ({got}); refusing to run")


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
    a, _ = np.linalg.lstsq(design, 1.0 / s[use], rcond=None)[0]
    return 1.0 / a


def implied_scc(x_fit, g_ex, x_eval, n_terms):
    basis = np.stack([_rk_basis(x_fit, k) for k in range(n_terms)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, g_ex, rcond=None)
    g2 = np.stack([_rk_basis(x_eval, k, derivative=2) for k in range(n_terms)], axis=1) @ coef
    return 1.0 / (1.0 / (x_eval * (1 - x_eval)) + g2 / (R * T))


def main():
    check_frozen()
    old_truth, new_truth = json.load(open(OLD_TRUTH)), json.load(open(NEW_TRUTH))
    old_runs = json.load(open(OLD_RUNS))

    runs, x, s_exact, h = {}, {}, {}, {}
    for t in ("020", "036", "050", "070"):
        runs[t] = list(old_runs[t]["runs"])
        x[t] = old_truth["per_composition"][t]["x_Zr"]
        s_exact[t] = old_truth["dS_mix"]["truth_calphy"][t] - old_truth["dS_mix"]["ideal"][t]
        h[t] = old_truth["dS_mix"]["mixing_enthalpy_kJ"][t] * 1000
    for t, n_zr in NEW.items():
        runs[t] = []
        for seed in SEEDS:
            frames, types, cell = read_dump(os.path.join(HERE, f"big_x{t}_s{seed}.dump"))
            if not np.isclose(types.mean(), n_zr / 864):
                sys.exit(f"x{t} s{seed}: Zr fraction {types.mean()} is not {n_zr}/864")
            r = concentration_structure_factor(frames, cell, types, k_max=K_FIT + 1e-9, n_blocks=2)
            runs[t].append(oz_intercept(r["k"], r["s_cc"]))
            print(f"x{t} s{seed}: {len(frames)} frames, {int((r['k'] <= K_FIT).sum())} vectors, "
                  f"S_cc(0) {runs[t][-1]:.4f}", flush=True)
        x[t] = n_zr / 864
        s_exact[t] = new_truth["S_ex"][t]
        h[t] = new_truth["H_mix_J"][t]

    tags = sorted(runs, key=lambda t: x[t])
    ss = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in runs.values())
    dof = sum(len(v) - 1 for v in runs.values())
    pooled_sd = float(np.sqrt(ss / dof))
    xs = np.array([x[t] for t in tags])
    mean = np.array([np.mean(runs[t]) for t in tags])
    err = np.array([pooled_sd / np.sqrt(len(runs[t])) for t in tags])
    s_ex_exact = np.array([s_exact[t] for t in tags])
    h_mix = np.array([h[t] for t in tags])
    print(f"pooled run-level SD {pooled_sd:.4f} ({dof} dof)")

    g_ex = h_mix - T * s_ex_exact
    implied = np.stack([implied_scc(xs, g_ex, xs, n) for n in (3, 4)])
    lo, hi = implied.min(axis=0), implied.max(axis=0)
    ok_b = (lo - 2 * err <= mean) & (mean <= hi + 2 * err)

    n = len(tags)
    data = MixingData(enthalpy_x=xs, enthalpy_t=np.full(n, T), enthalpy=h_mix, enthalpy_error=np.full(n, 50.0),
                      scc_x=xs, scc_t=np.full(n, T), scc=mean, scc_error=np.maximum(err, 1e-3))
    best, _ = select_excess_model(data, max_order=3, temperature_features=("a", "b"))
    s_model, s_model_err = best.entropy(xs, T)
    ok_a = np.abs(s_model - s_ex_exact) <= 0.5

    print(f"excess model: {best.features}, chi2_red {best.chi2_reduced:.2f}\n")
    print("  x_Zr  held  S_cc(0)        implied range      B     S_ex model       exact    A")
    for i, t in enumerate(tags):
        held = "new" if t in NEW else "old"
        print(f"{xs[i]:6.3f}  {held}  {mean[i]:.4f}+-{err[i]:.4f}  {lo[i]:.4f}-{hi[i]:.4f}  "
              f"{'PASS' if ok_b[i] else 'FAIL'}  {s_model[i]:6.2f}+-{s_model_err[i]:4.2f}  {s_ex_exact[i]:6.2f}  "
              f"{'PASS' if ok_a[i] else 'FAIL'}")
    new_idx = [i for i, t in enumerate(tags) if t in NEW]
    print(f"\nnew compositions only: max |S_ex - exact| {np.abs(s_model - s_ex_exact)[new_idx].max():.2f}")
    verdict = bool(ok_a.all() and ok_b.all())
    print(f"criterion A {'PASS' if ok_a.all() else 'FAIL'}, criterion B {'PASS' if ok_b.all() else 'FAIL'}, "
          f"verdict {'PASS' if verdict else 'FAIL'}")

    out = {"tags": tags, "x_Zr": xs.tolist(), "runs": {t: runs[t] for t in tags}, "pooled_sd": pooled_sd,
           "scc_mean": mean.tolist(), "scc_err": err.tolist(), "implied_lo": lo.tolist(), "implied_hi": hi.tolist(),
           "pass_b": ok_b.tolist(), "features": best.features, "chi2_reduced": best.chi2_reduced,
           "s_ex_model": s_model.tolist(), "s_ex_model_err": s_model_err.tolist(), "s_ex_exact": s_ex_exact.tolist(),
           "pass_a": ok_a.tolist(), "verdict": verdict}
    json.dump(out, open(os.path.join(HERE, "new_results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
