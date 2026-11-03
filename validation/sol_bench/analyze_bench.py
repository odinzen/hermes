"""Potential benchmark, outside every verdict: MACE-MP-0, MPA-0 and MP-0b3 on the development cases of the closed-shell
oxide test, Arm A and B2, every result reported.

For each model and case: the 298.15 K MD volume against the measured one (the entropy-blind potential choice drafted in
../oxide_heldout/RULE.md), whether the time-averaged structure keeps its space group, and the entropy by the Hermes 0.3
route (limit="bulk") with the box value beside it. Arm A adds the magnetic term and B2 the crystal-field term exactly as
their frozen scripts did. The electronic term of the Materials Project density of states is listed apart and left out of
the total, since every case is an insulator whose measured gap closes it (../explore/effect_v03.py).

MACE-MP-0 in the box limit must reproduce the frozen tests' S_vib, which checks this script against them.

    python analyze_bench.py [--fresh]     -> bench_output.txt, bench_results.json; predictions cached in predictions/
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
sys.path.insert(0, str(VALIDATION.parent))
sys.path.insert(0, str(HERE))
from hermes.compression_entropy import gaussian_entropy  # noqa: E402
from hermes.heat_capacity import entropy_from_heat_capacity  # noqa: E402
from hermes.quantum_correction import quantum_correction, vibrational_density_of_states  # noqa: E402
import volumes  # noqa: E402

DATA = Path.home() / "Code/juno/assets/local-archive/hermes-validation/data"
# Each model's runs, Sol first. MP-0's Al2O3 ran on the local GPU in the first campaign, with the same run_oxides.py.
RUNS = {"mp-0": [VALIDATION / "oxide_vibration" / "runs"],
        **{m: [DATA / "bench-sol-2026-10-27/results" / m / "runs", DATA / "bench-local-2026-10-27" / m / "runs"]
           for m in ("mpa-0", "mp-0b3")}}


def run_folder(model, tag):
    return next((r / tag for r in RUNS[model] if (r / tag / "progress.log").exists()
                 and "CASE_DONE" in (r / tag / "progress.log").read_text()), None)
LIMITS = ("bulk", "box")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vib = load("analyze_oxides", VALIDATION / "oxide_vibration" / "analyze_oxides.py")
arm_a = load("analyze_arm_a", VALIDATION / "extra_terms" / "analyze_arm_a.py")
arm_b = load("analyze_arm_b", VALIDATION / "extra_terms" / "analyze_arm_b.py")
from hermes.magnetic import magnetic_properties  # noqa: E402

T0, T1 = vib.T0, vib.T1


def cases():
    """tag: (group, composition, atoms per formula unit)."""
    out = {t: ("oxide", c[0], c[1]) for t, c in vib.CASES.items()}
    out |= {t: ("arm A", c[0], c[4]) for t, c in arm_a.CASES.items()}
    conv = json.loads((VALIDATION / "extra_terms" / "arm_b_conventions.json").read_text(encoding="utf-8"))["compounds"]
    out |= {c.lower(): ("B2", c, 6) for c, v in conv.items() if v["status"] == "scored"}
    return out


def predict(folder):
    """The frozen route of analyze_oxides.predict, with the 298.15 K anchor in both limits."""
    ladder = json.loads((folder / "ladder.json").read_text())
    t0, t1 = np.load(folder / "nvt_t0.npz"), np.load(folder / "nvt_t1.npz")
    out = {"space_group_T0": vib.space_group(t0["positions"], t0["cell"], t0["numbers"]),
           "space_group_T1": vib.space_group(t1["positions"], t1["cell"], t1["numbers"])}
    temperatures = np.array([row["T"] for row in ladder])
    enthalpies = np.array([row["pe_eV_atom"] + 1.5 * vib.KB * row["T"] + row["pv_eV_atom"] for row in ladder]) * vib.EV
    qc = []
    for run, temperature in ((t0, T0), (t1, T1)):
        nu, dos = vibrational_density_of_states(run["velocities"], float(run["velocity_dt_ps"]), run["masses"],
                                                segment_ps=5.0)
        qc.append(quantum_correction(nu, dos, temperature))
    out["QC"] = qc
    for limit in LIMITS:
        try:
            anchor = gaussian_entropy(t0["positions"], t0["cell"], t0["masses"], T0, species=t0["numbers"],
                                      limit=limit)["entropy"]
        except ValueError as err:
            out["refused"] = str(err)
            return out
        carried = entropy_from_heat_capacity(temperatures, enthalpies, T0, anchor, at=[T0, T1])
        out[limit] = [anchor + qc[0], float(carried["entropy"][1]) + qc[1]]
    return out


def reference_space_group(tag):
    """Space group of the starting structure, as recorded when it was fetched from the Materials Project."""
    for name in ("oxide_vibration/structures.json", "extra_terms/magnetic.json", "extra_terms/phosphates.json"):
        record = json.loads((VALIDATION / name).read_text(encoding="utf-8"))
        if tag in record:
            return record[tag]["space_group"]
    raise KeyError(tag)


def measured(tag, group, composition, apf, manifests):
    if group == "B2":
        conv = json.loads((VALIDATION / "extra_terms" / "arm_b_conventions.json").read_text(encoding="utf-8"))
        c = conv["compounds"][composition]
        m = arm_b.measured(manifests["phosphate"], composition)
        values = [m[t]["mean"] for t in (T0, T1)]
        if c.get("add_to_1000K") and values[1] is not None:
            values[1] += c["add_to_1000K"]
        return [v / apf if v is not None else None for v in values]
    exclude = (vib.CASES.get(tag) or arm_a.CASES[tag])[-1]
    m = vib.measured(manifests["oxide"], composition, exclude, apf)
    return [m["S298"], m.get("S1000")]


def extra_terms(tag, group, composition):
    """The non-vibrational term each frozen arm added, and the MP electronic term (reported, not added)."""
    s_el = arm_a.electronic(tag) if group != "oxide" else [0.0, 0.0]
    if group == "arm A":
        _, spin, t_n, ions, apf, _ = arm_a.CASES[tag]
        extra = [magnetic_properties(t, t_n, 2 * spin, arm_a.P, reference="ordered", ions_per_formula=ions,
                                     atoms_per_formula=apf)["entropy"] for t in (T0, T1)]
    elif group == "B2":
        conv = json.loads((VALIDATION / "extra_terms" / "arm_b_conventions.json").read_text(encoding="utf-8"))
        levels = json.loads((VALIDATION / "extra_terms" / "levels.json").read_text(encoding="utf-8"))["levels"]
        c = conv["compounds"][composition]
        extra = [x / 6 for x in arm_b.predicted(levels[c["ion"]], c["include_ground_degeneracy"])]
    else:
        extra = [0.0, 0.0]
    return [float(x) for x in extra], [float(x) for x in s_el]


def frozen_s_vib():
    """S_vib at 298.15 and 1000 K as the frozen MACE-MP-0 tests scored them (box limit, Hermes 0.2)."""
    out = {}
    for tag, r in json.loads((VALIDATION / "oxide_vibration" / "oxide_results.json").read_text())["cases"].items():
        out[tag] = r["prediction"]
    for tag, r in json.loads((VALIDATION / "extra_terms" / "arm_a_results.json").read_text())["cases"].items():
        out[tag] = r["prediction"]
    for comp, r in json.loads((VALIDATION / "extra_terms" / "b2_results.json").read_text()).items():
        out[comp.lower()] = r["prediction"]
    return {t: [p["S_vib_T0"], p["S_vib_T1"]] for t, p in out.items() if "S_vib_T0" in p}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="recompute every prediction")
    args = parser.parse_args()
    cache = HERE / "predictions"
    cache.mkdir(exist_ok=True)
    manifests = {"oxide": json.loads(vib.MANIFEST.read_text(encoding="utf-8")),
                 "phosphate": json.loads(arm_b.MANIFEST.read_text(encoding="utf-8"))}
    results = {}
    for tag, (group, composition, apf) in cases().items():
        meas = measured(tag, group, composition, apf, manifests)
        extra, s_el = extra_terms(tag, group, composition)
        row = {"group": group, "composition": composition, "measured": meas, "extra": extra, "S_el_MP": s_el,
               "space_group_ref": reference_space_group(tag), "models": {}}
        for model in RUNS:
            folder = run_folder(model, tag)
            if folder is None:
                row["models"][model] = {"status": "not run"}
                continue
            file = cache / f"{model}_{tag}.json"
            if file.exists() and not args.fresh:
                p = json.loads(file.read_text())
            else:
                p = predict(folder)
                file.write_text(json.dumps(p, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
                print(f"{model} {tag} done", flush=True)
            p["folder"] = str(folder)
            sg = (p["space_group_T0"][1], p["space_group_T1"][1])
            p["status"] = "refused" if "refused" in p else (
                "symmetry lost" if any(s != row["space_group_ref"] for s in sg) else "valid")
            if tag in volumes.OXIDES:
                p["volume_deviation"] = volumes.md_volume(folder, apf) / volumes.measured_volume(
                    tag, manifests["oxide"]) - 1
            for limit in LIMITS:
                if limit in p:
                    total = [p[limit][i] + extra[i] for i in (0, 1)]
                    p[f"total_{limit}"] = total
                    p[f"error_{limit}"] = [total[i] - meas[i] if meas[i] is not None else None for i in (0, 1)]
            row["models"][model] = p
        results[tag] = row

    check = []
    for tag, want in frozen_s_vib().items():
        got = results[tag]["models"]["mp-0"].get("box")
        worst = max(abs(got[i] - want[i]) for i in (0, 1)) if got else float("inf")
        check.append(worst)
        if worst > 1e-6:
            raise SystemExit(f"{tag}: MACE-MP-0 box S_vib {got} does not reproduce the frozen {want}")

    f = lambda x, w=6: f"{x:{w}.2f}" if isinstance(x, (int, float)) else f"{'-':>{w}s}"  # noqa: E731
    lines = ["Potential benchmark, outside every verdict. J/(K mol of atoms); err = predicted - measured.",
             f"Check: MACE-MP-0 box-limit S_vib reproduces all {len(check)} frozen values (worst {max(check):.1e}).",
             "Every run on Sol A100s except MACE-MP-0 Al2O3, run on the local GPU (same run_oxides.py).", "",
             "Molar volume at 298.15 K, MD against measured (Robie and Hemingway 1995)",
             f"{'case':7s} {'meas':>6s}" + "".join(f" {m:>9s}" for m in RUNS)]
    for tag in volumes.OXIDES:
        lines.append(f"{tag:7s} {volumes.measured_volume(tag, manifests['oxide']):6.2f}" + "".join(
            f" {100 * results[tag]['models'][m]['volume_deviation']:+8.2f}%" if "volume_deviation" in
            results[tag]["models"][m] else f" {'not run':>9s}" for m in RUNS))
    common = [t for t in volumes.OXIDES if all("volume_deviation" in results[t]["models"][m] for m in RUNS)]
    lines.append(f"{'mean |dev|':14s}" + "".join(f" {100 * np.mean([abs(results[t]['models'][m]['volume_deviation']) for t in common]):8.2f}%"
                                                 for m in RUNS) + f"   over the {len(common)} oxides all models ran")
    for limit in LIMITS:
        lines += ["", f"Entropy, limit={limit!r}: error at 298.15 K | 1000 K"
                  + (" (Hermes 0.3)" if limit == "bulk" else " (the frozen tests' limit)"),
                  f"{'case':9s} {'group':6s} {'meas298':>7s} {'meas1000':>8s}" + "".join(f" {m:>13s}" for m in RUNS)]
        for tag, row in results.items():
            cells = []
            for m in RUNS:
                p = row["models"][m]
                if f"error_{limit}" not in p:
                    cells.append(f" {p['status']:>13s}")
                    continue
                e = p[f"error_{limit}"]
                cells.append(f" {f(e[0])} {f(e[1])}" + ("*" if p["status"] != "valid" else ""))
            lines.append(f"{tag:9s} {row['group']:6s} {f(row['measured'][0], 7)} {f(row['measured'][1], 8)}" + "".join(cells))
        for group in ("oxide", "arm A", "B2"):
            parts = []
            for m in RUNS:
                e = [results[t]["models"][m][f"error_{limit}"][0] for t in results if results[t]["group"] == group
                     and results[t]["models"][m].get("status") == "valid"
                     and results[t]["models"][m][f"error_{limit}"][0] is not None]
                parts.append(f"{m} {np.mean(e):+.2f} (n={len(e)})" if e else f"{m} -")
            lines.append(f"  mean 298.15 K error, valid {group} cases: " + ", ".join(parts))
    lines += ["", "* symmetry lost or route refused; such a case is reported, not counted in the means.",
              "Space group changes: " + "; ".join(
                  f"{m} {t} {p['space_group_T0'][1]}/{p['space_group_T1'][1]} (ref {r['space_group_ref']})"
                  for t, r in results.items() for m, p in r["models"].items() if p.get("status") == "symmetry lost")]
    text = "\n".join(lines)
    print(text)
    (HERE / "bench_output.txt").write_text(text + "\n", encoding="utf-8")
    (HERE / "bench_results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.tolist()
                                                        if hasattr(o, "tolist") else str(o)), encoding="utf-8")


if __name__ == "__main__":
    main()
