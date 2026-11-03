"""B2 of RULE.md: the phosphates' full prediction S_vib + S_cf + S_el against the measured entropy, per mole of atoms,
reported outside the verdict. S_vib by the frozen oxide vibration route (run_oxides.py unchanged, MP structures from
fetch_phosphates.py); S_cf with the levels and conventions Arm B used; S_el from the MP density of states. Also given
without S_el, since the MP densities of states of EuPO4 and YbPO4 are gapless (recorded before scoring).
Writes b2_output.txt and b2_results.json.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vib = load("analyze_oxides", HERE.parent / "oxide_vibration" / "analyze_oxides.py")
arm_a = load("analyze_arm_a", HERE / "analyze_arm_a.py")
arm_b = load("analyze_arm_b", HERE / "analyze_arm_b.py")
ATOMS = 6
TEMPERATURES = (vib.T0, vib.T1)


def main():
    arm_a.check_frozen()
    arm_b.check_frozen()
    manifest = json.loads(arm_b.MANIFEST.read_text(encoding="utf-8"))
    levels = json.loads((HERE / "levels.json").read_text(encoding="utf-8"))["levels"]
    conv = json.loads((HERE / "arm_b_conventions.json").read_text(encoding="utf-8"))["compounds"]
    results, lines = {}, ["B2, phosphates, full prediction per mole of atoms (outside the verdict). err = pred - meas.",
                          f"{'compound':9s} {'S_vib':>7s} {'S_cf':>6s} {'S_el':>6s} {'pred':>7s} {'meas':>7s} {'err':>6s} "
                          f"{'err_noel':>8s}   (298.15 K, then 1000 K)"]
    f = lambda x, w=7: f"{x:{w}.2f}" if isinstance(x, (float, np.floating)) else f"{'-':>{w}s}"  # noqa: E731
    for comp, c in conv.items():
        if c["status"] != "scored":
            continue
        tag = comp.lower()
        p = vib.predict(HERE.parent / "oxide_vibration" / "runs" / tag)
        m = arm_b.measured(manifest, comp)
        if c.get("add_to_1000K") and m[vib.T1]["mean"] is not None:
            m[vib.T1]["mean"] += c["add_to_1000K"]
        s_cf = [x / ATOMS for x in arm_b.predicted(levels[c["ion"]], c["include_ground_degeneracy"])]
        s_el = arm_a.electronic(tag)
        row = {"prediction": p, "S_cf": s_cf, "S_el": s_el, "S_meas": [], "S_pred": [], "error": [], "error_no_el": []}
        if "refused" not in p:
            s_vib = [p["S_vib_T0"], p["S_vib_T1"]]
            for i, t in enumerate(TEMPERATURES):
                meas = m[t]["mean"] / ATOMS if m[t]["mean"] is not None else None
                pred = s_vib[i] + s_cf[i] + s_el[i]
                row["S_meas"].append(meas)
                row["S_pred"].append(pred)
                row["error"].append(pred - meas if meas is not None else None)
                row["error_no_el"].append(pred - s_el[i] - meas if meas is not None else None)
                lines.append(f"{comp if i == 0 else '':9s} {f(s_vib[i])} {f(s_cf[i], 6)} {f(s_el[i], 6)} {f(pred)} "
                             f"{f(meas)} {f(row['error'][-1], 6)} {f(row['error_no_el'][-1], 8)}   {t:g} K")
        else:
            lines.append(f"{comp:9s} vibrational route refused: {p['refused']}")
        results[comp] = row
    text = "\n".join(lines)
    print(text)
    (HERE / "b2_output.txt").write_text(text + "\n", encoding="utf-8")
    (HERE / "b2_results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.tolist()
                                                     if hasattr(o, "tolist") else str(o)), encoding="utf-8")


if __name__ == "__main__":
    main()
