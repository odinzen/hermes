"""Score Arm B (lanthanide orthophosphates) of RULE.md, frozen 2026-10-22, with the levels, conventions and data
handling fixed in PREREGISTERED.txt before any phosphate crystal-field entropy was computed.

B1: |S_cf,pred - (S_meas - S_lat)| <= 1.0 J/(K mol of Ln) at 298.15 K and at 1000 K where both the compound and its
lattice reference have data. S_cf,pred = crystal_field_properties(levels, degeneracies, T)["entropy"] with
include_ground_degeneracy from arm_b_conventions.json. S_meas is the mean of the primary sources in Juno's
Phosphate-minerals manifest. B2 (full prediction with the vibrational route) needs MD of the phosphates and is not run.
Writes arm_b_output.txt and arm_b_results.json; refuses to run if the rule, code, levels or conventions changed.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HERMES = HERE.parents[1]
sys.path.insert(0, str(HERMES))
from hermes.crystal_field import crystal_field_properties  # noqa: E402

MANIFEST = Path.home() / "Code/juno/assessments/Phosphate-minerals/manifest.json"
FROZEN = {
    "validation/extra_terms/RULE.md": "9e75e71012c07987601ae0b01cf5fe2e0d8e2c8f78f3f6e7cb462e96dc118db9",
    "hermes/crystal_field.py": "0c4694a7425256a9a253ef16f281e74dfbde2667e0fee429b7bd435bb4428f96",
    "validation/extra_terms/levels.json": "255918a196290b48db389cf9968117feb9cca8eb42cb5a7c7428835816d8f4b0",
    "validation/extra_terms/arm_b_conventions.json": "dbbb49f533427898ceb0c2efc011b9123ae5d6136250340cf92e3ab8852bbef7",
}
TOLERANCE = 1.0
T0, T1 = 298.15, 1000.0


def check_frozen():
    for rel, want in FROZEN.items():
        got = hashlib.sha256((HERMES / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            raise SystemExit(f"{rel} changed since the freeze; refusing to score")


def measured(manifest, composition):
    """Mean S at T0 and T1 over the primary sources; LuPO4 at 1000 K is S - S(2 K), where S(2 K) is negligible."""
    out = {}
    for t in (T0, T1):
        rows = [a for a in manifest["artifacts"] if a["quantity"] == "standard_entropy"
                and a.get("evidence_tier") == "primary_experimental"
                and (a.get("conditions") or {}).get("composition") == composition
                and abs((a["conditions"].get("temperature_K") or 0) - t) < 0.01
                and "monoclinic" not in str(a.get("phase"))]
        out[t] = {"mean": float(np.mean([a["value"] for a in rows])) if rows else None,
                  "spread": float(np.ptp([a["value"] for a in rows])) if len(rows) > 1 else 0.0,
                  "sources": {a["id"]: a["value"] for a in rows}}
    return out


def predicted(entry, include_ground_degeneracy):
    energies = [level[1] for m in entry["multiplets"].values() for level in m]
    degeneracies = [level[2] for m in entry["multiplets"].values() for level in m]
    return [crystal_field_properties(energies, degeneracies, t, unit=entry["unit"],
                                     include_ground_degeneracy=include_ground_degeneracy)["entropy"]
            for t in (T0, T1)]


def main():
    check_frozen()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    levels = json.loads((HERE / "levels.json").read_text(encoding="utf-8"))["levels"]
    conv = json.loads((HERE / "arm_b_conventions.json").read_text(encoding="utf-8"))
    lattice = {name: measured(manifest, comp) for name, comp in conv["lattice"].items()}
    results, lines = {}, []
    for comp, c in conv["compounds"].items():
        row = {"status": c["status"], "why": c["why"]}
        results[comp] = row
        if c["status"] != "scored":
            continue
        m = measured(manifest, comp)
        if c.get("add_to_1000K") and m[T1]["mean"] is not None:
            m[T1]["mean"] += c["add_to_1000K"]
        lat = lattice[c["structure"]]
        s_cf = predicted(levels[c["ion"]], c["include_ground_degeneracy"])
        row.update({"S_meas": [m[T0]["mean"], m[T1]["mean"]], "S_meas_spread_298": m[T0]["spread"],
                    "S_lat": [lat[T0]["mean"], lat[T1]["mean"]], "S_cf_pred": s_cf,
                    "sources": {**m[T0]["sources"], **m[T1]["sources"]}})
        errors = []
        for i, t in enumerate((T0, T1)):
            if row["S_meas"][i] is None or row["S_lat"][i] is None:
                errors.append(None)
                continue
            errors.append(s_cf[i] - (row["S_meas"][i] - row["S_lat"][i]))
        row["error"] = errors
        row["B1_pass"] = all(e is None or abs(e) <= TOLERANCE for e in errors) and errors[0] is not None
        if c["structure"] == "xenotime":
            second = lattice["xenotime_second"]
            row["error_YPO4_lattice"] = [None if row["S_meas"][i] is None or second[t]["mean"] is None
                                         else s_cf[i] - (row["S_meas"][i] - second[t]["mean"])
                                         for i, t in enumerate((T0, T1))]
        row["other_convention_error"] = [
            None if row["S_meas"][i] is None or row["S_lat"][i] is None
            else predicted(levels[c["ion"]], not c["include_ground_degeneracy"])[i] - (row["S_meas"][i] - row["S_lat"][i])
            for i in range(2)]
        if comp == "TmPO4":
            row["variant_without_assumed_Rln2"] = [e + 5.76 if e is not None else None for e in errors]

    scored = [k for k, r in results.items() if r["status"] == "scored"]
    b1 = all(results[k]["B1_pass"] for k in scored)
    f = lambda x, w=7: f"{x:{w}.2f}" if isinstance(x, (float, np.floating)) else f"{'-':>{w}s}"  # noqa: E731
    lines.append(f"Extra-terms test, Arm B (LnPO4), rule frozen 2026-10-22. B1: {'PASS' if b1 else 'FAIL'} "
                 f"({sum(results[k]['B1_pass'] for k in scored)}/{len(scored)} compounds)")
    lines.append("J/(K mol of Ln). excess = S_meas - S_lat; err = S_cf,pred - excess. Lattice: LaPO4 (monazite), LuPO4 "
                 "(xenotime).")
    lines.append(f"{'compound':9s} {'S_cf':>7s} {'excess':>7s} {'err':>7s}   {'S_cf':>7s} {'excess':>7s} {'err':>7s}  "
                 f"{'pass':5s} {'err(YPO4 lat)':>15s} {'err(other conv)':>16s}   (298.15 K | 1000 K)")
    for comp, r in results.items():
        if r["status"] != "scored":
            lines.append(f"{comp:9s} {r['status']}: {r['why']}")
            continue
        exc = [None if r["S_meas"][i] is None or r["S_lat"][i] is None else r["S_meas"][i] - r["S_lat"][i]
               for i in range(2)]
        y = r.get("error_YPO4_lattice", [None, None])
        o = r["other_convention_error"]
        lines.append(f"{comp:9s} {f(r['S_cf_pred'][0])} {f(exc[0])} {f(r['error'][0])}   {f(r['S_cf_pred'][1])} "
                     f"{f(exc[1])} {f(r['error'][1])}  {str(r['B1_pass']):5s} {f(y[0])}/{f(y[1])} {f(o[0])}/{f(o[1])}")
    tm = results.get("TmPO4", {})
    if "variant_without_assumed_Rln2" in tm:
        v = tm["variant_without_assumed_Rln2"]
        lines.append(f"TmPO4 without the source's assumed R ln 2 (outside the verdict): err {f(v[0])} | {f(v[1])}")
    lines.append(f"Left out, no measured levels in a phosphate: {', '.join(conv['left_out_no_levels'])}. "
                 "B2 not run (needs MD of the phosphates).")
    text = "\n".join(lines)
    print(text)
    (HERE / "arm_b_output.txt").write_text(text + "\n", encoding="utf-8")
    (HERE / "arm_b_results.json").write_text(json.dumps({"B1": b1, "cases": results}, indent=1, default=float),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
