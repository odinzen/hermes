"""Score the polymorph test exactly as RULE.md sets it out.

    python analyze_polymorph.py RESULTS     RESULTS holds mpa-0/runs/<case> as the Sol package writes it
                                            -> polymorph_output.txt, polymorph_results.json beside this file

Refuses to run if the rule, the code under test or the measured records have changed since the freeze, or before every
case has finished: the rule scores all cases together.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HERMES = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "oxide_heldout"))
from analyze_heldout import EV, KB, T0, T1, space_group  # noqa: E402

sys.path.insert(0, str(HERMES))
from hermes.compression_entropy import gaussian_entropy  # noqa: E402
from hermes.heat_capacity import entropy_from_heat_capacity  # noqa: E402
from hermes.quantum_correction import harmonic_entropy, quantum_correction, vibrational_density_of_states  # noqa: E402

MANIFEST = HERMES.parent / "juno/assessments/Refractory-Oxides/manifest.json"
FROZEN = {  # path under Code/: SHA-256 with LF endings
    "hermes-private/validation/polymorph_heldout/RULE.md": "241e2dee0d44412f439ab10ae9ec73c5cf13d4dd8bf72a3a31bac72fe7f301c6",
    "hermes-private/hermes/compression_entropy.py": "512e1cf35bdd0b0e5728fe8c07f8dd1f707e5891e0f793a74a4e25176a246c77",
    "hermes-private/hermes/heat_capacity.py": "541d7a3483bf84adc8949d6a492450e954284d4ff050bf7b6730e11838552b80",
    "hermes-private/hermes/quantum_correction.py": "8d968a9b7c40ef36cbed0a64b6fe25339abb39de40d9b8deeb9db510928bbdf8",
    "hermes-private/validation/oxide_heldout/analyze_heldout.py": "e5cf7e4314f41c8aeacb82980c81993314762ca58090f4432eeeeee45313e030",
}
RECORDS = "60a272dd9d6411cd03dd6d9092fce35d54155b78829442e1be6dc610c90e0472"  # SHA-256 of the 21 Robie-Hemingway records
RECORD_PREFIX = "meas:robie_hemingway_1995:heldout2_"
NA_CM3 = 1e-24 * 6.02214076e23
MODEL = "mpa-0"
OFFSET = {T0: 1.87, T1: 2.20}
RMS_TOLERANCE, MAX_TOLERANCE, POLYMORPH_TOLERANCE = 1.0, 2.0, 0.5
MAX_EXCLUDED, MIN_FAMILIES = 3, 2
BOTH, ROOM = (T0, T1), (T0,)

CASES = {  # tag: (formula, atoms per formula unit, space-group number, temperatures scored)
    "quartz": ("SiO2", 3, 152, ROOM),
    "coesite": ("SiO2", 3, 15, ROOM),
    "stishovite": ("SiO2", 3, 136, ROOM),
    "calcite": ("CaCO3", 5, 167, ROOM),
    "aragonite": ("CaCO3", 5, 62, ROOM),
    "wollastonite": ("CaSiO3", 5, 2, BOTH),
    "pseudowollastonite": ("CaSiO3", 5, 15, BOTH),
    "enstatite": ("MgSiO3", 5, 61, BOTH),
    "clinoenstatite": ("MgSiO3", 5, 14, BOTH),
    "perovskite": ("CaTiO3", 5, 62, BOTH),
    "chrysoberyl": ("BeAl2O4", 7, 62, BOTH),
    "diopside": ("CaMgSi2O6", 10, 15, BOTH),
    "jadeite": ("NaAlSi2O6", 10, 15, BOTH),
}
FAMILIES = {  # family: (pairs as (a, b) for S_a - S_b, temperatures)
    "SiO2": ((("coesite", "quartz"), ("stishovite", "quartz"), ("stishovite", "coesite")), ROOM),
    "CaCO3": ((("aragonite", "calcite"),), ROOM),
    "CaSiO3": ((("pseudowollastonite", "wollastonite"),), BOTH),
    "MgSiO3": ((("clinoenstatite", "enstatite"),), BOTH),
}


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_records(manifest):
    return sorted((a for a in manifest["artifacts"] if str(a.get("id", "")).startswith(RECORD_PREFIX)),
                  key=lambda a: a["id"])


def records_sha(manifest):
    return hashlib.sha256(json.dumps(test_records(manifest), sort_keys=True).encode()).hexdigest()


def check_frozen(manifest):
    if not FROZEN or RECORDS is None:
        raise SystemExit("not frozen")
    for rel, want in FROZEN.items():
        got = sha(HERMES.parent / rel)
        if got != want:
            raise SystemExit(f"{rel} changed since the freeze ({got[:12]} != {want[:12]}); refusing to score")
    if records_sha(manifest) != RECORDS:
        raise SystemExit("the Robie-Hemingway records changed since the freeze; refusing to score")


def measured(manifest, tag):
    """Robie-Hemingway S per mole of atoms at each scored temperature, selected by record id."""
    formula, apf, _, temps = CASES[tag]
    by_id = {a["id"]: a for a in manifest["artifacts"]}
    values = {}
    for temperature in temps:
        label = "S298" if temperature == T0 else "S1000"
        record = by_id[f"{RECORD_PREFIX}{label}_{formula}_{tag}"]
        if abs(record["conditions"]["temperature_K"] - temperature) > 0.01 or record["unit"] != "J/mol/K":
            raise SystemExit(f"{record['id']} is not S at {temperature} K in J/mol/K")
        values[temperature] = record["value"] / apf
    return values


def predict(folder, apf):
    ladder = json.loads((folder / "ladder.json").read_text())
    t0, t1 = np.load(folder / "nvt_t0.npz"), np.load(folder / "nvt_t1.npz")
    if abs(ladder[0]["T"] - T0) > 1e-6:
        raise SystemExit(f"{folder}: first ladder temperature is not {T0} K")
    out = {"atoms": int(len(t0["masses"])),
           "space_group": {T0: space_group(t0["positions"], t0["cell"], t0["numbers"]),
                           T1: space_group(t1["positions"], t1["cell"], t1["numbers"])},
           "V_md_cm3_mol": abs(np.linalg.det(np.asarray(ladder[0]["mean_cell"]))) / len(t0["masses"]) * apf * NA_CM3}
    temperatures = np.array([row["T"] for row in ladder])
    enthalpies = np.array([row["pe_eV_atom"] + 1.5 * KB * row["T"] + row["pv_eV_atom"] for row in ladder]) * EV
    qc = {}
    for run, temperature in ((t0, T0), (t1, T1)):
        nu, dos = vibrational_density_of_states(run["velocities"], float(run["velocity_dt_ps"]), run["masses"],
                                                segment_ps=5.0)
        qc[temperature] = quantum_correction(nu, dos, temperature)
        if temperature == T0:
            out["harmonic_quantum_T0"] = harmonic_entropy(nu, dos, T0, quantum=True)
    out["QC"] = qc
    for limit in ("bulk", "box"):
        try:
            anchor = gaussian_entropy(t0["positions"], t0["cell"], t0["masses"], T0, species=t0["numbers"],
                                      limit=limit)["entropy"]
        except ValueError as err:
            out["refused"] = str(err)
            return out
        carried = entropy_from_heat_capacity(temperatures, enthalpies, T0, anchor, at=[T0, T1])
        out[f"S_vib_{limit}"] = {T0: anchor + qc[T0], T1: float(carried["entropy"][1]) + qc[T1]}
    out["H_fit_residual_J_mol"] = carried["residual"]
    return out


def judge(predictions, measurements):
    """Validity, C1, C2 and the verdict from per-case predictions and measured values, both keyed by temperature."""
    results, excluded = {}, []
    for tag, (_, _, number, temps) in CASES.items():
        p, m = predictions[tag], measurements[tag]
        row = {"prediction": p, "measured": m}
        reason = p.get("refused") or next((f"space group at {t:g} K is {p['space_group'][t][1]} "
                                           f"({p['space_group'][t][0]}), not {number}"
                                           for t in temps if p["space_group"][t][0] != number), None)
        if reason:
            row["status"], row["reason"] = "excluded", reason
            excluded.append(tag)
        else:
            row["status"] = "scored"
            row["error"] = {t: p["S_vib_bulk"][t] - m[t] for t in temps}
            row["residual"] = {t: row["error"][t] - OFFSET[t] for t in temps}
            row["error_box"] = {t: p["S_vib_box"][t] - m[t] for t in temps}
        results[tag] = row

    c1_by_t = {}
    for t in (T0, T1):
        r = np.array([results[tag]["residual"][t] for tag in CASES if results[tag]["status"] == "scored"
                      and t in CASES[tag][3]])
        rms, worst = float(np.sqrt(np.mean(r ** 2))), float(np.max(np.abs(r)))
        c1_by_t[t] = {"n": len(r), "rms": rms, "max_abs": worst,
                      "holds": bool(rms <= RMS_TOLERANCE and worst <= MAX_TOLERANCE)}
    c1 = all(v["holds"] for v in c1_by_t.values())

    families = {}
    for name, (pairs, temps) in FAMILIES.items():
        members = {x for pair in pairs for x in pair}
        if any(results[x]["status"] != "scored" for x in members):
            families[name] = {"applied": False}
            continue
        comparisons = {}
        for a, b in pairs:
            for t in temps:
                pred = predictions[a]["S_vib_bulk"][t] - predictions[b]["S_vib_bulk"][t]
                meas = measurements[a][t] - measurements[b][t]
                comparisons[f"{a} - {b} @ {t:g} K"] = {"predicted": pred, "measured": meas, "error": pred - meas}
        families[name] = {"applied": True, "comparisons": comparisons,
                          "holds": all(abs(c["error"]) <= POLYMORPH_TOLERANCE for c in comparisons.values())}
    applied = [f for f in families.values() if f["applied"]]
    c2 = all(f["holds"] for f in applied)
    void = len(excluded) > MAX_EXCLUDED or len(applied) < MIN_FAMILIES
    verdict = "VOID (FAIL)" if void else ("PASS" if c1 and c2 else "FAIL")
    return {"verdict": verdict, "void": void, "C1": c1, "C1_by_T": c1_by_t, "C2": c2, "families": families,
            "excluded": excluded, "cases": results}


def report(out, volumes):
    f = lambda x, w=7: f"{x:{w}.2f}" if isinstance(x, (int, float)) else f"{'-':>{w}s}"  # noqa: E731
    cases = out["cases"]
    lines = [f"Polymorph test, MACE MPA-0, Hermes 0.3 (limit='bulk'). Verdict: {out['verdict']}",
             f"C1 {'holds' if out['C1'] else 'fails'}; C2 {'holds' if out['C2'] else 'fails'}; "
             f"scored {len(cases) - len(out['excluded'])}, excluded {len(out['excluded'])}. "
             "J/(K mol of atoms); err = predicted - measured; resid = err - D(T), D = +1.87 / +2.20.",
             f"{'case':19s} {'status':8s} {'S 298':>7s} {'meas':>7s} {'err':>6s} {'resid':>6s} {'box err':>7s} "
             f"{'S 1000':>7s} {'meas':>7s} {'err':>6s} {'resid':>6s} {'box err':>7s} {'QC298':>6s} {'QC1000':>6s} "
             f"{'harmQ298':>8s} {'H resid':>7s} {'V md':>6s} {'V R&H':>6s}"]
    for tag, r in cases.items():
        p, m = r["prediction"], r["measured"]
        s, e = p.get("S_vib_bulk", {}), r.get("error", {})
        res, eb, qc = r.get("residual", {}), r.get("error_box", {}), p.get("QC", {})
        cols = []
        for t in (T0, T1):
            cols += [f(s.get(t) if t in m else None), f(m.get(t)), f(e.get(t), 6), f(res.get(t), 6), f(eb.get(t))]
        lines.append(f"{tag:19s} {r['status']:8s} " + " ".join(cols) + f" {f(qc.get(T0), 6)} {f(qc.get(T1), 6)} "
                     f"{f(p.get('harmonic_quantum_T0'), 8)} {f(p.get('H_fit_residual_J_mol'))} "
                     f"{f(p.get('V_md_cm3_mol'), 6)} {f(volumes[tag]['V_cm3_mol'], 6)}"
                     + (f"  [{r['reason']}]" if r.get("reason") else ""))
    for t, c in out["C1_by_T"].items():
        errors = [cases[tag]["error"][t] for tag in cases if cases[tag]["status"] == "scored" and t in cases[tag]["error"]]
        lines.append(f"C1 at {t:g} K: {c['n']} cases, RMS residual {c['rms']:.2f}, largest |residual| "
                     f"{c['max_abs']:.2f}, {'holds' if c['holds'] else 'fails'}; uncorrected errors mean "
                     f"{np.mean(errors):+.2f} against D {OFFSET[t]:+.2f}, {sum(x > 0 for x in errors)} of {len(errors)} positive")
    for name, fam in out["families"].items():
        if not fam["applied"]:
            lines.append(f"C2 {name}: not applied (a member is excluded)")
            continue
        for label, c in fam["comparisons"].items():
            lines.append(f"C2 {name:7s} {label:38s} predicted {c['predicted']:+.2f}, R&H {c['measured']:+.2f}, "
                         f"error {c['error']:+.2f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_frozen(manifest)
    runs = args.results / MODEL / "runs"
    unfinished = [t for t in CASES if "CASE_DONE" not in ((runs / t / "progress.log").read_text()
                                                          if (runs / t / "progress.log").exists() else "")]
    if unfinished:
        raise SystemExit(f"not finished: {', '.join(unfinished)}; the rule scores all cases together")
    predictions = {tag: predict(runs / tag, CASES[tag][1]) for tag in CASES}
    measurements = {tag: measured(manifest, tag) for tag in CASES}
    out = judge(predictions, measurements)
    text = report(out, json.loads((HERE / "volumes_rh1995.json").read_text()))
    print(text)
    (HERE / "polymorph_output.txt").write_text(text + "\n", encoding="utf-8")
    (HERE / "polymorph_results.json").write_text(json.dumps(
        out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")


if __name__ == "__main__":
    main()
