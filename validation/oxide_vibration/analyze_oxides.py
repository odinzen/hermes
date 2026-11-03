"""Score the closed-shell oxide vibration test exactly as RULE.md (frozen 2026-10-23) sets it out.

    python analyze_oxides.py            the verdict, written to oxide_analysis_output.txt and oxide_results.json
    python analyze_oxides.py --smoke    CaO smoke run, prediction only, never compared with a measurement

Refuses to run if the code under test, the rule or the potential's record in PREREGISTERED.txt has changed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import spglib

HERE = Path(__file__).resolve().parent
HERMES = HERE.parents[1]
sys.path.insert(0, str(HERMES))
from hermes.compression_entropy import gaussian_entropy  # noqa: E402
from hermes.heat_capacity import entropy_from_heat_capacity  # noqa: E402
from hermes.quantum_correction import harmonic_entropy, quantum_correction, vibrational_density_of_states  # noqa: E402

MANIFEST = Path.home() / "Code/juno/assessments/Refractory-Oxides/manifest.json"
FROZEN = {
    "validation/oxide_vibration/RULE.md": "440185cb3f00b592e5f4358107de4f2e1565d6ea7ba897facdc5d24e03e993f7",
    "hermes/compression_entropy.py": "58e95b9206d6d287ce4ca1ad5faf18fd7d6a671d27f34a48edb1430643552fb6",
    "hermes/heat_capacity.py": "541d7a3483bf84adc8949d6a492450e954284d4ff050bf7b6730e11838552b80",
    "hermes/quantum_correction.py": "8d968a9b7c40ef36cbed0a64b6fe25339abb39de40d9b8deeb9db510928bbdf8",
}
EV = 96485.33212  # J/mol per eV/atom
KB = 8.617333262e-5
T0, T1 = 298.15, 1000.0
TOLERANCE = 1.0
SYMPREC = 0.1

CASES = {  # tag: (composition in the manifest, atoms per formula, space-group number, phase words that exclude a record)
    "mgo": ("MgO", 2, 225, ()),
    "al2o3": ("Al2O3", 5, 167, ("gamma", "delta", "kappa", "theta")),
    "tio2": ("TiO2", 3, 136, ("anatase", "brookite")),
    "y2o3": ("Y2O3", 5, 206, ()),
    "sc2o3": ("Sc2O3", 5, 206, ()),
    "lu2o3": ("Lu2O3", 5, 206, ()),
    "la2o3": ("La2O3", 5, 164, ("c-type", "cubic", "b-type", "monoclinic")),
    "srtio3": ("SrTiO3", 5, 221, ("tetragonal",)),
    "mgtio3": ("MgTiO3", 5, 148, ()),
    "mg3po42": ("Mg3(PO4)2", 13, 14, ()),
}


def check_frozen():
    for rel, want in FROZEN.items():
        got = hashlib.sha256((HERMES / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            raise SystemExit(f"{rel} changed since the freeze ({got[:12]} != {want[:12]}); refusing to score")


def mean_structure(positions, cell):
    frac = np.linalg.solve(cell.T, positions.reshape(-1, 3).T).T.reshape(positions.shape)
    step = frac - frac[0]
    step -= np.round(step)
    return np.mod((frac[0] + step).mean(axis=0), 1.0)


def space_group(positions, cell, numbers):
    dataset = spglib.get_symmetry_dataset((cell, mean_structure(positions, cell), numbers), symprec=SYMPREC)
    return (dataset.number, dataset.international) if dataset is not None else (0, "none")


def predict(folder, segment_ps=5.0):
    ladder = json.loads((folder / "ladder.json").read_text())
    t0 = np.load(folder / "nvt_t0.npz")
    t1 = np.load(folder / "nvt_t1.npz")
    out = {"atoms": int(len(t0["masses"]))}
    out["space_group_T0"] = space_group(t0["positions"], t0["cell"], t0["numbers"])
    out["space_group_T1"] = space_group(t1["positions"], t1["cell"], t1["numbers"])
    try:
        gauss = gaussian_entropy(t0["positions"], t0["cell"], t0["masses"], T0, species=t0["numbers"])
    except ValueError as err:
        out["refused"] = str(err)
        return out
    temperatures = np.array([row["T"] for row in ladder])
    enthalpies = np.array([row["pe_eV_atom"] + 1.5 * KB * row["T"] + row["pv_eV_atom"] for row in ladder]) * EV
    carried = entropy_from_heat_capacity(temperatures, enthalpies, T0, gauss["entropy"], at=[T0, T1])
    qc = {}
    for name, run, temperature in (("T0", t0, T0), ("T1", t1, T1)):
        nu, dos = vibrational_density_of_states(run["velocities"], float(run["velocity_dt_ps"]), run["masses"],
                                                segment_ps=segment_ps)
        qc[name] = quantum_correction(nu, dos, temperature)
        if name == "T0":
            out["harmonic_quantum_T0"] = harmonic_entropy(nu, dos, T0, quantum=True)
    out.update({
        "S_cl_T0": gauss["entropy"], "marginal_T0": gauss["marginal"], "correlation_T0": gauss["correlation"],
        "S_cl_T1": float(carried["entropy"][1]), "H_fit_residual_J_mol": carried["residual"],
        "QC_T0": qc["T0"], "QC_T1": qc["T1"],
        "S_vib_T0": gauss["entropy"] + qc["T0"], "S_vib_T1": float(carried["entropy"][1]) + qc["T1"],
    })
    return out


def measured(manifest, composition, exclude, atoms_per_formula):
    """The selection RULE.md fixes, per mole of atoms; None where there is no data."""
    def phase_ok(record):
        return not any(word in str(record.get("phase", "")).lower() for word in exclude)

    entropy = [a for a in manifest["artifacts"] if a["quantity"] == "standard_entropy"
               and (a.get("conditions") or {}).get("composition") == composition
               and a.get("evidence_tier") == "evaluated_compilation" and phase_ok(a)]
    at_298 = [a for a in entropy if abs((a["conditions"].get("temperature_K") or 0) - T0) < 0.01]
    out = {"S298_sources": {a["id"]: a["value"] for a in at_298}}
    if not at_298:
        return out | {"S298": None, "S1000": None}
    values = np.array([a["value"] for a in at_298]) / atoms_per_formula
    out |= {"S298": float(values.mean()), "S298_spread": float(np.ptp(values))}
    janaf = [a for a in entropy if a["paper"] == "paper:nist_janaf" and a["conditions"].get("temperature_K") == T1]
    if janaf:
        return out | {"S1000": janaf[0]["value"] / atoms_per_formula, "S1000_source": janaf[0]["id"]}
    models = [a for a in manifest["artifacts"] if a["quantity"] == "heat_capacity_model"
              and a["paper"] == "paper:robie_hemingway_1995"
              and (a.get("conditions") or {}).get("composition") == composition and phase_ok(a)
              and a["model"]["valid_range_K"][0] <= 298.15 + 1 and a["model"]["valid_range_K"][1] >= T1]
    if not models:
        return out | {"S1000": None}
    # Robie and Hemingway leave an absent term blank; the manifest stores the blank as null.
    c = {k: v if v is not None else 0.0 for k, v in models[0]["model"]["coefficients"].items()}
    t = np.linspace(T0, T1, 20001)
    cp = c["a"] + c["b"] * t + c["c"] * t**-2 + c["d"] * t**-0.5 + c["e"] * t**2
    s1000 = float(values.mean() * atoms_per_formula + np.trapezoid(cp / t, t)) / atoms_per_formula
    return out | {"S1000": s1000, "S1000_source": models[0]["id"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    check_frozen()
    if args.smoke:
        # the shortened smoke runs hold 1 to 2 ps of velocities, too short for the 5 ps window
        print(json.dumps(predict(HERE / "smoke" / "cao", segment_ps=0.5), indent=1, default=str))
        return
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results, lines, excluded, scored = {}, [], [], []
    for tag, (composition, n_atoms, number, exclude) in CASES.items():
        folder = HERE / "runs" / tag
        m = measured(manifest, composition, exclude, n_atoms)
        done = (folder / "nvt_t1.npz").exists()
        if m["S298"] is not None and not done:
            raise SystemExit(f"{tag} has data but its runs have not finished; the rule scores all cases together")
        # a case without data is reported, and needs no runs
        p = predict(folder) if done else {"space_group_T0": (0, "not run"), "space_group_T1": (0, "not run")}
        row = {"prediction": p, "measured": m}
        reason = p.get("refused") or next((f"space group at {name} is {p[key][1]} ({p[key][0]}), not {number}"
                                           for name, key in (("T0", "space_group_T0"), ("T1", "space_group_T1"))
                                           if p[key][0] != number), None)
        if m["S298"] is None:
            row["status"] = "no data"
        elif reason:
            row["status"], row["reason"] = "excluded", reason
            excluded.append(tag)
        else:
            row["status"] = "scored"
            scored.append(tag)
            row["error_T0"] = p["S_vib_T0"] - m["S298"]
            row["error_T1"] = p["S_vib_T1"] - m["S1000"] if m["S1000"] is not None else None
            row["pass"] = abs(row["error_T0"]) <= TOLERANCE and (row["error_T1"] is None
                                                                 or abs(row["error_T1"]) <= TOLERANCE)
        results[tag] = row
    void = len(excluded) > 2 or len(scored) < 6
    passed = not void and all(results[t]["pass"] for t in scored)
    verdict = "VOID (FAIL)" if void else ("PASS" if passed else "FAIL")

    lines.append(f"Closed-shell oxide vibration test, rule frozen 2026-10-23. Verdict: {verdict}")
    lines.append(f"scored {len(scored)}, excluded {len(excluded)}, no data "
                 f"{sum(r['status'] == 'no data' for r in results.values())}; J/(K mol of atoms)")
    lines.append(f"{'case':9s} {'status':8s} {'S_vib 298':>9s} {'meas':>7s} {'err':>6s} {'S_vib 1000':>10s} "
                 f"{'meas':>7s} {'err':>6s} {'QC298':>6s} {'QC1000':>6s} {'harmQ298':>8s} {'spread':>6s}")
    for tag, r in results.items():
        p, m = r["prediction"], r["measured"]
        f = lambda x, w=7: f"{x:{w}.2f}" if isinstance(x, float) else f"{'-':>{w}s}"  # noqa: E731
        lines.append(f"{tag:9s} {r['status']:8s} {f(p.get('S_vib_T0'), 9)} {f(m.get('S298'))} {f(r.get('error_T0'), 6)} "
                     f"{f(p.get('S_vib_T1'), 10)} {f(m.get('S1000'))} {f(r.get('error_T1'), 6)} "
                     f"{f(p.get('QC_T0'), 6)} {f(p.get('QC_T1'), 6)} {f(p.get('harmonic_quantum_T0'), 8)} "
                     f"{f(m.get('S298_spread'), 6)}" + (f"  [{r['reason']}]" if r.get("reason") else ""))
    errors = [results[t]["error_T0"] for t in scored]
    if errors:
        lines.append(f"298.15 K errors: mean {np.mean(errors):+.2f}, range {min(errors):+.2f} to {max(errors):+.2f}")
    text = "\n".join(lines)
    print(text)
    (HERE / "oxide_analysis_output.txt").write_text(text + "\n")
    (HERE / "oxide_results.json").write_text(json.dumps({"verdict": verdict, "cases": results}, indent=1,
                                                        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
