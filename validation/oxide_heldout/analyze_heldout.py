"""Score the held-out closed-shell oxide test exactly as RULE.md sets it out.

    python analyze_heldout.py RESULTS     RESULTS holds mpa-0/runs/<case> as the Sol package writes it
                                          -> heldout_output.txt, heldout_results.json beside this file

Refuses to run if the rule, the code under test or the measured data have changed since the freeze, or before every
case has finished: the rule scores all cases together.
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

MANIFEST = HERMES.parent / "juno/assessments/Refractory-Oxides/manifest.json"
FROZEN = {  # path under Code/: SHA-256 with LF endings
    "hermes-private/validation/oxide_heldout/RULE.md": "6729c45e261a8d3e462462cc49735302429cc43dbc214ec4eab7f47754be2d47",
    "hermes-private/hermes/compression_entropy.py": "512e1cf35bdd0b0e5728fe8c07f8dd1f707e5891e0f793a74a4e25176a246c77",
    "hermes-private/hermes/heat_capacity.py": "541d7a3483bf84adc8949d6a492450e954284d4ff050bf7b6730e11838552b80",
    "hermes-private/hermes/quantum_correction.py": "8d968a9b7c40ef36cbed0a64b6fe25339abb39de40d9b8deeb9db510928bbdf8",
}
RECORDS = "7261461de99ee6c244acf7ff395cec75760123cf685c869a2c41ed21bee3adc5"  # SHA-256 of the 22 held-out JANAF records
EV = 96485.33212  # J/mol per eV/atom
KB = 8.617333262e-5
T0, T1 = 298.15, 1000.0
TOLERANCE, POLYMORPH_TOLERANCE = 1.0, 0.5
SYMPREC = 0.1
MODEL = "mpa-0"

CASES = {  # tag: (JANAF record suffix, atoms per formula unit, space-group number)
    "beo": ("BeO_bromellite", 2, 186),
    "cao": ("CaO_lime", 2, 225),
    "sro": ("SrO_rock_salt", 2, 225),
    "bao": ("BaO_rock_salt", 2, 225),
    "li2o": ("Li2O_antifluorite", 3, 225),
    "zro2": ("ZrO2_baddeleyite", 3, 14),
    "mg2sio4": ("Mg2SiO4_forsterite", 7, 62),
    "zrsio4": ("ZrSiO4_zircon", 6, 141),
    "andalusite": ("Al2SiO5_andalusite", 8, 58),
    "kyanite": ("Al2SiO5_kyanite", 8, 2),
    "sillimanite": ("Al2SiO5_sillimanite", 8, 62),
}
PAIRS = (("kyanite", "andalusite"), ("sillimanite", "andalusite"), ("sillimanite", "kyanite"))


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def heldout_records(manifest):
    """The held-out records alone, so other additions to the manifest do not touch the freeze."""
    return sorted((a for a in manifest["artifacts"] if str(a.get("id", "")).startswith("meas:nist_janaf:heldout_")),
                  key=lambda a: a["id"])


def records_sha(manifest):
    return hashlib.sha256(json.dumps(heldout_records(manifest), sort_keys=True).encode()).hexdigest()


def check_frozen(manifest):
    for rel, want in FROZEN.items():
        got = sha(HERMES.parent / rel)
        if got != want:
            raise SystemExit(f"{rel} changed since the freeze ({got[:12]} != {want[:12]}); refusing to score")
    if records_sha(manifest) != RECORDS:
        raise SystemExit("the held-out JANAF records changed since the freeze; refusing to score")


def mean_structure(positions, cell):
    frac = np.linalg.solve(cell.T, positions.reshape(-1, 3).T).T.reshape(positions.shape)
    step = frac - frac[0]
    step -= np.round(step)
    return np.mod((frac[0] + step).mean(axis=0), 1.0)


def space_group(positions, cell, numbers):
    dataset = spglib.get_symmetry_dataset((cell, mean_structure(positions, cell), numbers), symprec=SYMPREC)
    return (dataset.number, dataset.international) if dataset is not None else (0, "none")


def predict(folder):
    ladder = json.loads((folder / "ladder.json").read_text())
    t0, t1 = np.load(folder / "nvt_t0.npz"), np.load(folder / "nvt_t1.npz")
    out = {"atoms": int(len(t0["masses"])),
           "space_group_T0": space_group(t0["positions"], t0["cell"], t0["numbers"]),
           "space_group_T1": space_group(t1["positions"], t1["cell"], t1["numbers"])}
    temperatures = np.array([row["T"] for row in ladder])
    enthalpies = np.array([row["pe_eV_atom"] + 1.5 * KB * row["T"] + row["pv_eV_atom"] for row in ladder]) * EV
    qc = []
    for run, temperature in ((t0, T0), (t1, T1)):
        nu, dos = vibrational_density_of_states(run["velocities"], float(run["velocity_dt_ps"]), run["masses"],
                                                segment_ps=5.0)
        qc.append(quantum_correction(nu, dos, temperature))
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
        out[f"S_vib_{limit}"] = [anchor + qc[0], float(carried["entropy"][1]) + qc[1]]
    out["H_fit_residual_J_mol"] = carried["residual"]  # the H(T) fit does not depend on the anchor
    return out


def measured(manifest, suffix, atoms_per_formula):
    """JANAF S at 298.15 and 1000 K per mole of atoms, selected by record id."""
    by_id = {a["id"]: a for a in manifest["artifacts"]}
    values = []
    for label, temperature in (("S298", T0), ("S1000", T1)):
        record = by_id[f"meas:nist_janaf:heldout_{label}_{suffix}"]
        if abs(record["conditions"]["temperature_K"] - temperature) > 0.01 or record["unit"] != "J/mol/K":
            raise SystemExit(f"{record['id']} is not S at {temperature} K in J/mol/K")
        values.append(record["value"] / atoms_per_formula)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    if not FROZEN:
        raise SystemExit("not frozen")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    check_frozen(manifest)
    runs = args.results / MODEL / "runs"
    unfinished = [t for t in CASES if "CASE_DONE" not in ((runs / t / "progress.log").read_text()
                                                          if (runs / t / "progress.log").exists() else "")]
    if unfinished:
        raise SystemExit(f"not finished: {', '.join(unfinished)}; the rule scores all cases together")
    results, scored, excluded = {}, [], []
    for tag, (suffix, apf, number) in CASES.items():
        p = predict(runs / tag)
        m = measured(manifest, suffix, apf)
        row = {"prediction": p, "measured": m}
        reason = p.get("refused") or next((f"space group at {name} is {p[key][1]} ({p[key][0]}), not {number}"
                                           for name, key in (("T0", "space_group_T0"), ("T1", "space_group_T1"))
                                           if p[key][0] != number), None)
        if reason:
            row["status"], row["reason"] = "excluded", reason
            excluded.append(tag)
        else:
            row["status"] = "scored"
            scored.append(tag)
            row["error"] = [p["S_vib_bulk"][i] - m[i] for i in (0, 1)]
            row["error_box"] = [p["S_vib_box"][i] - m[i] for i in (0, 1)]
            row["pass"] = all(abs(e) <= TOLERANCE for e in row["error"])
        results[tag] = row

    void = len(excluded) > 2
    c1 = all(results[t]["pass"] for t in scored)
    polymorphs = {}
    if all(results[t]["status"] == "scored" for t in ("andalusite", "kyanite", "sillimanite")):
        for a, b in PAIRS:
            pred = [results[a]["prediction"]["S_vib_bulk"][i] - results[b]["prediction"]["S_vib_bulk"][i] for i in (0, 1)]
            meas = [results[a]["measured"][i] - results[b]["measured"][i] for i in (0, 1)]
            polymorphs[f"{a} - {b}"] = {"predicted": pred, "measured": meas, "error": [pred[i] - meas[i] for i in (0, 1)]}
        c2 = all(abs(e) <= POLYMORPH_TOLERANCE for d in polymorphs.values() for e in d["error"])
    else:
        c2 = None
    passed = not void and c1 and c2 is not False
    verdict = "VOID (FAIL)" if void else ("PASS" if passed else "FAIL")

    f = lambda x, w=7: f"{x:{w}.2f}" if isinstance(x, (int, float)) else f"{'-':>{w}s}"  # noqa: E731
    lines = [f"Held-out closed-shell oxide test, MACE MPA-0, Hermes 0.3 (limit='bulk'). Verdict: {verdict}",
             f"C1 {'holds' if c1 else 'fails'}; C2 " + ("not applied" if c2 is None else ("holds" if c2 else "fails"))
             + f"; scored {len(scored)}, excluded {len(excluded)}. J/(K mol of atoms), err = predicted - measured.",
             f"{'case':12s} {'status':8s} {'S 298':>7s} {'meas':>7s} {'err':>6s} {'box err':>7s} {'S 1000':>7s} "
             f"{'meas':>7s} {'err':>6s} {'box err':>7s} {'QC298':>6s} {'QC1000':>6s} {'harmQ298':>8s} {'H resid':>7s}"]
    for tag, r in results.items():
        p, m = r["prediction"], r["measured"]
        s, e, eb = p.get("S_vib_bulk", [None, None]), r.get("error", [None, None]), r.get("error_box", [None, None])
        qc = p.get("QC", [None, None])
        lines.append(f"{tag:12s} {r['status']:8s} {f(s[0])} {f(m[0])} {f(e[0], 6)} {f(eb[0])} {f(s[1])} {f(m[1])} "
                     f"{f(e[1], 6)} {f(eb[1])} {f(qc[0], 6)} {f(qc[1], 6)} {f(p.get('harmonic_quantum_T0'), 8)} "
                     f"{f(p.get('H_fit_residual_J_mol'))}" + (f"  [{r['reason']}]" if r.get("reason") else ""))
    for name, d in polymorphs.items():
        lines.append(f"C2 {name:26s} predicted {d['predicted'][0]:+.2f} / {d['predicted'][1]:+.2f}, JANAF "
                     f"{d['measured'][0]:+.2f} / {d['measured'][1]:+.2f}, error {d['error'][0]:+.2f} / {d['error'][1]:+.2f}"
                     "  (298.15 / 1000 K)")
    for i, temperature in enumerate((T0, T1)):
        errors = [results[t]["error"][i] for t in scored]
        if errors:
            lines.append(f"{temperature:g} K errors: mean {np.mean(errors):+.2f}, range {min(errors):+.2f} to "
                         f"{max(errors):+.2f}, {sum(e > 0 for e in errors)} of {len(errors)} positive")
    text = "\n".join(lines)
    print(text)
    (HERE / "heldout_output.txt").write_text(text + "\n", encoding="utf-8")
    (HERE / "heldout_results.json").write_text(json.dumps(
        {"verdict": verdict, "C1": c1, "C2": c2, "polymorphs": polymorphs, "cases": results}, indent=1,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")


if __name__ == "__main__":
    main()
