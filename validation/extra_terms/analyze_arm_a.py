"""Score Arm A (magnetic oxides) of RULE.md, frozen 2026-10-22, exactly as written there.

S_pred(T) = S_vib(T) + S_mag(T) + S_el(T), J/(K mol of atoms), at 298.15 K and 1000 K:
S_vib by the frozen oxide vibration route (../oxide_vibration/analyze_oxides.predict, MD from run_oxides.py unchanged),
S_mag = magnetic_properties(T, T_N, 2S, 0.28, reference="ordered"), S_el = electronic_entropy of the Materials Project
GGA+U density of states (dos/<tag>.npz). Measured values by the same selection as the oxide test. Writes
arm_a_output.txt and arm_a_results.json. Refuses to run if the rule or any code under test has changed.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HERMES = HERE.parents[1]
spec = importlib.util.spec_from_file_location("analyze_oxides", HERE.parent / "oxide_vibration" / "analyze_oxides.py")
vib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vib)
from hermes.electronic import electronic_entropy  # noqa: E402  (path set by analyze_oxides)
from hermes.magnetic import magnetic_properties, spin_only_moment  # noqa: E402

FROZEN = {
    "validation/extra_terms/RULE.md": "9e75e71012c07987601ae0b01cf5fe2e0d8e2c8f78f3f6e7cb462e96dc118db9",
    "hermes/crystal_field.py": "0c4694a7425256a9a253ef16f281e74dfbde2667e0fee429b7bd435bb4428f96",
    "hermes/magnetic.py": "9fbe13e7227a9568208e52c692432fc0098ef9d771972204567134e34a69ee28",
    "hermes/electronic.py": "3b9ebc37f87a4f83025bd48b4e1d24d9c17b40f41ae22573a9dfa242e6883e85",
}
TOLERANCE = 1.0
P = 0.28
TEMPERATURES = (vib.T0, vib.T1)
CASES = {  # tag: (composition, spin S, T_N, magnetic ions, atoms per formula, phase words that exclude a record)
    "nio": ("NiO", 1.0, 523.0, 1, 2, ()),
    "mno": ("MnO", 2.5, 119.0, 1, 2, ()),
    "coo": ("CoO", 1.5, 289.0, 1, 2, ()),
    "fe2o3": ("Fe2O3", 2.5, 950.0, 2, 5, ("maghemite", "gamma")),
    "cr2o3": ("Cr2O3", 1.5, 308.5, 2, 5, ()),
}
PRIMARY = ("nio", "mno", "fe2o3", "cr2o3")


def check_frozen():
    vib.check_frozen()
    for rel, want in FROZEN.items():
        got = hashlib.sha256((HERMES / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            raise SystemExit(f"{rel} changed since the freeze ({got[:12]} != {want[:12]}); refusing to score")


def electronic(tag):
    d = np.load(HERE / "dos" / f"{tag}.npz")
    return [electronic_entropy(d["energies"], np.vstack([d["up"], d["down"]]), t, atoms=int(d["atoms"]),
                               fermi_energy=float(d["efermi"])) for t in TEMPERATURES]


def main():
    check_frozen()
    manifest = json.loads(vib.MANIFEST.read_text(encoding="utf-8"))
    results = {}
    for tag, (comp, spin, t_n, ions, n_atoms, exclude) in CASES.items():
        folder = HERE.parent / "oxide_vibration" / "runs" / tag
        if not (folder / "nvt_t1.npz").exists():
            raise SystemExit(f"{tag} has not finished")
        p = vib.predict(folder)
        m = vib.measured(manifest, comp, exclude, n_atoms)
        beta = 2 * spin
        assert abs(beta - spin_only_moment(spin)) < 1e-12
        s_mag = [magnetic_properties(t, t_n, beta, P, reference="ordered", ions_per_formula=ions,
                                     atoms_per_formula=n_atoms)["entropy"] for t in TEMPERATURES]
        s_el = electronic(tag)
        row = {"prediction": p, "measured": m, "S_mag": s_mag, "S_el": s_el}
        if "refused" not in p:
            row["S_pred"] = [p["S_vib_T0"] + s_mag[0] + s_el[0], p["S_vib_T1"] + s_mag[1] + s_el[1]]
            row["error_298"] = row["S_pred"][0] - m["S298"] if m["S298"] is not None else None
            row["error_1000"] = row["S_pred"][1] - m["S1000"] if m.get("S1000") is not None else None
            if row["error_298"] is not None:
                row["vib_only_error_298"] = p["S_vib_T0"] + s_el[0] - m["S298"]
                row["A2_magnetic_term_helps"] = abs(row["error_298"]) < abs(row["vib_only_error_298"])
        results[tag] = row

    def a1_ok(tag):
        r = results[tag]
        return (r.get("error_298") is not None and abs(r["error_298"]) <= TOLERANCE
                and (r["error_1000"] is None or abs(r["error_1000"]) <= TOLERANCE))

    a1 = all(a1_ok(t) for t in PRIMARY)
    coo = results["coo"]
    a3 = (None if coo.get("error_1000") is None else coo["error_1000"] < -TOLERANCE)
    vib_test = (HERE.parent / "oxide_vibration" / "oxide_results.json")
    vib_verdict = json.loads(vib_test.read_text())["verdict"] if vib_test.exists() else "not yet scored"

    lines = [f"Extra-terms test, Arm A (magnetic oxides), rule frozen 2026-10-22. A1: {'PASS' if a1 else 'FAIL'}",
             f"Oxide vibration test verdict: {vib_verdict}" + ("" if vib_verdict == "PASS" else
                                                              "; Arm A is limited by the vibrational route"),
             "Overall verdict needs Arm B (B1); not run.",
             f"{'case':6s} {'S_vib':>7s} {'S_mag':>6s} {'S_el':>6s} {'pred':>7s} {'meas':>7s} {'err':>6s}   "
             f"{'S_vib':>7s} {'S_mag':>6s} {'S_el':>6s} {'pred':>7s} {'meas':>7s} {'err':>6s}  A2   (298.15 K | 1000 K)"]
    f = lambda x, w=7: f"{x:{w}.2f}" if isinstance(x, (float, np.floating)) else f"{'-':>{w}s}"  # noqa: E731
    for tag, r in results.items():
        p, m = r["prediction"], r["measured"]
        pred = r.get("S_pred", [None, None])
        lines.append(f"{tag:6s} {f(p.get('S_vib_T0'))} {f(r['S_mag'][0], 6)} {f(r['S_el'][0], 6)} {f(pred[0])} "
                     f"{f(m.get('S298'))} {f(r.get('error_298'), 6)}   {f(p.get('S_vib_T1'))} {f(r['S_mag'][1], 6)} "
                     f"{f(r['S_el'][1], 6)} {f(pred[1])} {f(m.get('S1000'))} {f(r.get('error_1000'), 6)}  "
                     f"{r.get('A2_magnetic_term_helps', '-')}" + (f"  [refused: {p['refused']}]" if 'refused' in p else ""))
    lines.append(f"A3 (CoO short by more than 1.0 at 1000 K, predicted in advance): "
                 + ("no data at 1000 K" if a3 is None else ("prediction held" if a3 else "prediction FAILED")))
    text = "\n".join(lines)
    print(text)
    (HERE / "arm_a_output.txt").write_text(text + "\n")
    (HERE / "arm_a_results.json").write_text(json.dumps({"A1": a1, "A3": a3, "cases": results}, indent=1,
                                                        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
