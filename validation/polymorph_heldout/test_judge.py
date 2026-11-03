"""Planted cases for the polymorph scorer: each must give the verdict the rule says. Synthetic numbers only.

    python test_judge.py
"""
import copy

from analyze_polymorph import CASES, OFFSET, T0, T1, judge

BASE = 20.0


def synthetic(residual=0.0):
    predictions, measurements = {}, {}
    for i, (tag, (_, _, number, temps)) in enumerate(CASES.items()):
        m = {t: BASE + i + (30.0 if t == T1 else 0.0) for t in temps}
        s = {t: m.get(t, BASE + i + 30.0) + OFFSET[t] + residual for t in (T0, T1)}
        predictions[tag] = {"space_group": {T0: (number, "x"), T1: (number, "x")}, "S_vib_bulk": s,
                            "S_vib_box": dict(s)}
        measurements[tag] = m
    return predictions, measurements


def check(name, predictions, measurements, verdict, c1=None, c2=None):
    out = judge(predictions, measurements)
    ok = out["verdict"] == verdict and (c1 is None or out["C1"] == c1) and (c2 is None or out["C2"] == c2)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {out['verdict']}, C1 {out['C1']}, C2 {out['C2']}")
    return ok


def main():
    results = []
    p, m = synthetic()
    results.append(check("exact after offset", p, m, "PASS", True, True))
    results.append(check("all residuals +0.9", *synthetic(0.9), "PASS", True, True))
    results.append(check("all residuals +1.1 (RMS)", *synthetic(1.1), "FAIL", False, True))

    p, m = synthetic()
    p["jadeite"]["S_vib_bulk"][T1] += 2.1
    results.append(check("one residual 2.1 at 1000 K", p, m, "FAIL", False, True))

    p, m = synthetic()
    p["stishovite"]["S_vib_bulk"][T0] += 0.6
    results.append(check("stishovite off by 0.6 (C2)", p, m, "FAIL", True, False))

    p, m = synthetic()
    p["clinoenstatite"]["S_vib_bulk"][T1] += 0.6
    results.append(check("clinoenstatite off by 0.6 at 1000 K only", p, m, "FAIL", True, False))

    p, m = synthetic()
    p["quartz"]["space_group"][T1] = (154, "P3_221 or beta")
    results.append(check("quartz changes at 1000 K, scored at 298 only", p, m, "PASS", True, True))

    p, m = synthetic()
    p["quartz"]["space_group"][T0] = (1, "P1")
    p["stishovite"]["S_vib_bulk"][T0] += 0.6
    out = judge(p, m)
    ok = out["verdict"] == "PASS" and not out["families"]["SiO2"]["applied"] and out["excluded"] == ["quartz"]
    print(f"{'ok  ' if ok else 'FAIL'} quartz invalid: SiO2 family not applied, so stishovite's 0.6 is not judged")
    results.append(ok)

    p, m = synthetic()
    for tag in ("quartz", "calcite", "wollastonite"):
        p[tag]["space_group"][T0] = (1, "P1")
    results.append(check("three families lose a member (void)", p, m, "VOID (FAIL)"))

    p, m = synthetic()
    for tag in ("perovskite", "chrysoberyl", "diopside", "jadeite"):
        p[tag]["refused"] = "not a crystal"
    results.append(check("four cases excluded (void)", p, m, "VOID (FAIL)"))

    p, m = synthetic()
    q = copy.deepcopy(p)
    q["diopside"]["space_group"][T1] = (14, "P2_1/c")
    results.append(check("diopside changes at 1000 K, scored there (excluded, not void)", q, m, "PASS"))

    print(f"{sum(results)} of {len(results)} planted cases behave as the rule says")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
