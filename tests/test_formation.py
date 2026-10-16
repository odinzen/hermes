import json

import numpy as np
import pytest

from hermes.formation import PhaseProperties, atom_fractions, espei_formation_datasets, formation_properties

T = np.linspace(300.0, 1500.0, 9)
# a formation Gibbs energy per mole of atoms in ESPEI's own features: a + b T + c T ln T + d T^2 + e / T
A, B, C, D, E = -24000.0, 6.0, -0.8, -1.5e-3, 9.0e4


def formation_truth(t):
    h = A - C * t - D * t**2 + 2 * E / t
    s = -(B + C * (np.log(t) + 1) + 2 * D * t - E / t**2)
    cp = -C - 2 * D * t - 2 * E / t**2
    return h, s, cp


def absolute(t, scale):
    """Some absolute properties for a reference element (any smooth set will do)."""
    return PhaseProperties(t, enthalpy=-30000.0 * scale + 26.0 * t + 4e-3 * t**2,
                           entropy=40.0 * scale + 26.0 * np.log(t / 300.0) + 8e-3 * (t - 300.0),
                           heat_capacity=26.0 + 8e-3 * t)


REFS = {"CU": absolute(T, 1.0), "ZR": absolute(T, 1.3)}


def compound():
    h, s, cp = formation_truth(T)
    return PhaseProperties(T, enthalpy=0.5 * REFS["CU"].enthalpy + 0.5 * REFS["ZR"].enthalpy + h,
                           entropy=0.5 * REFS["CU"].entropy + 0.5 * REFS["ZR"].entropy + s,
                           heat_capacity=0.5 * REFS["CU"].heat_capacity + 0.5 * REFS["ZR"].heat_capacity + cp)


def test_atom_fractions():
    assert atom_fractions(["AL", "NI"], [0.75, 0.25]) == pytest.approx({"AL": 0.75, "NI": 0.25})
    assert atom_fractions(["NI", "VA"], [1, 3]) == {"NI": 1.0}
    with pytest.raises(ValueError, match="endmembers"):
        atom_fractions([["CU", "ZR"], "ZR"], [1, 1])


def test_formation_is_the_atom_balanced_difference():
    out = formation_properties(compound(), REFS, ["CU", "ZR"], [1, 1])
    h, s, cp = formation_truth(T)
    assert np.allclose(out.enthalpy, h, rtol=0, atol=1e-9)
    assert np.allclose(out.entropy, s, rtol=0, atol=1e-12)
    assert np.allclose(out.heat_capacity, cp, rtol=0, atol=1e-12)


def test_references_must_match_the_phase():
    with pytest.raises(ValueError, match="no reference"):
        formation_properties(compound(), {"CU": REFS["CU"]}, ["CU", "ZR"], [1, 1])
    shifted = {"CU": REFS["CU"], "ZR": absolute(T + 10.0, 1.3)}
    with pytest.raises(ValueError, match="other temperatures"):
        formation_properties(compound(), shifted, ["CU", "ZR"], [1, 1])
    partial = {"CU": REFS["CU"], "ZR": PhaseProperties(T, entropy=REFS["ZR"].entropy)}
    with pytest.raises(ValueError, match="has no enthalpy"):
        formation_properties(compound(), partial, ["CU", "ZR"], [1, 1])


def test_dataset_layout():
    ds = espei_formation_datasets("CUZR", ["CU", "ZR"], [1, 1], compound(), REFS)
    assert [d["output"] for d in ds] == ["HM_FORM", "SM_FORM", "CPM_FORM"]
    for d in ds:
        assert np.array(d["values"]).shape == (1, len(T), 1)
        assert d["conditions"]["T"] == list(T)
        assert d["solver"]["sublattice_configurations"] == [["CU", "ZR"]]
    only_s = PhaseProperties(T, entropy=compound().entropy)
    assert [d["output"] for d in espei_formation_datasets("CUZR", ["CU", "ZR"], [1, 1], only_s, REFS)] == ["SM_FORM"]


def test_espei_fit_reproduces_the_formation_properties(tmp_path):
    datasets_module = pytest.importorskip("espei.datasets")
    paramselect = pytest.importorskip("espei.paramselect")
    pycalphad = pytest.importorskip("pycalphad")
    for d in espei_formation_datasets("CUZR", ["CU", "ZR"], [1, 1], compound(), REFS):
        datasets_module.check_dataset(d)
        (tmp_path / f"{d['output']}.json").write_text(json.dumps(d))
    datasets = datasets_module.load_datasets(sorted(str(p) for p in tmp_path.glob("*.json")))
    phase_models = {"components": ["CU", "ZR"],
                    "phases": {"CUZR": {"sublattice_model": [["CU"], ["ZR"]], "sublattice_site_ratios": [1, 1]}}}
    # ridge_alpha as small as ESPEI documents: with its default (None) scikit-learn's LinearRegression (1.9) drops
    # the badly scaled 1/T column of the heat capacity and the fit comes out wrong
    dbf = paramselect.generate_parameters(phase_models, datasets, "SGTE91", "linear", ridge_alpha=1e-100)
    (param,) = [p for p in dbf._parameters.all() if p["phase_name"] == "CUZR" and p["parameter_type"] == "G"]
    temperature = pycalphad.variables.T
    import symengine

    # the endmember is written as GHSERCU + GHSERZR + the fitted formation part, coefficients as named symbols
    g = param["parameter"].xreplace({symengine.Symbol("GHSERCU"): 0, symengine.Symbol("GHSERZR"): 0})
    g = g.xreplace({symengine.Symbol(k): v for k, v in dbf.symbols.items()})
    g = g / 2  # per formula unit of two atoms -> per atom
    dg, d2g = g.diff(temperature), g.diff(temperature, 2)
    h, s, cp = formation_truth(T)
    for i, t in enumerate(T):
        at = {temperature: t}
        assert float(-dg.xreplace(at)) == pytest.approx(s[i], abs=1e-3)
        assert float((g - temperature * dg).xreplace(at)) == pytest.approx(h[i], abs=0.5)
        assert float((-temperature * d2g).xreplace(at)) == pytest.approx(cp[i], abs=1e-3)
    # and pycalphad, reading the fitted database, gives the compound's entropy as SGTE elements plus that formation
    for i, t in enumerate(T[2:], start=2):  # above 298.15 K, where the GHSER pieces apply
        res = pycalphad.calculate(dbf, ["CU", "ZR"], "CUZR", T=t, P=101325, N=1, output="SM")
        ser = 0.0
        for element in ("CU", "ZR"):
            ghser = dbf.symbols[f"GHSER{element}"]
            ser += 0.5 * float(-ghser.diff(temperature).xreplace({temperature: t}))
        assert float(np.squeeze(res.SM.values)) == pytest.approx(ser + s[i], abs=1e-3)
