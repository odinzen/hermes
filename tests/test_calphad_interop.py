"""Hermes output read back by pycalphad and ESPEI; skipped where those are not installed."""

import numpy as np
import pytest

from hermes.excess_model import ExcessModel, espei_datasets

MODEL = ExcessModel(
    features=[(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (2, "a")],
    coefficients=np.array([-21000.0, 4.5, -0.3, 6500.0, -1.2, 2500.0]),
    covariance=np.zeros((6, 6)), score=0.0, chi2_reduced=1.0,
)


def test_pycalphad_reads_the_tdb_parameters_as_meant():
    pycalphad = pytest.importorskip("pycalphad")
    tdb = "\n".join([
        "ELEMENT /-   ELECTRON_GAS 0 0 0 !",
        "ELEMENT VA   VACUUM 0 0 0 !",
        "ELEMENT CU   FCC_A1 63.546 0 0 !",
        "ELEMENT ZR   HCP_A3 91.224 0 0 !",
        "PHASE LIQUID % 1 1 !",
        "CONSTITUENT LIQUID :CU,ZR: !",
        "PARAMETER G(LIQUID,CU;0) 298.15 0; 6000 N !",
        "PARAMETER G(LIQUID,ZR;0) 298.15 0; 6000 N !",
        *MODEL.tdb_parameters("LIQUID", "CU", "ZR"),
    ])
    dbf = pycalphad.Database(tdb)
    r = float(pycalphad.variables.R)  # the CALPHAD value 8.3145
    xs = np.array([0.1, 0.36, 0.5, 0.9])
    for t in (1500.0, 2900.0):
        def prop(name):
            res = pycalphad.calculate(dbf, ["CU", "ZR"], "LIQUID", T=t, P=101325, N=1,
                                      points=np.stack([1 - xs, xs], axis=1), output=name)
            return np.squeeze(getattr(res, name).values)
        ideal_s = -r * (xs * np.log(xs) + (1 - xs) * np.log(1 - xs))
        assert prop("GM") + t * ideal_s == pytest.approx(MODEL.gibbs(xs, t)[0], abs=1e-6)
        assert prop("HM") == pytest.approx(MODEL.enthalpy(xs, t)[0], abs=1e-6)
        assert prop("SM") - ideal_s == pytest.approx(MODEL.entropy(xs, t)[0], abs=1e-9)


def test_espei_accepts_the_datasets():
    datasets = pytest.importorskip("espei.datasets")
    for output, values in (("HM_MIX", [-4000.0, -8000.0, -3900.0]), ("SM_MIX", [-1.2, -1.7, -1.1])):
        for ds in espei_datasets([0.2, 0.5, 0.2], [2300, 2300, 2600], values, output, ("CU", "ZR")):
            datasets.check_dataset(ds)


def test_espei_fit_of_excess_entropy_gives_ideal_plus_excess(tmp_path):
    # Excess SM_MIX only lands right if ESPEI is told not to subtract ideal mixing from it; a fit
    # that treats it as total comes out short by R ln 2 = 5.76 at x = 0.5.
    pycalphad = pytest.importorskip("pycalphad")
    espei_datasets_module = pytest.importorskip("espei.datasets")
    paramselect = pytest.importorskip("espei.paramselect")
    import json

    xs, t = np.linspace(0.2, 0.8, 7), 1000.0
    s_ex = 4.0 * xs * (1 - xs)
    for output, values in (("HM_MIX", -20000.0 * xs * (1 - xs)), ("SM_MIX", s_ex)):
        for i, ds in enumerate(espei_datasets(xs, np.full_like(xs, t), values, output, ("CU", "MG"))):
            (tmp_path / f"{output}-{i}.json").write_text(json.dumps(ds))
    datasets = espei_datasets_module.load_datasets(sorted(str(p) for p in tmp_path.glob("*.json")))
    phase_models = {"components": ["CU", "MG"],
                    "phases": {"LIQUID": {"sublattice_model": [["CU", "MG"]], "sublattice_site_ratios": [1]}}}
    dbf = paramselect.generate_parameters(phase_models, datasets, "SGTE91", "linear")

    res = pycalphad.calculate(dbf, ["CU", "MG"], "LIQUID", T=t, P=101325, N=1,
                              points=np.stack([1 - xs, xs], axis=1), output="SM_MIX")
    r = float(pycalphad.variables.R)
    ideal = -r * (xs * np.log(xs) + (1 - xs) * np.log(1 - xs))
    assert np.squeeze(res.SM_MIX.values) == pytest.approx(ideal + s_ex, abs=1e-3)
