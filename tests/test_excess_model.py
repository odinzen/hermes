import numpy as np
import pytest

from hermes.compression_entropy import GAS_CONSTANT
from hermes.excess_model import (
    ExcessModel,
    MixingData,
    candidate_feature_sets,
    espei_datasets,
    fit_excess_model,
    modified_aicc,
    select_excess_model,
)

R = GAS_CONSTANT
X = np.array([0.2, 0.36, 0.5, 0.7])
TS = np.array([2300.0, 2600.0, 2900.0])
# A liquid with L0 = -20000 + 3 T and L1 = 6000 (no T), so S_ex = -3 x(1-x).
TRUE = {(0, "a"): -20000.0, (0, "b"): 3.0, (1, "a"): 6000.0}


def synthetic(noise_scc=0.0, noise_h=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xx, tt = (g.ravel() for g in np.meshgrid(X, TS))
    y = 1 - 2 * xx
    h = xx * (1 - xx) * (TRUE[(0, "a")] + TRUE[(1, "a")] * y)
    # second derivatives: x(1-x) -> -2, x(1-x)(1-2x) -> -6(1-2x)
    l0 = TRUE[(0, "a")] + TRUE[(0, "b")] * tt
    g2 = -2 * l0 + TRUE[(1, "a")] * (-6 * y)
    scc = R * tt / (R * tt / (xx * (1 - xx)) + g2)
    return MixingData(
        enthalpy_x=xx, enthalpy_t=tt, enthalpy=h + rng.normal(0, noise_h, h.size) if noise_h else h,
        enthalpy_error=np.full(h.size, max(noise_h, 1.0)),
        scc_x=xx, scc_t=tt, scc=scc + rng.normal(0, noise_scc, scc.size) if noise_scc else scc,
        scc_error=np.full(scc.size, max(noise_scc, 1e-4)),
    )


def test_exact_data_recover_parameters_and_entropy():
    model = fit_excess_model(synthetic(), [(0, "a"), (0, "b"), (1, "a")])
    assert model.coefficients == pytest.approx([TRUE[(0, "a")], TRUE[(0, "b")], TRUE[(1, "a")]], rel=1e-6)
    s, _ = model.entropy(0.5, 2300.0)
    assert s == pytest.approx(-3.0 * 0.25, rel=1e-6)


def test_enthalpy_alone_cannot_see_entropy():
    d = synthetic()
    h_only = MixingData(enthalpy_x=d.enthalpy_x, enthalpy_t=d.enthalpy_t, enthalpy=d.enthalpy,
                        enthalpy_error=d.enthalpy_error)
    with pytest.warns(UserWarning, match="do not determine"):
        model = fit_excess_model(h_only, [(0, "a"), (0, "b")])
    # b_0 has no column in H: it is left at zero and its uncertainty is unknown, not zero
    assert model.interaction_parameters()[0]["b"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(model.entropy(0.5, 2300.0)[1])


def test_candidates_are_successive_like_espei():
    sets = candidate_feature_sets(2, ("a", "b"))
    assert sets == [[(0, "a")], [(0, "a"), (0, "b")],
                    [(0, "a"), (1, "a")], [(0, "a"), (1, "a"), (1, "b")],
                    [(0, "a"), (0, "b"), (1, "a")], [(0, "a"), (0, "b"), (1, "a"), (1, "b")]]
    # ESPEI's count N(1 - N^M)/(1 - N) for N temperature sets and M orders
    assert len(candidate_feature_sets(3, ("a", "b"))) == 14
    assert len(candidate_feature_sets(3, ("a", "b", "c"))) == 39
    assert len(candidate_feature_sets(3, ("a",))) == 3


def test_candidates_fall_back_to_shared_sets_past_the_limit():
    sets = candidate_feature_sets(3, ("a", "b"), candidate_limit=10)
    assert len(sets) == 6
    assert [(0, "a"), (0, "b"), (1, "a")] not in sets


def test_aicc_matches_espei_formula():
    n, k, rss = 12, 3, 4.0
    expected = n * np.log(rss / n) + 2 * k + (2 * k**2 + 2 * k) / (n - k - 1)
    assert modified_aicc(rss, n, k) == pytest.approx(expected)


def test_selection_with_noise_brackets_the_true_entropy():
    model, fits = select_excess_model(synthetic(noise_scc=0.004, noise_h=30.0, seed=3), max_order=3)
    s, err = model.entropy(X, 2300.0)
    assert len(fits) == 14
    assert np.all(np.abs(s - (-3.0 * X * (1 - X))) < 4 * err + 1e-9)
    assert np.all(err < 0.5)


def test_espei_dataset_shape():
    ds = espei_datasets([0.2, 0.5, 0.2], [2300, 2300, 2600], [-4000.0, -8000.0, -3900.0], "HM_MIX", ("cu", "zr"))
    assert len(ds) == 2
    first = ds[0]
    assert first["components"] == ["CU", "ZR"]
    assert np.array(first["values"]).shape == (1, 1, 2)
    assert len(first["solver"]["sublattice_configurations"]) == 2
    assert first["solver"]["sublattice_occupancies"][0] == [[0.8, 0.2]]
    assert all(d["excluded_model_contributions"] == ["idmix"] for d in ds)


def test_model_quantities_are_thermodynamically_consistent():
    features = [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "c")]
    model = ExcessModel(features, np.array([-20000.0, 7.0, -1.3, 5000.0, 0.8]), np.zeros((5, 5)), 0.0, 1.0)
    x, t, dt = np.array([0.15, 0.5, 0.8]), 1700.0, 1e-3
    g, h, s = model.gibbs(x, t)[0], model.enthalpy(x, t)[0], model.entropy(x, t)[0]
    assert np.allclose(g, h - t * s, rtol=1e-12)
    ds_numeric = -(model.gibbs(x, t + dt)[0] - model.gibbs(x, t - dt)[0]) / (2 * dt)
    assert np.allclose(s, ds_numeric, rtol=1e-6)


def test_reduced_chi_square_is_near_one_when_errors_are_right():
    true = [(0, "a"), (0, "b"), (1, "a")]
    chi2 = [fit_excess_model(synthetic(noise_scc=0.003, noise_h=30.0, seed=s), true).chi2_reduced for s in range(30)]
    assert 0.8 < np.mean(chi2) < 1.2


def test_a_poor_fit_inflates_the_reported_errors():
    true = [(0, "a"), (0, "b"), (1, "a")]
    honest = synthetic(noise_scc=0.003, noise_h=30.0, seed=4)
    claimed = MixingData(**{**honest.__dict__, "scc_error": honest.scc_error / 3, "enthalpy_error": honest.enthalpy_error / 3})
    err_honest = fit_excess_model(honest, true).entropy(0.5, 2300.0)[1]
    err_claimed = fit_excess_model(claimed, true).entropy(0.5, 2300.0)[1]
    assert err_claimed == pytest.approx(err_honest, rel=0.3)


def test_writers_refuse_components_out_of_alphabetical_order():
    model = ExcessModel([(1, "a")], np.array([5000.0]), np.zeros((1, 1)), 0.0, 1.0)
    assert model.tdb_parameters("LIQUID", "CU", "ZR")[0].startswith("PARAMETER G(LIQUID,CU,ZR;1)")
    with pytest.raises(ValueError, match="alphabetical"):
        model.tdb_parameters("LIQUID", "ZR", "CU")
    with pytest.raises(ValueError, match="alphabetical"):
        espei_datasets([0.5], [2300.0], [-8000.0], "HM_MIX", ("ZR", "CU"))
