import numpy as np
import pytest

from hermes.compression_entropy import GAS_CONSTANT
from hermes.concentration_fluctuations import (
    concentration_structure_factor,
    excess_gibbs_from_scc,
    excess_mixing_entropy,
    long_wavelength_limit,
    reciprocal_vectors,
    redlich_kister,
)

R = GAS_CONSTANT


def test_reciprocal_vectors_of_a_cubic_cell():
    k = reciprocal_vectors(np.eye(3) * 10.0, 2 * np.pi / 10.0 * 1.01)
    assert len(k) == 3  # (1,0,0), (0,1,0), (0,0,1), one of each +/- pair
    assert np.allclose(np.linalg.norm(k, axis=1), 2 * np.pi / 10.0)


def test_random_labels_give_ideal_mixing():
    rng = np.random.default_rng(0)
    box, n = 20.0, 500
    x = rng.uniform(0, box, (200, n, 3))
    species = (np.arange(n) < 150).astype(int)  # x_B = 0.3 with B = label 1
    result = concentration_structure_factor(x, np.eye(3) * box, species, k_max=1.2)
    s0, err = long_wavelength_limit(result["k"], result["s_cc"], result["blocks"], k_fit=1.2)
    assert result["x_B"] == pytest.approx(0.3)
    assert s0 == pytest.approx(0.21, abs=4 * err)
    assert err < 0.01


def curvature_to_scc(x, coefficients, temperature):
    h = 1e-5
    g2 = (redlich_kister(x + h, coefficients) - 2 * redlich_kister(x, coefficients)
          + redlich_kister(x - h, coefficients)) / h**2
    rt = R * temperature
    return rt / (rt / (x * (1 - x)) + g2)


@pytest.mark.parametrize("coefficients", [[-15000.0], [-17000.0, 7000.0], [-18000.0, 7000.0, 11000.0]])
def test_redlich_kister_round_trip(coefficients):
    x = np.array([0.2, 0.36, 0.5, 0.7])
    s = curvature_to_scc(x, coefficients, 2300.0)
    fitted = excess_gibbs_from_scc(x, s, 2300.0, order=len(coefficients))
    assert fitted == pytest.approx(coefficients, rel=1e-5)


def test_ideal_scc_gives_zero_excess():
    x = np.array([0.25, 0.5, 0.75])
    out = excess_mixing_entropy(x, x * (1 - x), np.zeros(3), 1000.0)
    assert np.allclose(out["coefficients"], 0.0, atol=1e-6)
    assert np.allclose(out["s_ex"], 0.0, atol=1e-9)


def test_regular_solution_entropy():
    # A regular solution has S_ex = 0: G_ex = H_ex = L0 x(1-x).
    x, l0, t = np.array([0.2, 0.5, 0.8]), -12000.0, 1500.0
    out = excess_mixing_entropy(x, curvature_to_scc(x, [l0], t), l0 * x * (1 - x), t, order=1)
    assert np.allclose(out["s_ex"], 0.0, atol=1e-4)


def test_random_labels_on_a_perfect_lattice_give_ideal_mixing():
    # the total density of a perfect lattice vanishes away from Bragg peaks, so this tells the
    # concentration weight x_B rho_A - x_A rho_B from the density weight x_B rho_A + x_A rho_B
    n = 8
    sites = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float) * 3.0
    species = np.random.default_rng(12).permutation(np.repeat([0, 1], len(sites) // 2))
    out = concentration_structure_factor(sites[None], np.eye(3) * 3.0 * n, species, k_max=1.5)
    assert out["s_cc"].mean() == pytest.approx(0.25, rel=0.1)


def test_long_wavelength_limit_recovers_a_quadratic_intercept():
    k = np.linspace(0.2, 1.0, 30)
    s0, _ = long_wavelength_limit(k, 0.17 + 0.4 * k**2)
    assert s0 == pytest.approx(0.17, rel=1e-9)


def test_excess_entropy_sign_from_a_known_regular_solution():
    # G_ex = -12 kJ x(1-x) and H_ex = -20 kJ x(1-x), so S_ex = (H - G) / T = -8 kJ x(1-x) / T
    t, x = 1500.0, np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    s_cc = 1.0 / (1.0 / (x * (1 - x)) + 2 * 12000.0 / (R * t))
    out = excess_mixing_entropy(x, s_cc, -20000.0 * x * (1 - x), t, order=1)
    assert np.allclose(out["s_ex"], -8000.0 * x * (1 - x) / t, rtol=1e-9)
