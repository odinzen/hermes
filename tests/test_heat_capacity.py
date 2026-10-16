import numpy as np
import pytest
from scipy.integrate import quad

from hermes.heat_capacity import entropy_from_heat_capacity

R = 8.314462618


def test_classical_harmonic_solid():
    t = np.linspace(400.0, 1600.0, 7)
    out = entropy_from_heat_capacity(t, -40000.0 + 3 * R * t, 400.0, 30.0)
    assert np.allclose(out["entropy"], 30.0 + 3 * R * np.log(t / 400.0), rtol=0, atol=1e-9)
    assert np.allclose(out["heat_capacity"], 3 * R, rtol=1e-9)


def test_anharmonic_enthalpy_against_direct_integration():
    a = [-41000.0, 3 * R, 2.1e-3, 6.0e-7]  # H = a0 + a1 T + a2 T^2 + a3 T^3
    t = np.linspace(500.0, 1500.0, 8)
    out = entropy_from_heat_capacity(t, np.polynomial.polynomial.polyval(t, a), 500.0, 35.0, at=[900.0, 1500.0])
    heat_capacity = np.polynomial.polynomial.polyder(a)
    for temp, s in zip(out["temperature"], out["entropy"]):
        expected = 35.0 + quad(lambda x: np.polynomial.polynomial.polyval(x, heat_capacity) / x, 500.0, temp)[0]
        assert s == pytest.approx(expected, rel=0, abs=1e-8)


def test_by_parts_form_agrees():
    # S(T) - S(T0) = H(T)/T - H(T0)/T0 + int H/T^2 dT, the form that needs no derivative
    a = [-41000.0, 25.0, 3.0e-3, -4.0e-7]
    t = np.linspace(600.0, 1400.0, 6)
    h = np.polynomial.polynomial.polyval(t, a)
    out = entropy_from_heat_capacity(t, h, 600.0, 0.0, at=1400.0)
    hof = lambda x: np.polynomial.polynomial.polyval(x, a)  # noqa: E731
    by_parts = hof(1400.0) / 1400.0 - hof(600.0) / 600.0 + quad(lambda x: hof(x) / x**2, 600.0, 1400.0)[0]
    assert out["entropy"][0] == pytest.approx(by_parts, abs=1e-8)


def test_anchor_is_returned_at_its_own_temperature():
    t = np.linspace(300.0, 900.0, 6)
    out = entropy_from_heat_capacity(t, 3 * R * t + 1e-3 * t**2, 300.0, 22.5, at=300.0)
    assert out["entropy"][0] == pytest.approx(22.5, abs=1e-10)


def test_noisy_enthalpies_still_give_the_entropy():
    rng = np.random.default_rng(1)
    t = np.linspace(500.0, 1500.0, 7)
    h = -41000.0 + 3 * R * t + 2.0e-3 * t**2 + rng.normal(0, 15.0, t.size)  # 15 J/mol of MD noise
    out = entropy_from_heat_capacity(t, h, 500.0, 35.0, at=1500.0)
    exact = 35.0 + 3 * R * np.log(3.0) + 2 * 2.0e-3 * 1000.0
    assert out["entropy"][0] == pytest.approx(exact, abs=0.15)
    assert out["residual"] < 30.0


def test_refuses_extrapolation_and_too_few_runs():
    t = np.linspace(500.0, 1000.0, 6)
    with pytest.raises(ValueError, match="within the runs"):
        entropy_from_heat_capacity(t, 3 * R * t, 500.0, 30.0, at=1200.0)
    with pytest.raises(ValueError, match="within the runs"):
        entropy_from_heat_capacity(t, 3 * R * t, 300.0, 30.0)
    with pytest.raises(ValueError, match="at least"):
        entropy_from_heat_capacity(t[:4], 3 * R * t[:4], 500.0, 30.0)
