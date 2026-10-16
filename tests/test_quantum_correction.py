import numpy as np
import pytest

from hermes.quantum_correction import (
    GAS_CONSTANT,
    H_OVER_KB,
    harmonic_entropy,
    quantum_correction,
    vibrational_density_of_states,
)

R = GAS_CONSTANT


def einstein_dos(nu0, width=0.01):
    nu = np.linspace(0, 3 * nu0, 6001)
    g = np.exp(-0.5 * ((nu - nu0) / width) ** 2)
    return nu, 3 * g / np.trapezoid(g, nu)


@pytest.mark.parametrize("temperature", [50.0, 300.0, 1500.0])
def test_einstein_solid(temperature):
    nu0 = 5.0
    nu, g = einstein_dos(nu0)
    x = H_OVER_KB * nu0 / temperature
    exact_q = 3 * R * (x / np.expm1(x) - np.log(-np.expm1(-x)))
    exact_cl = 3 * R * (1 - np.log(x))
    assert harmonic_entropy(nu, g, temperature) == pytest.approx(exact_q, rel=1e-4)
    assert harmonic_entropy(nu, g, temperature, quantum=False) == pytest.approx(exact_cl, rel=1e-4, abs=1e-4)
    assert quantum_correction(nu, g, temperature) == pytest.approx(exact_q - exact_cl, rel=1e-4)


def test_correction_is_positive_and_follows_x_squared_over_24():
    nu, g = einstein_dos(2.0)
    temperature = 3000.0
    x = H_OVER_KB * 2.0 / temperature
    correction = quantum_correction(nu, g, temperature)
    assert correction > 0
    assert correction == pytest.approx(3 * R * x**2 / 24, rel=0.01)


def test_spectrum_of_oscillating_velocities():
    dt, nu0 = 0.004, 6.0
    t = dt * np.arange(5000)
    rng = np.random.default_rng(0)
    phases = rng.uniform(0, 2 * np.pi, (1, 50, 3))
    v = np.cos(2 * np.pi * nu0 * t[:, None, None] + phases)
    nu, g = vibrational_density_of_states(v, dt)
    assert np.trapezoid(g, nu) == pytest.approx(3.0)
    assert nu[np.argmax(g)] == pytest.approx(nu0, abs=0.2)


def test_segment_longer_than_trajectory_is_refused():
    with pytest.raises(ValueError):
        vibrational_density_of_states(np.zeros((10, 2, 3)), 0.004, segment_ps=5.0)


def test_entropy_of_a_5_thz_oscillator_from_first_principles():
    # x = h nu / k_B T with the SI constants written out, so the THz-to-K factor is checked independently
    x = 6.62607015e-34 * 5e12 / (1.380649e-23 * 300.0)
    s_exact = 3 * 8.314462618 * (x / np.expm1(x) - np.log(-np.expm1(-x)))
    nu, dos = einstein_dos(5.0)
    assert harmonic_entropy(nu, dos, 300.0) == pytest.approx(s_exact, rel=1e-4)


def test_spectrum_weights_atoms_by_mass():
    # equipartition: the heavy atom's velocity is 10 times smaller, its kinetic energy the same
    dt, steps = 0.002, 5000
    t = dt * np.arange(steps)[:, None]
    v = np.zeros((steps, 2, 3))
    v[:, 0, :] = np.cos(2 * np.pi * 4.0 * t)
    v[:, 1, :] = 0.1 * np.cos(2 * np.pi * 9.0 * t)
    nu, dos = vibrational_density_of_states(v, dt, masses_amu=[1.0, 100.0])
    low, high = nu < 6.5, nu >= 6.5
    assert np.trapezoid(np.where(low, dos, 0), nu) == pytest.approx(1.5, rel=0.02)
    assert np.trapezoid(np.where(high, dos, 0), nu) == pytest.approx(1.5, rel=0.02)
