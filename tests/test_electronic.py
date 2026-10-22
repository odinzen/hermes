import numpy as np
import pytest
from scipy.integrate import simpson

from hermes.electronic import (
    BOLTZMANN_EV,
    GAS_CONSTANT,
    chemical_potential,
    electron_count,
    electronic_entropy,
    electronic_free_energy,
)

R = GAS_CONSTANT


def free_electron_dos(fermi=5.0, top=20.0, n=4001, electrons=2.0):
    # g = c sqrt(E) holding `electrons` below the Fermi energy, on a 5 meV grid
    e = np.linspace(0.0, top, n)
    c = electrons / (2.0 / 3.0 * fermi**1.5)
    return e, c * np.sqrt(e), c * np.sqrt(fermi)


def gapped_dos(gap=5.0, n=2001):
    # a valence band of 8 states below -gap/2 and an empty conduction band above +gap/2
    e = np.linspace(-12.0, 12.0, n)
    g = np.where((e > -8.0) & (e < -gap / 2), 2.0, 0.0) + np.where((e > gap / 2) & (e < 8.0), 1.5, 0.0)
    return e, g


@pytest.mark.parametrize("temperature", [30.0, 300.0, 1000.0])
def test_free_electron_metal_gives_gamma_t(temperature):
    e, g, g_f = free_electron_dos()
    s = electronic_entropy(e, g, temperature, atoms=1, n_electrons=2.0)
    gamma = np.pi**2 / 3 * R * BOLTZMANN_EV * g_f  # J/(K^2 mol) for a one-atom cell
    # Sommerfeld corrections are of order (k_B T / E_F)^2, 3e-4 at 1000 K
    assert s == pytest.approx(gamma * temperature, rel=5e-4)


def test_sommerfeld_coefficient_from_first_principles_constants():
    # gamma = (pi^2 / 3) k_B^2 N_A g(E_F) with g in states per joule: 2.357 mJ/(K^2 mol) per state/eV/atom
    k, e_charge, n_a = 1.380649e-23, 1.602176634e-19, 6.02214076e23
    gamma = np.pi**2 / 3 * k**2 * n_a / e_charge
    assert gamma == pytest.approx(2.357e-3, rel=1e-3)
    energies = np.linspace(-10.0, 10.0, 8001)
    flat = np.ones_like(energies)  # 1 state/eV/atom, half full
    s = electronic_entropy(energies, flat, 300.0, atoms=1, fermi_energy=0.0)
    assert s == pytest.approx(gamma * 300.0, rel=1e-6)


@pytest.mark.parametrize("temperature", [100.0, 300.0])
def test_fermi_function_resolved_on_a_coarse_grid(temperature):
    # a 0.2 eV grid, eight to twenty-three k_B T: the quadrature, not the grid, has to resolve the Fermi function
    e = np.linspace(-10.0, 10.0, 101)
    s = electronic_entropy(e, np.ones_like(e), temperature, atoms=1, fermi_energy=0.05)
    assert s == pytest.approx(np.pi**2 / 3 * R * BOLTZMANN_EV * temperature, rel=1e-9)


def test_wide_gap_insulator_has_no_electronic_entropy():
    e, g = gapped_dos()
    n = electron_count(e, g, 0.0)
    assert n == pytest.approx(2.0 * 5.5, rel=1e-3)
    assert electronic_entropy(e, g, 1500.0, atoms=2, n_electrons=n) < 1e-5
    # mu sits in the gap, (k_B T / 2) ln(4/3) toward the band with fewer states (1.5 against 2 per eV)
    mu = chemical_potential(e, g, n, 1500.0)
    assert mu == pytest.approx(0.5 * BOLTZMANN_EV * 1500.0 * np.log(4 / 3), abs=1e-3)


def test_two_levels_across_a_gap():
    # one state at 0 and one at delta, one electron: f = 1 / (1 + exp(delta / 2 k_B T)) on each, mu at delta / 2
    delta, width = 0.3, 2e-4
    e = np.linspace(-1.0, 1.3, 230001)
    g = sum(np.exp(-0.5 * ((e - c) / width) ** 2) / (width * np.sqrt(2 * np.pi)) for c in (0.0, delta))
    t = 800.0
    x = delta / (2 * BOLTZMANN_EV * t)
    f = 1 / (1 + np.exp(x))
    exact = -2 * (f * np.log(f) + (1 - f) * np.log(1 - f))
    assert chemical_potential(e, g, 1.0, t) == pytest.approx(delta / 2, abs=1e-6)
    assert electronic_entropy(e, g, t, atoms=1, n_electrons=1.0) == pytest.approx(R * exact, rel=1e-4)


def test_spin_channels_are_summed_and_per_cell_unit():
    e, g, _ = free_electron_dos()
    total = electronic_entropy(e, g, 500.0, atoms=4, n_electrons=2.0)
    split = electronic_entropy(e, np.vstack([0.5 * g, 0.5 * g]), 500.0, atoms=4, n_electrons=2.0)
    assert split == pytest.approx(total)
    per_cell = electronic_entropy(e, g, 500.0, atoms=4, n_electrons=2.0, per_cell=True)
    # J/(K mol of atoms) * atoms / N_A is J/K per cell, / e is eV/K
    assert per_cell == pytest.approx(total * 4 / 6.02214076e23 / 1.602176634e-19)


def test_fermi_energy_and_electron_count_agree():
    e, g, _ = free_electron_dos()
    assert electron_count(e, g, 5.0) == pytest.approx(2.0, rel=1e-5)
    a = electronic_entropy(e, g, [100.0, 700.0], atoms=1, fermi_energy=5.0)
    b = electronic_entropy(e, g, [100.0, 700.0], atoms=1, n_electrons=electron_count(e, g, 5.0))
    assert a == pytest.approx(b)


def test_chemical_potential_follows_sommerfeld():
    # mu = E_F [1 - (pi^2 / 12) (k_B T / E_F)^2] for g ~ sqrt(E)
    e, g, _ = free_electron_dos()
    t = 3000.0
    kt = BOLTZMANN_EV * t
    assert chemical_potential(e, g, 2.0, t) == pytest.approx(5.0 * (1 - np.pi**2 / 12 * (kt / 5.0) ** 2), abs=2e-4)


def test_free_energy_of_a_metal_is_minus_half_ts():
    # S = gamma T, so F - F(0) = -gamma T^2 / 2
    e, g, _ = free_electron_dos()
    t = np.array([200.0, 800.0])
    s = electronic_entropy(e, g, t, atoms=1, n_electrons=2.0)
    f = electronic_free_energy(e, g, t, atoms=1, n_electrons=2.0)
    assert f == pytest.approx(-0.5 * t * s, rel=1e-3)
    assert electronic_free_energy(e, g, 0.0, atoms=1, n_electrons=2.0) == 0.0


def test_free_energy_is_minus_the_integral_of_entropy():
    e, g = gapped_dos(gap=0.6)
    n = electron_count(e, g, 0.0)
    t = np.linspace(100.0, 1200.0, 221)
    s = electronic_entropy(e, g, t, atoms=1, n_electrons=n)
    f = electronic_free_energy(e, g, 1200.0, atoms=1, n_electrons=n)
    lower = electronic_free_energy(e, g, 100.0, atoms=1, n_electrons=n)
    assert f - lower == pytest.approx(-simpson(s, x=t), rel=1e-6)


def test_zero_temperature_and_scalar_shapes():
    e, g, _ = free_electron_dos()
    assert electronic_entropy(e, g, 0.0, atoms=1, n_electrons=2.0) == 0.0
    out = electronic_entropy(e, g, [[0.0, 300.0]], atoms=1, n_electrons=2.0)
    assert out.shape == (1, 2)


def test_bad_input_is_refused():
    e, g, _ = free_electron_dos()
    with pytest.raises(ValueError):
        electronic_entropy(e, g, 300.0, atoms=1)  # no electron count
    with pytest.raises(ValueError):
        electronic_entropy(e, g, 300.0, atoms=1, n_electrons=2.0, fermi_energy=5.0)
    with pytest.raises(ValueError):
        electronic_entropy(e, g, 300.0, atoms=1, n_electrons=1e6)  # more electrons than states
    with pytest.raises(ValueError):
        electronic_entropy(e[::-1], g, 300.0, atoms=1, n_electrons=2.0)
    with pytest.raises(ValueError):
        electronic_entropy(e, -g, 300.0, atoms=1, n_electrons=2.0)
    flat_e = np.linspace(-10.0, 10.0, 2001)
    with pytest.raises(ValueError):
        # a band cut off by the window, with mu half an eV inside it
        electronic_entropy(flat_e, np.ones_like(flat_e), 3000.0, atoms=1, n_electrons=0.5)
    # the same mu next to a band edge is fine: the free-electron band has no states below 0
    assert electronic_entropy(e, g, 3000.0, atoms=1, n_electrons=2.0) > 0
