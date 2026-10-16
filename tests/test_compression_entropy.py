# Third-Party Library Imports
import numpy as np
import pytest
import scipy.constants

# Hermes Imports
from hermes.compression_entropy import (
    GAS_CONSTANT,
    coefficient_bits,
    compress,
    displacement_residuals,
    microstate_entropy,
    thermal_de_broglie_wavelength,
    trajectory_entropy,
)
from hermes.md_configuration import CLASSICAL_ENTROPY_OFFSET, configuration_from_md
from hermes.system import System

R = GAS_CONSTANT


def lattice(n=7, a=4.05):
    sites = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], dtype=float) * a
    return sites, np.eye(3) * n * a


def einstein_entropy(mass_amu, frequency_hz, temperature, n_samples=4, seed=0):
    """Exact classical entropy and the calibrated compression entropy of a sampled Einstein solid."""
    rng = np.random.default_rng(seed)
    sites, cell = lattice()
    omega = 2 * np.pi * frequency_hz
    m = mass_amu * scipy.constants.atomic_mass
    sigma = np.sqrt(scipy.constants.k * temperature / (m * omega**2)) / 1e-10
    lam = thermal_de_broglie_wavelength([mass_amu], temperature)[0]
    values = [microstate_entropy(sites + rng.normal(0, sigma, sites.shape), sites + rng.normal(0, sigma, sites.shape),
                                 cell, lam) for _ in range(n_samples)]
    exact = 3 * R * (1 + np.log(scipy.constants.k * temperature / (scipy.constants.hbar * omega)))
    return exact, float(np.mean(values)), sigma / lam


def test_de_broglie_wavelength_of_aluminum_at_room_temperature():
    assert thermal_de_broglie_wavelength([26.98], 300.0)[0] == pytest.approx(0.194, abs=0.001)


def test_compress_meets_the_bound_with_ternary_coefficients():
    rng = np.random.default_rng(1)
    residuals = rng.normal(0, 0.2, (300, 3))
    eps = rng.uniform(0.01, 0.05, 300)
    basis, coefficients = compress(residuals, eps)
    assert set(np.unique(coefficients)) <= {-1, 0, 1}
    assert np.all(np.linalg.norm(residuals - coefficients @ basis, axis=1) <= eps * (1 + 1e-12))


def test_nothing_to_encode_gives_zero_entropy():
    sites, cell = lattice(3)
    assert microstate_entropy(sites, sites, cell, 0.05, group_size=None) == pytest.approx(0.0, abs=1e-12)


def test_coefficient_bits_per_column_and_pooled():
    c = np.array([[1, 0], [-1, 0], [1, 0], [-1, 1]])
    # column 0: +1 and -1 each half, 1 bit per entry; column 1: 3/4 zeros, 0.811 bits per entry
    per_column = 4 * 1.0 + 4 * -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    assert coefficient_bits(c) == pytest.approx(per_column)
    p = np.array([2, 3, 3]) / 8  # -1, 0, 1 over all eight entries
    assert coefficient_bits(c, pooled=True) == pytest.approx(8 * -(p * np.log2(p)).sum())
    assert coefficient_bits(c) <= coefficient_bits(c, pooled=True)


def test_residuals_use_the_nearest_periodic_image():
    cell = np.eye(3) * 10.0
    target = np.array([[9.95, 0.0, 0.0]])
    reference = np.array([[0.05, 0.0, 0.0]])
    assert displacement_residuals(target, reference, cell)[0] == pytest.approx([-0.1, 0.0, 0.0], abs=1e-12)


def test_doubling_the_amplitude_adds_three_bits_per_atom():
    # Information dimension of three coordinates per atom: S(2 sigma) - S(sigma) = 3 R ln 2.
    rng = np.random.default_rng(2)
    sites, cell = lattice()
    s = {}
    for sigma in (0.1, 0.2, 0.4):
        s[sigma] = np.mean([microstate_entropy(sites + rng.normal(0, sigma, sites.shape),
                                               sites + rng.normal(0, sigma, sites.shape), cell, 0.03)
                            for _ in range(3)])
    step = (s[0.4] - s[0.1]) / 2
    assert step == pytest.approx(3 * R * np.log(2), rel=0.06)


@pytest.mark.parametrize("mass, frequency, temperature", [(26.98, 6e12, 1100.0), (12.011, 15e12, 2600.0), (26.98, 6e12, 3500.0)])
def test_calibrated_entropy_matches_the_exact_einstein_solid(mass, frequency, temperature):
    exact, compression, ratio = einstein_entropy(mass, frequency, temperature)
    assert ratio > 0.8  # inside the stated range of validity
    assert compression + CLASSICAL_ENTROPY_OFFSET == pytest.approx(exact, abs=1.0)


def test_trajectory_entropy_returns_its_samples():
    rng = np.random.default_rng(3)
    sites, cell = lattice(4)
    frames = sites + rng.normal(0, 0.15, (12,) + sites.shape)
    out = trajectory_entropy(frames, cell, np.full(len(sites), 26.98), 1500.0, n_pairs=6, min_separation=2, dither=2,
                             group_size=None)
    assert len(out["samples"]) == 6
    assert out["minimum"] <= out["mean"]


def test_configuration_from_md_round_trips_the_entropy_through_system():
    # A single configuration: the System entropy must equal the input entropy at every T and V.
    volumes = np.linspace(15.0, 18.0, 7)
    temperatures = np.linspace(800.0, 1600.0, 5)
    n = 4
    s_comp = 40.0 + 3 * R * np.log(temperatures[:, None] / 800.0) + 2 * R * np.log(volumes[None, :] / 15.0)
    potential = -3.4 * n + 0.5 * (volumes[None, :] - 16.5) ** 2 + 1.5 * n * 8.617e-5 * temperatures[:, None]
    pressure = -160.21766208 * np.gradient(potential, volumes, axis=1)
    config = configuration_from_md("md", 1, n, volumes, temperatures, potential, pressure, s_comp)
    system = System({"md": config}, ground_state="md")
    expected = (s_comp + CLASSICAL_ENTROPY_OFFSET) * n / (scipy.constants.Avogadro * scipy.constants.electron_volt)
    assert np.allclose(system.entropies, expected, rtol=1e-10)
    assert np.allclose(system.configurational_entropies, 0.0, atol=1e-15)


def test_configuration_from_md_rejects_a_zero_kelvin_grid():
    with pytest.raises(ValueError):
        z = np.zeros((2, 3))
        configuration_from_md("md", 1, 4, [1.0, 2.0, 3.0], [0.0, 100.0], z, z, z)


def test_configuration_from_md_subtracts_the_correlation():
    volumes, temperatures, n = np.linspace(15.0, 18.0, 4), np.linspace(800.0, 1600.0, 3), 4
    z = np.zeros((3, 4))
    s_comp, corr = np.full((3, 4), 40.0), np.full((3, 4), 5.3)
    plain = configuration_from_md("md", 1, n, volumes, temperatures, z, z, s_comp)
    corrected = configuration_from_md("md", 1, n, volumes, temperatures, z, z, s_comp, correlation_entropies=corr)
    per_mole = 5.3 * n / (scipy.constants.Avogadro * scipy.constants.electron_volt)
    assert np.allclose(plain.entropies - corrected.entropies, per_mole, rtol=1e-12)
    with pytest.raises(ValueError):
        configuration_from_md("md", 1, n, volumes, temperatures, z, z, s_comp, correlation_entropies=corr[:2])


def test_configuration_from_md_energy_and_pressure():
    volumes, temperatures, n = np.linspace(15.0, 18.0, 4), np.linspace(800.0, 1600.0, 3), 4
    potential = -3.4 * n + 0.1 * volumes[None, :] + np.zeros((3, 1))
    pressure = np.full((3, 4), 2.5)  # GPa
    config = configuration_from_md("md", 1, n, volumes, temperatures, potential, pressure, np.full((3, 4), 40.0))
    energy = config.helmholtz_energies + temperatures[:, None] * config.entropies
    assert np.allclose(energy, potential + 1.5 * n * scipy.constants.k / scipy.constants.e * temperatures[:, None],
                       rtol=1e-12)
    assert np.allclose(config.helmholtz_energies_dV, -2.5 / 160.21766208, rtol=1e-12)


def test_npt_strain_between_frames_is_not_coded_as_displacement():
    sites, cell = lattice(n=6)
    frac = np.linalg.solve(cell.T, sites.T).T
    strained = 1.005 * cell
    residuals = displacement_residuals(frac @ strained, sites, strained, reference_cell=cell)
    assert np.abs(residuals).max() < 1e-9


def test_too_few_frames_for_the_separation_raise():
    sites, cell = lattice(n=3)
    with pytest.raises(ValueError):
        trajectory_entropy(sites[None], cell, np.full(len(sites), 27.0), 1000.0, n_pairs=2, min_separation=1)


def test_offset_does_not_depend_on_the_box():
    # coded all at once, the fixed cost per atom fell by about 1.5 J/(K mol) from 343 to 1728 atoms
    rng = np.random.default_rng(8)
    exact, offsets = None, {}
    for n in (7, 12):
        sites, cell = lattice(n)
        sigma, lam = 0.12, thermal_de_broglie_wavelength([26.98], 1500.0)[0]
        omega = np.sqrt(scipy.constants.k * 1500.0 / (26.98 * scipy.constants.atomic_mass * (sigma * 1e-10) ** 2))
        exact = 3 * R * (1 + np.log(scipy.constants.k * 1500.0 / (scipy.constants.hbar * omega)))
        values = [microstate_entropy(sites + rng.normal(0, sigma, sites.shape), sites + rng.normal(0, sigma, sites.shape),
                                     cell, lam, seed=k) for k in range(4)]
        offsets[n] = exact - np.mean(values)
    assert offsets[7] == pytest.approx(offsets[12], abs=0.6)
    assert offsets[12] == pytest.approx(CLASSICAL_ENTROPY_OFFSET, abs=0.6)


def test_fewer_atoms_than_one_group_raise():
    sites, cell = lattice(5)
    with pytest.raises(ValueError, match="fewer than one group"):
        microstate_entropy(sites, sites + 0.1, cell, 0.05)


def test_small_cells_code_as_many_groups_as_large_ones():
    rng = np.random.default_rng(14)
    sites, cell = lattice(7)  # 343 atoms, one group of 256 per pair
    frames = sites + rng.normal(0, 0.12, (12,) + sites.shape)
    out = trajectory_entropy(frames, cell, np.full(len(sites), 26.98), 1500.0, n_pairs=2, min_separation=2, dither=1,
                             min_groups=5)
    assert len(out["samples"]) == 5
