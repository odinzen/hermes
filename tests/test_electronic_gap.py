import numpy as np
import pytest

from hermes.electronic import electron_count, electronic_entropy, electronic_free_energy


def gapped(gap, width=4.0, points=16001):
    """Two flat bands of 2 states/eV each, ``width`` eV wide, with ``gap`` eV between them; the lower one full."""
    e = np.linspace(-width - gap / 2 - 1, width + gap / 2 + 1, points)
    g = np.where((np.abs(e) >= gap / 2) & (np.abs(e) <= gap / 2 + width), 2.0, 0.0)
    return e, g, electron_count(e, g, 0.0)  # the Fermi energy mid-gap fills the lower band exactly


def test_a_gapless_density_of_states_opened_to_a_wide_gap_has_no_electronic_entropy():
    # a metal-like density of states from DFT for what is really an insulator with a 3 eV gap
    e = np.linspace(-6, 6, 2401)
    g = np.full_like(e, 2.0)
    metal = electronic_entropy(e, g, 1000.0, atoms=2, fermi_energy=0.0)
    opened = electronic_entropy(e, g, 1000.0, atoms=2, fermi_energy=0.0, band_gap=3.0)
    assert metal > 1.0
    assert opened < 1e-4  # exp(-E_gap / 2 k_B T) small


def test_opening_a_gap_matches_a_density_of_states_built_with_that_gap():
    e, g, n = gapped(0.4)
    target_e, target_g, target_n = gapped(1.2)
    t = np.array([600.0, 1500.0, 3000.0])
    got = electronic_entropy(e, g, t, atoms=4, n_electrons=n, band_gap=1.2)
    want = electronic_entropy(target_e, target_g, t, atoms=4, n_electrons=target_n)
    # equal but for where the band edges fall on the two grids
    assert got == pytest.approx(want, rel=1e-2)


def test_a_larger_gap_is_left_alone():
    e, g, n = gapped(2.0)
    t = np.array([1500.0, 3000.0])
    assert electronic_entropy(e, g, t, atoms=4, n_electrons=n, band_gap=1.0) == pytest.approx(
        electronic_entropy(e, g, t, atoms=4, n_electrons=n), rel=1e-12)


def test_opening_the_gap_keeps_the_electron_count_and_reaches_the_free_energy():
    e = np.linspace(-6, 6, 2401)
    g = 1.0 + 0.5 * np.cos(e)
    n = electron_count(e, g, 0.3)
    f_metal = electronic_free_energy(e, g, 1200.0, atoms=2, n_electrons=n)
    f_opened = electronic_free_energy(e, g, 1200.0, atoms=2, n_electrons=n, band_gap=2.5)
    assert f_metal < -1.0
    assert abs(f_opened) < 0.05


def test_a_negative_gap_is_refused():
    e, g, n = gapped(0.5)
    with pytest.raises(ValueError, match="band_gap"):
        electronic_entropy(e, g, 300.0, atoms=4, n_electrons=n, band_gap=-0.1)
