from fractions import Fraction

import numpy as np
import pytest

from hermes.crystal_field import (
    FOUR_F_ELECTRONS,
    GAS_CONSTANT,
    crystal_field_properties,
    free_ion_entropy,
    hund_ground_j,
)

R = GAS_CONSTANT


def test_two_level_schottky_anomaly():
    delta = 100.0  # K
    t = np.linspace(5.0, 400.0, 400)
    out = crystal_field_properties([0.0, delta], [1, 1], t, unit="K")
    x = delta / t
    exact_c = R * x**2 * np.exp(-x) / (1 + np.exp(-x)) ** 2
    exact_s = R * (np.log(1 + np.exp(-x)) + x * np.exp(-x) / (1 + np.exp(-x)))
    assert out["heat_capacity"] == pytest.approx(exact_c, rel=1e-10)
    assert out["entropy"] == pytest.approx(exact_s, rel=1e-10)
    # the textbook peak: C_max = 0.4392 R at T = 0.4168 delta
    fine = crystal_field_properties([0.0, delta], [1, 1], np.linspace(40.0, 43.0, 3001), unit="K")
    peak = np.argmax(fine["heat_capacity"])
    assert fine["heat_capacity"][peak] == pytest.approx(0.4392 * R, abs=1e-4 * R)
    assert np.linspace(40.0, 43.0, 3001)[peak] == pytest.approx(41.68, abs=0.01)


def test_degenerate_upper_level_moves_the_peak():
    delta, t = 100.0, np.linspace(5.0, 400.0, 400)
    out = crystal_field_properties([0.0, delta], [2, 6], t, unit="K")
    x, r = delta / t, 3.0
    exact = R * r * x**2 * np.exp(-x) / (1 + r * np.exp(-x)) ** 2
    assert out["heat_capacity"] == pytest.approx(exact, rel=1e-10)


def test_high_temperature_limit_is_r_ln_2j_plus_1():
    # Nd3+: J = 9/2, five Kramers doublets
    levels = [0.0, 120.0, 230.0, 300.0, 500.0]
    s = crystal_field_properties(levels, [2] * 5, 1e7)["entropy"]
    assert s == pytest.approx(free_ion_entropy(Fraction(9, 2)), rel=1e-5)
    assert free_ion_entropy(4.5) == pytest.approx(R * np.log(10))


def test_low_temperature_limit_is_the_ground_degeneracy():
    levels, g = [0.0, 300.0, 600.0], [2, 2, 2]
    full = crystal_field_properties(levels, g, 5.0)
    bare = crystal_field_properties(levels, g, 5.0, include_ground_degeneracy=False)
    assert full["entropy"] == pytest.approx(R * np.log(2), rel=1e-12)
    assert bare["entropy"] == pytest.approx(0.0, abs=1e-12)
    hot = crystal_field_properties(levels, g, 800.0)
    hot_bare = crystal_field_properties(levels, g, 800.0, include_ground_degeneracy=False)
    assert hot["entropy"] - hot_bare["entropy"] == pytest.approx(R * np.log(2))
    assert hot["gibbs_energy"] - hot_bare["gibbs_energy"] == pytest.approx(-R * 800.0 * np.log(2))


def test_gibbs_energy_is_consistent_with_entropy_and_enthalpy():
    levels, g = [0.0, 80.0, 250.0, 410.0], [2, 2, 2, 2]
    t = np.linspace(20.0, 1500.0, 300)
    out = crystal_field_properties(levels, g, t)
    assert out["gibbs_energy"] == pytest.approx(out["enthalpy"] - t * out["entropy"], rel=1e-10, abs=1e-8)
    h = 1e-3
    up, down = crystal_field_properties(levels, g, t + h), crystal_field_properties(levels, g, t - h)
    assert -(up["gibbs_energy"] - down["gibbs_energy"]) / (2 * h) == pytest.approx(out["entropy"], rel=1e-6)
    assert t * (up["entropy"] - down["entropy"]) / (2 * h) == pytest.approx(out["heat_capacity"], rel=1e-5, abs=1e-9)
    assert (up["enthalpy"] - down["enthalpy"]) / (2 * h) == pytest.approx(out["heat_capacity"], rel=1e-5, abs=1e-9)


def test_entropy_from_first_principles_constants():
    # 100 cm^-1 and 10 meV doublets, the conversions written out from the SI constants
    to_k_cm = 6.62607015e-34 * 299792458.0 * 100.0 / 1.380649e-23
    to_k_mev = 1e-3 * 1.602176634e-19 / 1.380649e-23
    assert to_k_cm == pytest.approx(1.438776877, rel=1e-9)
    t = 150.0
    for energy, unit, factor in [(100.0, "cm-1", to_k_cm), (10.0, "meV", to_k_mev)]:
        x = energy * factor / t
        z = 2 + 2 * np.exp(-x)
        exact = 8.314462618 * (np.log(z) + 2 * x * np.exp(-x) / z)
        assert crystal_field_properties([0.0, energy], [2, 2], t, unit=unit)["entropy"] == pytest.approx(exact, rel=1e-9)


def test_per_mole_of_atoms():
    levels, g, t = [0.0, 150.0], [2, 2], 600.0
    per_ion = crystal_field_properties(levels, g, t)
    per_atom = crystal_field_properties(levels, g, t, ions_per_formula=1, atoms_per_formula=6)  # NdPO4
    for key in per_ion:
        assert per_atom[key] == pytest.approx(per_ion[key] / 6)
    two = crystal_field_properties(levels, g, t, ions_per_formula=2, atoms_per_formula=5)
    assert two["entropy"] == pytest.approx(per_ion["entropy"] * 2 / 5)


def test_level_order_and_offset_do_not_matter():
    a = crystal_field_properties([0.0, 100.0, 300.0], [2, 4, 2], 250.0)
    b = crystal_field_properties([400.0, 100.0, 200.0], [2, 2, 4], 250.0)
    for key in a:
        assert a[key] == pytest.approx(b[key])


def test_scalar_and_array_temperatures():
    out = crystal_field_properties([0.0, 100.0], [1, 1], 300.0)
    assert isinstance(out["entropy"], float)
    arr = crystal_field_properties([0.0, 100.0], [1, 1], [[300.0, 400.0]])
    assert arr["entropy"].shape == (1, 2)
    assert arr["entropy"][0, 0] == pytest.approx(out["entropy"])


@pytest.mark.parametrize(
    ("energies", "degeneracies", "temperature", "unit"),
    [([0, 1], [1], 300.0, "cm-1"), ([0, 1], [1, 0], 300.0, "cm-1"), ([0, 1], [1, 1], 0.0, "cm-1"),
     ([0, 1], [1, 1], 300.0, "eV")],
)
def test_bad_input_is_refused(energies, degeneracies, temperature, unit):
    with pytest.raises(ValueError):
        crystal_field_properties(energies, degeneracies, temperature, unit=unit)


def test_hund_rules_give_the_lanthanide_ground_multiplets():
    # J of the trivalent ions from the free-ion term symbols (2F5/2, 3H4, 4I9/2, ... 2F7/2)
    known = {"La": 0, "Ce": "5/2", "Pr": 4, "Nd": "9/2", "Pm": 4, "Sm": "5/2", "Eu": 0, "Gd": "7/2", "Tb": 6,
             "Dy": "15/2", "Ho": 8, "Er": "15/2", "Tm": 6, "Yb": "7/2", "Lu": 0}
    for element, j in known.items():
        assert hund_ground_j(FOUR_F_ELECTRONS[element]) == Fraction(j), element
    with pytest.raises(ValueError):
        hund_ground_j(15)
