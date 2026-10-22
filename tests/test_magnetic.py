import numpy as np
import pytest
from scipy.integrate import quad

from hermes.magnetic import (
    GAS_CONSTANT,
    STRUCTURE_FACTOR_BCC,
    STRUCTURE_FACTOR_OTHER,
    magnetic_properties,
    paramagnetic_entropy,
    spin_only_moment,
)

R = GAS_CONSTANT
REFERENCES = ["ordered", "disordered", "calphad"]


def hillert_jarl_g(tau, p):
    # the published form, with tau^-1 below T*, written independently of the module's rearrangement
    d = 518 / 1125 + 11692 / 15975 * (1 / p - 1)
    if tau <= 1:
        return 1 - (79 / (140 * p * tau) + 474 / 497 * (1 / p - 1) * (tau**3 / 6 + tau**9 / 135 + tau**15 / 600)) / d
    return -(tau**-5 / 10 + tau**-15 / 315 + tau**-25 / 1500) / d


def test_structure_factors_are_indens():
    assert (STRUCTURE_FACTOR_BCC, STRUCTURE_FACTOR_OTHER) == (0.40, 0.28)


@pytest.mark.parametrize("p", [STRUCTURE_FACTOR_BCC, STRUCTURE_FACTOR_OTHER])
@pytest.mark.parametrize("tau", [0.1, 0.5, 0.99, 1.0, 1.01, 2.0, 5.0])
def test_gibbs_energy_against_the_published_form(p, tau):
    t_star, beta = 700.0, 1.7
    t = tau * t_star
    exact = 8.314462618 * t * np.log(beta + 1) * hillert_jarl_g(tau, p)
    g = magnetic_properties(t, t_star, beta, p, reference="calphad")["gibbs_energy"]
    assert g == pytest.approx(exact, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("p", [STRUCTURE_FACTOR_BCC, STRUCTURE_FACTOR_OTHER])
def test_entropy_and_heat_capacity_are_derivatives_of_the_gibbs_energy(reference, p):
    t = np.concatenate([np.linspace(30.0, 590.0, 60), np.linspace(610.0, 3000.0, 60)])
    kw = dict(critical_temperature=600.0, moment=2.2, structure_factor=p, reference=reference)
    out = magnetic_properties(t, **kw)
    h = 1e-3
    up, down = magnetic_properties(t + h, **kw), magnetic_properties(t - h, **kw)
    assert -(up["gibbs_energy"] - down["gibbs_energy"]) / (2 * h) == pytest.approx(out["entropy"], rel=1e-6, abs=1e-8)
    assert t * (up["entropy"] - down["entropy"]) / (2 * h) == pytest.approx(out["heat_capacity"], rel=1e-5, abs=1e-8)
    assert out["gibbs_energy"] == pytest.approx(out["enthalpy"] - t * out["entropy"], rel=1e-10, abs=1e-8)


def test_gibbs_energy_and_entropy_are_continuous_at_the_critical_temperature():
    kw = dict(critical_temperature=500.0, moment=3.0, structure_factor=0.28, reference="ordered")
    below, above = magnetic_properties(500.0 - 1e-7, **kw), magnetic_properties(500.0 + 1e-7, **kw)
    assert below["gibbs_energy"] == pytest.approx(above["gibbs_energy"], abs=1e-4)
    assert below["entropy"] == pytest.approx(above["entropy"], abs=1e-5)
    assert below["enthalpy"] == pytest.approx(above["enthalpy"], abs=1e-3)


@pytest.mark.parametrize("p", [STRUCTURE_FACTOR_BCC, STRUCTURE_FACTOR_OTHER])
def test_ordering_releases_r_ln_beta_plus_1_and_p_of_the_enthalpy_lies_above_t_star(p):
    t_star, beta = 400.0, 2.0

    def c(t):
        return magnetic_properties(t, t_star, beta, p, reference="ordered")["heat_capacity"]

    def integral(f, lo, hi):
        return quad(f, lo, hi, limit=500, points=[2 * t_star, 10 * t_star] if lo == t_star else None)[0]

    entropy = integral(lambda t: c(t) / t, 0.0, t_star) + integral(lambda t: c(t) / t, t_star, 1e6)
    assert entropy == pytest.approx(R * np.log(beta + 1), rel=1e-9)
    below, above = integral(c, 0.0, t_star), integral(c, t_star, 1e6)
    assert above / (below + above) == pytest.approx(p, rel=1e-9)
    g0 = magnetic_properties(0.0, t_star, beta, p, reference="calphad")["gibbs_energy"]
    assert -g0 == pytest.approx(below + above, rel=1e-9)


def test_the_three_references():
    kw = dict(critical_temperature=525.0, moment=4.6, structure_factor=0.28)
    t = np.array([0.0, 100.0, 525.0, 1000.0, 5.0e5])
    o = magnetic_properties(t, reference="ordered", **kw)
    d = magnetic_properties(t, reference="disordered", **kw)
    c = magnetic_properties(t, reference="calphad", **kw)
    r_ln = R * np.log(5.6)
    # ordered: nothing at 0 K, the paramagnetic entropy at high temperature
    for key in ("gibbs_energy", "enthalpy", "entropy"):
        assert o[key][0] == pytest.approx(0.0, abs=1e-9)
    assert o["entropy"][-1] == pytest.approx(r_ln, rel=1e-9)
    # disordered: the ordering enthalpy at 0 K, nothing left of it at high temperature
    assert d["enthalpy"][0] == pytest.approx(c["gibbs_energy"][0])
    assert d["enthalpy"][0] < 0
    assert d["enthalpy"][-1] == pytest.approx(0.0, abs=1e-6)
    assert d["entropy"] == pytest.approx(o["entropy"])
    assert d["gibbs_energy"] - o["gibbs_energy"] == pytest.approx(np.full(5, c["gibbs_energy"][0]))
    # calphad: zero for the paramagnet, -R ln(beta + 1) of entropy at 0 K
    assert c["entropy"][0] == pytest.approx(-r_ln)
    assert c["gibbs_energy"] - d["gibbs_energy"] == pytest.approx(r_ln * t)
    for out in (o, d):
        assert out["heat_capacity"] == pytest.approx(c["heat_capacity"])


def test_spin_only_paramagnetic_limit():
    # Mn2+, S = 5/2: beta = 5 and R ln 6
    assert spin_only_moment(2.5) == 5.0
    assert paramagnetic_entropy(2.5) == pytest.approx(8.314462618 * np.log(6))
    s = magnetic_properties(1e5, 118.0, spin_only_moment(2.5), 0.28, reference="ordered")["entropy"]
    assert s == pytest.approx(paramagnetic_entropy(2.5), rel=1e-9)


def test_per_mole_of_atoms():
    kw = dict(critical_temperature=950.0, moment=5.0, structure_factor=0.28, reference="ordered")
    per_ion = magnetic_properties(600.0, **kw)
    per_atom = magnetic_properties(600.0, ions_per_formula=2, atoms_per_formula=5, **kw)  # Fe2O3
    for key in per_ion:
        assert per_atom[key] == pytest.approx(per_ion[key] * 2 / 5)


def test_reference_must_be_chosen():
    with pytest.raises(TypeError):
        magnetic_properties(300.0, 500.0, 2.0, 0.28)
    with pytest.raises(ValueError):
        magnetic_properties(300.0, 500.0, 2.0, 0.28, reference="paramagnetic")


@pytest.mark.parametrize(("t_star", "beta", "p", "t"), [(-525.0, 2.0, 0.28, 300.0), (500.0, -1.0, 0.28, 300.0),
                                                         (500.0, 2.0, 1.0, 300.0), (500.0, 2.0, 0.28, -1.0)])
def test_bad_input_is_refused(t_star, beta, p, t):
    with pytest.raises(ValueError):
        magnetic_properties(t, t_star, beta, p, reference="ordered")


TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT FE BCC_A2 55.847 4489 27.28 !
TYPE_DEFINITION % SEQ * !
TYPE_DEFINITION A GES A_P_D BCC_A2 MAGNETIC -1.0 0.4 !
TYPE_DEFINITION B GES A_P_D FCC_A1 MAGNETIC -3.0 0.28 !
PHASE BCC_A2 %A 2 1 3 !
CONSTITUENT BCC_A2 :FE:VA: !
PHASE FCC_A1 %B 2 1 1 !
CONSTITUENT FCC_A1 :FE:VA: !
PARAMETER G(BCC_A2,FE:VA;0) 1 0; 6000 N !
PARAMETER TC(BCC_A2,FE:VA;0) 1 1043; 6000 N !
PARAMETER BMAGN(BCC_A2,FE:VA;0) 1 2.22; 6000 N !
PARAMETER G(FCC_A1,FE:VA;0) 1 0; 6000 N !
PARAMETER TC(FCC_A1,FE:VA;0) 1 -201; 6000 N !
PARAMETER BMAGN(FCC_A1,FE:VA;0) 1 -2.1; 6000 N !
"""


@pytest.mark.parametrize(("phase", "t_star", "beta", "p"), [("BCC_A2", 1043.0, 2.22, 0.4), ("FCC_A1", 67.0, 0.7, 0.28)])
def test_pycalphad_magnetic_model_agrees(phase, t_star, beta, p):
    # fcc Fe: an antiferromagnet stored as TC = -3 T_N and BMAGN = -3 beta. pycalphad 0.11 uses R = 8.3145, and
    # returns zero exactly at T = TC (its piecewise conditions are strict on both sides), so T* itself is left out.
    pycalphad = pytest.importorskip("pycalphad")
    db = pycalphad.Database(TDB)
    t = np.array([50.0, 300.0, 900.0, 1042.0, 1044.0, 1200.0, 2500.0])
    out = magnetic_properties(t, t_star, beta, p, reference="calphad")
    for prop, key in [("GM", "gibbs_energy"), ("SM", "entropy"), ("HM", "enthalpy"), ("CPM", "heat_capacity")]:
        res = pycalphad.calculate(db, ["FE", "VA"], phase, T=t, P=101325, N=1, output=prop)
        expected = out[key] * 8.3145 / R
        assert np.squeeze(getattr(res, prop).values) == pytest.approx(expected, rel=1e-9, abs=1e-12), prop
