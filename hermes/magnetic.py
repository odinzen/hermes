"""Magnetic entropy and Gibbs energy of transition-metal ions, by the Inden / Hillert-Jarl model used in CALPHAD.

Spins disorder over a range of temperature around the ordering temperature T* (Curie or Neel), releasing the entropy
R ln(beta + 1) per mole of magnetic ions, beta being the mean moment in Bohr magnetons; for a spin-only ion,
beta = 2S and the limit is the paramagnetic R ln(2S + 1). Hillert and Jarl [1] wrote Inden's heat capacity [2] as a
Gibbs energy that is zero for the fully disordered paramagnet,

    G_HJ = R T ln(beta + 1) g(tau),    tau = T / T*

with g a polynomial in tau below T* and in 1/tau above it, and the structure factor p the fraction of the ordering
enthalpy absorbed above T* (0.40 for bcc, 0.28 for other structures). At 0 K, G_HJ is the ordering enthalpy and the
entropy is -R ln(beta + 1), the disorder that the paramagnetic reference holds and the ordered state has lost.

What to add depends on the energy it is added to. An MLIP or DFT energy at 0 K of a magnetically ordered cell already
holds the ordering energy, and MD gives the motion of the atoms but no spin entropy; adding G_HJ there would count the
ordering enthalpy twice and set the entropy below zero. ``reference`` says which energy the term is added to:

- ``"ordered"``: an ordered (ferro-, ferri- or antiferromagnetic) cell at 0 K. Adds G_HJ - G_HJ(0) - R T ln(beta + 1),
  zero at 0 K: the entropy from 0 to R ln(beta + 1) and the enthalpy taken up by disordering, no ordering enthalpy.
- ``"disordered"``: a spin-disordered cell (disordered local moments, special quasirandom spins) at 0 K with no spin
  entropy. Adds G_HJ - R T ln(beta + 1): the same entropy, and the ordering enthalpy G_HJ(0) at low temperature.
- ``"calphad"``: G_HJ itself, as a CALPHAD database adds it to a paramagnetic lattice function that already holds
  R ln(beta + 1). For comparison with assessed databases, not for adding to MD.

The model knows only T*, beta and p. It assumes the ordered state of the 0 K calculation is the true ground state (a
ferromagnetic cell for an antiferromagnet is off by their energy difference, which this does not correct) and spins
alone: an orbital moment, as in Co2+ and high-spin Fe2+ on octahedral sites, carries entropy beyond R ln(2S + 1).

1. M. Hillert and M. Jarl, "A model for alloying in ferromagnetic metals", Calphad 2, 227 (1978).
   https://doi.org/10.1016/0364-5916(78)90011-1
2. G. Inden, "The role of magnetism in the calculation of phase diagrams", Physica B+C 103, 82 (1981).
   https://doi.org/10.1016/0378-4363(81)91004-4
"""

# Standard Library Imports
from typing import Literal

# Third-Party Library Imports
import numpy as np
import scipy.constants

GAS_CONSTANT = scipy.constants.R
STRUCTURE_FACTOR_BCC = 0.40
STRUCTURE_FACTOR_OTHER = 0.28

Reference = Literal["ordered", "disordered", "calphad"]


def paramagnetic_entropy(spin) -> float:
    """R ln(2S + 1), the entropy of disordered spins S, in J/(K mol of ions)."""
    return float(GAS_CONSTANT * np.log(2 * float(spin) + 1))


def spin_only_moment(spin) -> float:
    """Moment 2S in Bohr magnetons (g = 2), for which R ln(beta + 1) is the paramagnetic limit."""
    return 2.0 * float(spin)


def _hillert_jarl(t: np.ndarray, t_star: float, p: float) -> tuple:
    # g, -dG/dT and T dS/dT per R ln(beta + 1), from Hillert and Jarl's polynomials and their exact derivatives
    d = 518.0 / 1125.0 + 11692.0 / 15975.0 * (1.0 / p - 1.0)
    k = 474.0 / 497.0 * (1.0 / p - 1.0)
    tau = t / t_star
    below = tau <= 1.0
    lo = np.where(below, tau, 0.0)
    hi = np.where(below, 1.0, tau)
    # below T* the 1/tau term is written as T* so that 0 K needs no division by zero
    g_below = t - (79.0 * t_star / (140.0 * p) + t * k * (lo**3 / 6 + lo**9 / 135 + lo**15 / 600)) / d
    s_below = -(1.0 - k / d * (2 * lo**3 / 3 + 2 * lo**9 / 27 + 2 * lo**15 / 75))
    c_below = k / d * (2 * lo**3 + 2 * lo**9 / 3 + 2 * lo**15 / 5)
    g_above = -t * (hi**-5 / 10 + hi**-15 / 315 + hi**-25 / 1500) / d
    s_above = -(2 * hi**-5 / 5 + 2 * hi**-15 / 45 + 2 * hi**-25 / 125) / d
    c_above = (2 * hi**-5 + 2 * hi**-15 / 3 + 2 * hi**-25 / 5) / d
    return (np.where(below, g_below, g_above), np.where(below, s_below, s_above), np.where(below, c_below, c_above),
            -79.0 * t_star / (140.0 * p * d))


def magnetic_properties(
    temperatures,
    critical_temperature: float,
    moment: float,
    structure_factor: float,
    *,
    reference: Reference,
    ions_per_formula: float = 1.0,
    atoms_per_formula: float = None,
) -> dict:
    """
    Magnetic entropy, heat capacity, enthalpy and Gibbs energy of one kind of magnetic ion.

    Args:
        temperatures: Temperature or temperatures in K.
        critical_temperature: Curie or Neel temperature T* in K. A CALPHAD database stores an antiferromagnet's as a
            negative TC times its antiferromagnetic factor (-1 for bcc, -3 for fcc); pass the physical T_N.
        moment: Mean moment beta per magnetic ion, in Bohr magnetons (``spin_only_moment(S)`` for a spin-only ion).
        structure_factor: p, the fraction of the ordering enthalpy above T*: ``STRUCTURE_FACTOR_BCC`` or
            ``STRUCTURE_FACTOR_OTHER``.
        reference: The energy the term is added to, ``"ordered"``, ``"disordered"`` or ``"calphad"`` (see the
            module notes). There is no default: the wrong choice double counts the ordering enthalpy.
        ions_per_formula: Magnetic ions per formula unit.
        atoms_per_formula: Atoms per formula unit. If given, results are per mole of atoms
            (times ions_per_formula / atoms_per_formula); otherwise per mole of magnetic ions.

    Returns:
        ``{"entropy", "heat_capacity", "enthalpy", "gibbs_energy"}`` in J/(K mol) and J/mol, each of the shape of
        ``temperatures``.
    """
    if reference not in ("ordered", "disordered", "calphad"):
        raise ValueError(f"reference {reference!r}; use 'ordered', 'disordered' or 'calphad'")
    if critical_temperature <= 0:
        raise ValueError("critical temperature must be positive; pass T_N, not a CALPHAD TC with its factor")
    if moment < 0 or not 0 < structure_factor < 1:
        raise ValueError("moment must be >= 0 and the structure factor between 0 and 1")
    t = np.asarray(temperatures, dtype=float)
    if np.any(t < 0):
        raise ValueError("temperatures must not be negative")
    tt = t.reshape(-1)
    g, s, c, g0 = _hillert_jarl(tt, float(critical_temperature), float(structure_factor))
    r_ln = GAS_CONSTANT * np.log(moment + 1.0)
    gibbs, entropy, heat_capacity = r_ln * g, r_ln * s, r_ln * c
    if reference != "calphad":
        entropy = entropy + r_ln
        gibbs = gibbs - r_ln * tt
    if reference == "ordered":
        gibbs = gibbs - r_ln * g0
    scale = ions_per_formula / atoms_per_formula if atoms_per_formula is not None else 1.0
    out = {"entropy": entropy, "heat_capacity": heat_capacity, "enthalpy": gibbs + tt * entropy, "gibbs_energy": gibbs}
    return {k: (float(v[0]) if t.ndim == 0 else v.reshape(t.shape)) * scale for k, v in out.items()}
