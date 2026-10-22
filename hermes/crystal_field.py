"""Crystal-field (Schottky) entropy of open-4f lanthanide ions, from measured level energies.

The 4f shell of a lanthanide ion is split by the crystal field into a few levels within some hundreds of cm^-1, which
fill as the temperature rises. Their entropy, heat capacity and free energy follow from the partition function of the
levels alone,

    Z = sum g_i exp(-E_i / k_B T),    S = R [ln Z + <E> / k_B T],    C = R Var(E) / (k_B T)^2,    G = -R T ln Z

per mole of ions, with energies measured from the ground level. Atomic motion carries none of this, so an MD or phonon
entropy misses it, and DFT does not resolve the levels. They are taken as input, from optical spectroscopy or inelastic
neutron scattering, never fitted to the thermodynamic data they are meant to predict. With every level of the ground J
multiplet given, S tends to R ln(2J+1) at high temperature (19.1 J/(K mol) for Nd3+); for Sm3+ and Eu3+, whose next
multiplets lie within about 1000 and 350 cm^-1, give those levels too.

The term is additive: it rides on the lattice of the host, taken to be unchanged by which levels are filled, and the
levels do not shift with temperature or volume, so C_p = C_v here. At 1000 K, -R T ln Z for Nd3+ lies between
-R T ln 2 (only the ground doublet filled) and -R T ln 10 (all five), -10 to -33 meV per atom of NdPO4.

The ground level's degeneracy is a choice. A Kramers doublet keeps R ln 2 down to the lowest temperatures, where
magnetic order or the hyperfine interaction releases it, usually below 2 K. A third-law entropy from calorimetry that
integrates through that anomaly, or adds R ln g_0 for it, includes the term; an excess heat capacity integrated from
above it does not. ``include_ground_degeneracy`` selects which.
"""

# Standard Library Imports
from fractions import Fraction

# Third-Party Library Imports
import numpy as np
import scipy.constants

GAS_CONSTANT = scipy.constants.R
# level energies in these units, times the factor, give E / k_B in K
TO_KELVIN = {
    "K": 1.0,
    "cm-1": scipy.constants.h * scipy.constants.c * 100.0 / scipy.constants.k,
    "meV": 1e-3 * scipy.constants.e / scipy.constants.k,
}
# number of 4f electrons of each trivalent lanthanide
FOUR_F_ELECTRONS = {"La": 0, "Ce": 1, "Pr": 2, "Nd": 3, "Pm": 4, "Sm": 5, "Eu": 6, "Gd": 7, "Tb": 8, "Dy": 9,
                    "Ho": 10, "Er": 11, "Tm": 12, "Yb": 13, "Lu": 14}


def hund_ground_j(n_electrons: int, l: int = 3) -> Fraction:
    """Total angular momentum J of the ground multiplet of an l^n shell by Hund's rules (l = 3 for 4f)."""
    if not 0 <= n_electrons <= 2 * (2 * l + 1):
        raise ValueError(f"an l = {l} shell holds 0 to {2 * (2 * l + 1)} electrons, not {n_electrons}")
    capacity = 2 * l + 1
    up, down = min(n_electrons, capacity), max(0, n_electrons - capacity)
    spin = Fraction(up - down, 2)
    # each spin channel fills m_l from +l downward
    orbital = abs(sum(range(l, l - up, -1)) + sum(range(l, l - down, -1)))
    return abs(orbital - spin) if n_electrons < capacity else orbital + spin


def free_ion_entropy(j) -> float:
    """R ln(2J+1), the high-temperature limit of the crystal-field entropy of a J multiplet, in J/(K mol of ions)."""
    return float(GAS_CONSTANT * np.log(2 * float(j) + 1))


def crystal_field_properties(
    energies,
    degeneracies,
    temperatures,
    unit: str = "cm-1",
    include_ground_degeneracy: bool = True,
    ions_per_formula: float = 1.0,
    atoms_per_formula: float = None,
) -> dict:
    """
    Entropy, heat capacity, enthalpy and Gibbs energy of a set of crystal-field levels.

    Args:
        energies: Level energies, any order; the lowest is taken as zero.
        degeneracies: Degeneracy of each level (2 for every level of a Kramers ion: Ce, Nd, Sm, Gd, Dy, Er, Yb).
        temperatures: Temperature or temperatures in K, all above zero.
        unit: Unit of the energies, ``"cm-1"``, ``"meV"`` or ``"K"``.
        include_ground_degeneracy: Count R ln g_0 of the ground level (see the module notes). If False the entropy
            starts from zero at 0 K and the Gibbs energy is -R T ln(Z / g_0).
        ions_per_formula: Lanthanide ions per formula unit.
        atoms_per_formula: Atoms per formula unit. If given, results are per mole of atoms
            (times ions_per_formula / atoms_per_formula); otherwise per mole of ions.

    Returns:
        ``{"entropy", "heat_capacity", "enthalpy", "gibbs_energy"}`` in J/(K mol) and J/mol, each of the shape of
        ``temperatures``. Enthalpy and Gibbs energy are measured from the ground level at 0 K.
    """
    if unit not in TO_KELVIN:
        raise ValueError(f"unit {unit!r}; use one of {sorted(TO_KELVIN)}")
    e = np.asarray(energies, dtype=float).ravel() * TO_KELVIN[unit]
    g = np.asarray(degeneracies, dtype=float).ravel()
    if e.shape != g.shape or e.size == 0:
        raise ValueError(f"{e.size} energies and {g.size} degeneracies")
    if np.any(g <= 0):
        raise ValueError("degeneracies must be positive")
    t = np.asarray(temperatures, dtype=float)
    if np.any(t <= 0):
        raise ValueError("temperatures must be above 0 K")
    e = e - e.min()
    x = e[None, :] / t.reshape(-1, 1)
    w = g * np.exp(-x)
    z = w.sum(axis=1)
    mean = (w * x).sum(axis=1) / z
    variance = (w * x**2).sum(axis=1) / z - mean**2
    log_z = np.log(z)
    if not include_ground_degeneracy:
        log_z = log_z - np.log(g[e == 0].sum())
    scale = ions_per_formula / atoms_per_formula if atoms_per_formula is not None else 1.0
    tt = t.reshape(-1)
    out = {
        "entropy": GAS_CONSTANT * (log_z + mean),
        "heat_capacity": GAS_CONSTANT * variance,
        "enthalpy": GAS_CONSTANT * tt * mean,
        "gibbs_energy": -GAS_CONSTANT * tt * log_z,
    }
    return {k: (float(v[0]) if t.ndim == 0 else v.reshape(t.shape)) * scale for k, v in out.items()}
