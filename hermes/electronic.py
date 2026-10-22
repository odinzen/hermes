"""Electronic entropy from an electronic density of states.

Electrons excited across the Fermi level carry the entropy of independent fermions,

    S_el = -k_B int g(E) [f ln f + (1 - f) ln(1 - f)] dE,    f = 1 / (exp((E - mu) / k_B T) + 1)

with g the density of states of the cell (both spins) and the chemical potential mu set at each temperature so that
the electron count stays that of the cell at 0 K. For a metal at low temperature this is the Sommerfeld gamma T,
gamma = (pi^2 / 3) k_B^2 g(E_F); for an insulator it is exp(-E_gap / 2 k_B T) small and vanishes for a wide gap. The
free energy at fixed volume follows as F_el(T) - F_el(0) = -int_0^T S_el dT'.

The density of states is that of one fixed structure, usually the DFT ground state (the Materials Project publishes
one per material: ``dos.energies``, ``dos.get_densities()`` summed over spins, ``dos.efermi``, per cell of
``len(dos.structure)`` atoms). Thermal expansion, phonons and magnetic disorder that broaden or shift it are left
out, and a band gap that semilocal DFT closes, as in many transition-metal oxides without +U, gives a spurious metallic
entropy. Where the window of the density of states cuts through a band it must reach well beyond mu; it is
refused otherwise.
"""

# Standard Library Imports
import scipy.constants

# Third-Party Library Imports
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

GAS_CONSTANT = scipy.constants.R
BOLTZMANN_EV = scipy.constants.k / scipy.constants.e  # eV/K
# how far the window must reach past mu, in k_B T; the Fermi tail there is below e^-30
WINDOW_KT = 30.0
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(4)


def _clean(energies, dos) -> tuple:
    e = np.asarray(energies, dtype=float)
    g = np.asarray(dos, dtype=float)
    if g.ndim == 2:  # spin up and down
        g = g.sum(axis=0)
    if e.ndim != 1 or g.shape != e.shape or e.size < 2 or np.any(np.diff(e) <= 0):
        raise ValueError("energies must increase, with one density of states value each (or one per spin)")
    if np.any(g < 0):
        raise ValueError("the density of states must not be negative")
    return e, g


def _states_below(e: np.ndarray, g: np.ndarray, x: float) -> float:
    # exact integral of the piecewise-linear density of states from the bottom of the window to x
    x = min(max(x, e[0]), e[-1])
    below = e < x
    pts = np.append(e[below], x)
    return float(np.trapezoid(np.interp(pts, e, g), pts))


def _near_mu(e: np.ndarray, g: np.ndarray, mu: float, kt: float) -> tuple:
    # quadrature over the stretch within WINDOW_KT k_B T of mu, where f is neither 0 nor 1: each interval of the grid,
    # on which g is linear, is cut into pieces no longer than kT / 2 with a 4-point Gauss-Legendre rule on each, so
    # that the kinks of g are respected and the result varies smoothly with mu and T
    lo, hi = max(mu - WINDOW_KT * kt, e[0]), min(mu + WINDOW_KT * kt, e[-1])
    if hi <= lo:
        return np.array([lo]), np.zeros(1), np.interp([lo], e, g)
    edges = np.concatenate([[lo], e[(e > lo) & (e < hi)], [hi]])
    pieces = np.maximum(1, np.ceil(2 * np.diff(edges) / kt)).astype(int)
    width = np.repeat(np.diff(edges) / pieces, pieces)
    index = np.arange(pieces.sum()) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    start = np.repeat(edges[:-1], pieces) + index * width
    x = (start[:, None] + 0.5 * width[:, None] * (_GAUSS_X + 1.0)).ravel()
    w = (0.5 * width[:, None] * _GAUSS_W).ravel()
    return x, w, np.interp(x, e, g)


def _occupied(e: np.ndarray, g: np.ndarray, mu: float, kt: float) -> float:
    x, w, gx = _near_mu(e, g, mu, kt)
    return _states_below(e, g, max(mu - WINDOW_KT * kt, e[0])) + float(np.sum(w * gx * expit(-(x - mu) / kt)))


def electron_count(energies, dos, fermi_energy: float) -> float:
    """States below ``fermi_energy`` at 0 K, the integral of the (linearly interpolated) density of states."""
    e, g = _clean(energies, dos)
    if not e[0] <= fermi_energy <= e[-1]:
        raise ValueError("the Fermi energy is outside the energy window")
    return _states_below(e, g, fermi_energy)


def chemical_potential(energies, dos, n_electrons: float, temperature: float) -> float:
    """Chemical potential in eV that puts ``n_electrons`` in the density of states at ``temperature`` (K, > 0)."""
    e, g = _clean(energies, dos)
    if temperature <= 0:
        raise ValueError("temperature must be above 0 K")
    kt = BOLTZMANN_EV * temperature
    total = _states_below(e, g, e[-1])
    if not 0 < n_electrons < total:
        raise ValueError(f"{n_electrons} electrons do not fit in a window holding {total:.6g} states")
    mu = brentq(lambda m: _occupied(e, g, m, kt) - n_electrons, e[0] - WINDOW_KT * kt, e[-1] + WINDOW_KT * kt,
                xtol=1e-12 * max(1.0, np.abs(e).max()))
    # where the window cuts through a band, states beyond it would take part; a band edge (no states) is harmless
    if (g[0] > 0 and mu - e[0] < WINDOW_KT * kt) or (g[-1] > 0 and e[-1] - mu < WINDOW_KT * kt):
        raise ValueError(f"mu = {mu:.4g} eV is within {WINDOW_KT:g} k_B T of a band cut off by the window "
                         f"at {temperature:g} K; widen the window")
    return float(mu)


def _entropy_per_cell(e, g, n_electrons, temperature) -> float:
    # in units of k_B
    kt = BOLTZMANN_EV * temperature
    mu = chemical_potential(e, g, n_electrons, temperature)
    x, w, gx = _near_mu(e, g, mu, kt)
    y = np.abs(x - mu) / kt
    # -[f ln f + (1 - f) ln(1 - f)] is even in y; written for |y| it neither overflows nor cancels
    return float(np.sum(w * gx * (np.log1p(np.exp(-y)) + y * expit(-y))))


def electronic_entropy(
    energies,
    dos,
    temperatures,
    *,
    atoms: int,
    n_electrons: float = None,
    fermi_energy: float = None,
    per_cell: bool = False,
):
    """
    Electronic entropy of a density of states.

    Args:
        energies: Energies in eV, increasing.
        dos: Density of states of the cell in states/eV, both spins: shape ``(n,)``, or ``(2, n)`` for spin up and
            down, which are summed.
        temperatures: Temperature or temperatures in K. At 0 K the entropy is zero.
        atoms: Atoms in the cell the density of states is for.
        n_electrons: Electrons in the window. Give this or ``fermi_energy``.
        fermi_energy: Fermi energy in eV at 0 K; the electron count is the states below it.
        per_cell: If True, return eV/K per cell (the unit of ``configuration_from_md``'s ``electronic_entropies``)
            instead of J/(K mol of atoms).

    Returns:
        The entropy, of the shape of ``temperatures``.
    """
    e, g = _clean(energies, dos)
    if (n_electrons is None) == (fermi_energy is None):
        raise ValueError("give either n_electrons or fermi_energy")
    if n_electrons is None:
        n_electrons = electron_count(e, g, fermi_energy)
    t = np.asarray(temperatures, dtype=float)
    if np.any(t < 0):
        raise ValueError("temperatures must not be negative")
    s = np.array([_entropy_per_cell(e, g, n_electrons, ti) if ti > 0 else 0.0 for ti in t.reshape(-1)])
    s = s * BOLTZMANN_EV if per_cell else s * GAS_CONSTANT / atoms
    return float(s[0]) if t.ndim == 0 else s.reshape(t.shape)


def electronic_free_energy(energies, dos, temperatures, *, atoms: int, n_electrons: float = None,
                           fermi_energy: float = None, points: int = 24):
    """
    F_el(T) - F_el(0) = -int_0^T S_el dT' at fixed volume, in J/(mol of atoms), by Gauss-Legendre quadrature.

    Arguments as for ``electronic_entropy``; ``points`` is the number of quadrature points per temperature.
    """
    t = np.asarray(temperatures, dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(points)
    out = []
    for ti in t.reshape(-1):
        if ti <= 0:
            out.append(0.0)
            continue
        s = electronic_entropy(energies, dos, 0.5 * ti * (nodes + 1.0), atoms=atoms, n_electrons=n_electrons,
                               fermi_energy=fermi_energy)
        out.append(-0.5 * ti * float(np.dot(weights, s)))
    out = np.array(out)
    return float(out[0]) if t.ndim == 0 else out.reshape(t.shape)
