"""Excess mixing entropy of a binary liquid from its composition fluctuations.

Pair-distribution entropies miss the chemical short-range order that sets the excess mixing entropy
(:mod:`hermes.liquid_entropy`). This module takes it from thermodynamics instead. The long-wavelength
limit of the Bhatia-Thornton concentration-concentration structure factor fixes the curvature of
the Gibbs energy of mixing::

    S_cc(0) = R T / (d^2 G_m / d x^2)_{T,P}
    S_cc(k) = < | x_B rho_A(k) - x_A rho_B(k) |^2 > / N,    rho_a(k) = sum_{j in a} exp(i k.r_j)

(A. B. Bhatia and D. E. Thornton, Phys. Rev. B 2, 3004 (1970)). Ideal mixing gives S_cc = x_A x_B.
Fitting a Redlich-Kister excess Gibbs energy to the curvatures at several compositions, with
G_ex = 0 at both ends, gives G_ex(x); the mixing enthalpy comes straight from the MD energies, and::

    S_ex = (H_ex - G_ex) / T.

S_cc(k) is evaluated on the reciprocal lattice of the cell, where it is exact for a periodic box,
and extrapolated to k = 0 as S_0 + a k^2. Long-wavelength composition modes relax by interdiffusion,
on the order of 1 / (D k^2), so the run has to be long against that time: on 864-atom liquid Cu-Zr
at 2300 K, 60 ps leaves S_cc at the smallest k uncertain by about 20 %.
"""

# Third-Party Library Imports
import numpy as np

# Local Imports
from hermes.compression_entropy import GAS_CONSTANT


def reciprocal_vectors(cell: np.ndarray, k_max: float) -> np.ndarray:
    """
    Reciprocal lattice vectors of the cell with 0 < ``|k|`` <= k_max, one of each +k/-k pair.

    Args:
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å.
        k_max: Largest wavevector in 1/Å.

    Returns:
        Wavevectors, shape ``(n_k, 3)``, in 1/Å.
    """
    b = 2 * np.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T
    n_max = int(np.ceil(k_max / np.min(np.linalg.norm(b, axis=1)))) + 1
    grid = np.array(np.meshgrid(*[np.arange(-n_max, n_max + 1)] * 3, indexing="ij")).reshape(3, -1).T
    # keep one of each +n / -n pair: first nonzero index positive
    first = np.array([row[np.nonzero(row)[0][0]] if np.any(row) else 0 for row in grid])
    grid = grid[first > 0]
    k = grid @ b
    return k[np.linalg.norm(k, axis=1) <= k_max]


def concentration_structure_factor(
    positions: np.ndarray, cell: np.ndarray, species: np.ndarray, k_max: float = 1.0, n_blocks: int = 5
) -> dict:
    """
    Bhatia-Thornton S_cc(k) of a binary system on the reciprocal lattice of a fixed cell.

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)``, in Å.
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å (NVT).
        species: Two species labels, shape ``(N,)``; the second of the sorted labels is B.
        k_max: Largest wavevector in 1/Å.
        n_blocks: Consecutive time blocks used for the error estimate.

    Returns:
        Dictionary with ``k`` (magnitudes, 1/Å), ``s_cc`` (per wavevector, averaged over all frames),
        ``blocks`` (shape ``(n_blocks, n_k)``) and ``x_B``.
    """
    positions = np.asarray(positions, dtype=float)
    labels = np.unique(species)
    if len(labels) != 2:
        raise ValueError("S_cc needs exactly two species")
    is_b = np.asarray(species) == labels[1]
    x_b = is_b.mean()
    weight = np.where(is_b, -(1.0 - x_b), x_b)
    kvec = reciprocal_vectors(cell, k_max)
    per_frame = np.array([np.abs(weight @ np.exp(1j * x @ kvec.T)) ** 2 for x in positions]) / len(weight)
    blocks = np.array([b.mean(0) for b in np.array_split(per_frame, n_blocks)])
    return {"k": np.linalg.norm(kvec, axis=1), "s_cc": per_frame.mean(0), "blocks": blocks, "x_B": float(x_b)}


def long_wavelength_limit(k: np.ndarray, s_cc: np.ndarray, blocks: np.ndarray = None, k_fit: float = 1.0) -> tuple:
    """
    S_cc(0) from a least-squares fit S_0 + a k^2 over all wavevectors with k <= k_fit.

    Returns:
        ``(s0, error)``: the intercept and, if ``blocks`` is given, its standard error from the spread
        of the block intercepts (otherwise NaN).
    """
    use = k <= k_fit
    design = np.stack([np.ones(use.sum()), k[use] ** 2], axis=1)

    def intercept(s):
        return np.linalg.lstsq(design, s[use], rcond=None)[0][0]

    s0 = float(intercept(s_cc))
    if blocks is None:
        return s0, float("nan")
    b = np.array([intercept(row) for row in blocks])
    return s0, float(b.std(ddof=1) / np.sqrt(len(b)))


def _rk_second_derivative_basis(x: np.ndarray, order: int) -> np.ndarray:
    """d^2/dx^2 of x(1-x)(1-2x)^k for k < order, shape (len(x), order)."""
    x = np.asarray(x, dtype=float)
    y = 1.0 - 2.0 * x
    f, f1, f2 = x * (1 - x), 1 - 2 * x, -2.0
    cols = []
    for k in range(order):
        p = y**k
        p1 = -2.0 * k * y ** (k - 1) if k >= 1 else 0.0 * x
        p2 = 4.0 * k * (k - 1) * y ** (k - 2) if k >= 2 else 0.0 * x
        cols.append(f2 * p + 2 * f1 * p1 + f * p2)
    return np.stack(cols, axis=1)


def excess_gibbs_from_scc(fractions, s_cc, temperature: float, order: int = 2, errors=None) -> np.ndarray:
    """
    Redlich-Kister coefficients L_k of G_ex = x(1-x) sum_k L_k (1-2x)^k from S_cc(0) values.

    The curvature target R T (1/S_cc - 1/(x(1-x))) is linear in the L_k, so this is a (weighted)
    linear least-squares fit. x is the mole fraction of B; L_k in J/mol of atoms.
    """
    x = np.asarray(fractions, dtype=float)
    s = np.asarray(s_cc, dtype=float)
    rt = GAS_CONSTANT * temperature
    target = rt * (1.0 / s - 1.0 / (x * (1 - x)))
    design = _rk_second_derivative_basis(x, order)
    if errors is not None:
        sigma = rt * np.asarray(errors, dtype=float) / s**2
        design, target = design / sigma[:, None], target / sigma
    return np.linalg.lstsq(design, target, rcond=None)[0]


def redlich_kister(x, coefficients) -> np.ndarray:
    """G_ex(x) = x(1-x) sum_k L_k (1-2x)^k."""
    x = np.asarray(x, dtype=float)
    return x * (1 - x) * sum(c * (1 - 2 * x) ** k for k, c in enumerate(coefficients))


def excess_mixing_entropy(fractions, s_cc, mixing_enthalpies, temperature: float, order: int = 2, errors=None) -> dict:
    """
    Excess mixing entropy S_ex = (H_ex - G_ex) / T at each composition, in J/(K mol of atoms).

    Args:
        fractions: Mole fraction of B at each composition.
        s_cc: S_cc(0) at each composition.
        mixing_enthalpies: H_ex in J/mol of atoms from the MD energies, against the pure liquids at
            the same T and P.
        temperature: Temperature in K.
        order: Number of Redlich-Kister terms.
        errors: Standard errors of ``s_cc``, used as weights.

    Returns:
        Dictionary with the Redlich-Kister ``coefficients`` of G_ex, ``g_ex`` and ``s_ex`` at each
        composition.
    """
    coefficients = excess_gibbs_from_scc(fractions, s_cc, temperature, order, errors)
    g_ex = redlich_kister(fractions, coefficients)
    return {"coefficients": coefficients, "g_ex": g_ex,
            "s_ex": (np.asarray(mixing_enthalpies, dtype=float) - g_ex) / temperature}
