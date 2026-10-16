"""Entropy from the compressibility of atomic configurations.

An implementation of the compression route to entropy of Fisher and Hong, "Thermodynamic Entropy
as Information" (arXiv:2512.02221, 2025), called ASDF there. The residual displacements between a
target microstate and a reference microstate from the same equilibrated trajectory are encoded with
a dictionary of basis vectors and ternary coefficients {-1, 0, 1} until every atom is reproduced
within a threshold epsilon. The Shannon entropy of the coefficient stream, in bits, converts to
entropy through S = k_B ln(2) H (their Eqs. 2 and 7).

The paper describes the compressor in words and its code was not released with it. This module is
a clean-room implementation written from the preprint alone, https://arxiv.org/abs/2512.02221. The choices
here are therefore this implementation's, and are stated where they are made:

- the basis directions are the principal axes of the residuals, fixed once, and each step is two
  thirds of the remaining spread along one axis (balanced ternary). The paper takes a new
  eigenvector at every step and scales it at a "knee point"; tested on an exactly sampled Einstein
  solid, the fixed axes come within 2 % of the exact information dimension, 3 per atom;
- each coefficient column is coded with its own symbol frequencies, a shorter valid code than
  the paper's single pooled frequency (still available, ``pooled=True``);
- the final re-optimization is a coordinate sweep that sets a coefficient to zero whenever the
  atom stays within epsilon, and otherwise picks the value with the smallest residual;
- epsilon defaults to the thermal de Broglie wavelength h / sqrt(2 pi m k_B T) of each atom's
  element. The paper writes this length both with and without a further factor of 1/sqrt(3).

The result is the configurational part only. The kinetic entropy depends on T and the masses alone
and cancels between phases at the same T; electronic entropy is not contained in the positions and
has to be added separately.
"""

# Standard Library Imports
from fractions import Fraction

import scipy.constants

# Third-Party Library Imports
import numpy as np
from scipy.spatial import cKDTree
from scipy import special, stats

# Gas constant in J/(K mol)
GAS_CONSTANT = scipy.constants.R
ANGSTROM = 1e-10
# Atoms coded together. The code's fixed cost per atom depends on how many atoms it sees (its first
# step is set by the largest residual), so groups of one size keep the offset the same for any box.
GROUP_SIZE = 256


def thermal_de_broglie_wavelength(mass_amu: np.ndarray, temperature: float) -> np.ndarray:
    """
    Thermal de Broglie wavelength :math:`h / \\sqrt{2 \\pi m k_B T}` in Å.

    Args:
        mass_amu: Atomic masses in atomic mass units.
        temperature: Temperature in K.

    Returns:
        Wavelengths in Å, one per mass.
    """
    m = np.asarray(mass_amu, dtype=float) * scipy.constants.atomic_mass
    return scipy.constants.h / np.sqrt(2 * np.pi * m * scipy.constants.k * temperature) / ANGSTROM


def displacement_residuals(
    target: np.ndarray, reference: np.ndarray, cell: np.ndarray, reference_cell: np.ndarray = None
) -> np.ndarray:
    """
    Residual vectors from each target atom to the nearest atom of the reference microstate.

    The nearest atom is found over the periodic images of the reference, so the mapping holds for
    any cell shape and also for liquids, where an atom's nearest reference atom need not be itself.
    Each microstate is read in its own cell, so in an NPT run the strain between the two frames is
    not coded as displacement.

    Args:
        target: Cartesian positions of the target microstate, shape ``(N, 3)``, in Å.
        reference: Cartesian positions of the reference microstate, shape ``(N, 3)``, in Å.
        cell: Lattice vectors of the target as rows, shape ``(3, 3)``, in Å.
        reference_cell: Lattice vectors of the reference, if they differ from ``cell``.

    Returns:
        Residuals :math:`\\Delta r_i`, shape ``(N, 3)``, in Å.
    """
    cell = np.asarray(cell, dtype=float)
    # Wrap both into the cell, then search the reference and its 26 neighboring images.
    frac_t = np.linalg.solve(cell.T, np.asarray(target, dtype=float).T).T % 1.0
    ref_cell = cell if reference_cell is None else np.asarray(reference_cell, dtype=float)
    frac_r = np.linalg.solve(ref_cell.T, np.asarray(reference, dtype=float).T).T % 1.0
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    images = (frac_r[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ cell
    target_cart = frac_t @ cell
    _, index = cKDTree(images).query(target_cart)
    return target_cart - images[index]


def compress(residuals: np.ndarray, epsilon, sweeps: int = 3, max_basis: int = 10_000):
    r"""
    Encode residual vectors with basis vectors and ternary coefficients until each is within epsilon.

    The basis directions are the principal axes of the residuals, fixed once at the start. Each step
    takes the axis along which the residuals still above epsilon spread furthest, sets the step to
    two thirds of that spread, and subtracts it with coefficient -1, 0 or 1 (balanced ternary), which
    cuts the spread along that axis by a factor of three. Holding the axes fixed keeps every
    reconstruction cell a box, the case in which successive ternary refinement costs close to one
    bit per halving per coordinate. A final set of coordinate sweeps re-optimizes the coefficients
    against the whole basis and drops basis vectors that end up unused.

    On an exactly sampled Einstein solid each doubling of the amplitude adds 3.06 R ln 2 (exact
    3 R ln 2) with per-column coding (:func:`coefficient_bits`). The paper takes a new eigenvector
    of the residual covariance at every step; doing that here gives 3.5 R ln 2.

    Args:
        residuals: Residual vectors, shape ``(N, 3)``, in Å.
        epsilon: Reconstruction threshold in Å, a scalar or one value per atom.
        sweeps: Number of re-optimization sweeps over the basis.
        max_basis: Safety limit on the number of basis vectors.

    Returns:
        ``(basis, coefficients)``: basis vectors, shape ``(M, 3)``, and coefficients in
        {-1, 0, 1}, shape ``(N, M)``. Every atom satisfies
        :math:`\lVert \Delta r_i - \sum_j c_{ij} b_j \rVert \le \epsilon_i`.

    Raises:
        RuntimeError: If a step shortens no residual, or the basis limit is reached.
    """
    target = np.asarray(residuals, dtype=float)
    n_atoms = len(target)
    eps = np.broadcast_to(np.asarray(epsilon, dtype=float), (n_atoms,))
    remaining = target.copy()
    basis, columns = [], []
    _, axes = np.linalg.eigh(target.T @ target / max(n_atoms, 1))

    active = np.linalg.norm(remaining, axis=1) > eps
    while active.any():
        if len(basis) >= max_basis:
            raise RuntimeError(f"no convergence within {max_basis} basis vectors")
        spread = np.abs(remaining[active] @ axes).max(axis=0)
        axis = int(np.argmax(spread))
        step = 2.0 / 3.0 * spread[axis]
        projection = remaining @ axes[:, axis]
        coefficient = np.zeros(n_atoms, dtype=np.int8)
        apply = active & (np.abs(projection) > step / 2)
        if not apply.any():
            raise RuntimeError("a step shortened no residual")
        coefficient[apply] = np.sign(projection[apply]).astype(np.int8)
        remaining -= np.outer(coefficient, step * axes[:, axis])
        basis.append(step * axes[:, axis])
        columns.append(coefficient)
        active = np.linalg.norm(remaining, axis=1) > eps

    if not basis:
        return np.zeros((0, 3)), np.zeros((n_atoms, 0), dtype=np.int8)
    basis, coefficients = np.array(basis), np.stack(columns, axis=1)

    # Re-optimization. Each change keeps every atom within epsilon: zero is taken only if it stays
    # within, and otherwise the value with the smallest residual, which is no larger than before.
    for _ in range(sweeps):
        changed = False
        for j in range(len(basis)):
            without = target - coefficients @ basis + np.outer(coefficients[:, j], basis[j])
            trial = np.stack([np.linalg.norm(without - c * basis[j], axis=1) for c in (-1, 0, 1)], axis=1)
            best = np.array([-1, 0, 1], dtype=np.int8)[np.argmin(trial, axis=1)]
            choice = np.where(trial[:, 1] <= eps, np.int8(0), best)
            if np.any(choice != coefficients[:, j]):
                coefficients[:, j] = choice
                changed = True
        if not changed:
            break
    keep = np.any(coefficients != 0, axis=0)
    basis, coefficients = basis[keep], coefficients[:, keep]

    error = np.linalg.norm(target - coefficients @ basis, axis=1)
    if np.any(error > eps * (1 + 1e-12)):
        raise RuntimeError("an atom left the epsilon bound during re-optimization")
    return basis, coefficients


def coefficient_bits(coefficients: np.ndarray, pooled: bool = False) -> float:
    r"""
    Information of a coefficient matrix in bits.

    Each basis vector's column is coded with its own symbol frequencies: early columns are mostly
    +1 and -1, late ones mostly 0, and one code for all of them wastes the difference. ``pooled=True``
    gives the paper's form (Eq. 2), one Shannon entropy over all entries, which is never smaller.

    Args:
        coefficients: Ternary coefficients, shape ``(N, M)``.
        pooled: Code all entries with one set of symbol frequencies.

    Returns:
        :math:`\sum_j N H_j`, with :math:`H_j = -\sum_x p_{jx} \log_2 p_{jx}` over the symbols of
        column j, or :math:`N M H` with one pooled H.
    """
    c = np.asarray(coefficients)
    if c.size == 0:
        return 0.0

    def entropy(x):
        _, counts = np.unique(x, return_counts=True)
        p = counts / x.size
        return float(-(p * np.log2(p)).sum())

    if pooled:
        return c.size * entropy(c)
    return float(sum(len(c) * entropy(c[:, j]) for j in range(c.shape[1])))


def microstate_entropy(
    target: np.ndarray, reference: np.ndarray, cell: np.ndarray, epsilon, dither: int = 8,
    reference_cell: np.ndarray = None, group_size: int = GROUP_SIZE, seed: int = 0, **kwargs
) -> float:
    """
    Configurational entropy in J/(K mol of atoms) from one target and reference pair.

    Each ternary step shrinks the residuals by a factor of three, so the number of steps, and with
    it the bit count, jumps as the amplitude grows: on an Einstein solid the result oscillates by
    about 4 J/(K mol) with period ln 3 in the amplitude. Dithering removes it. The residuals are
    scaled by factors alpha spread log-uniformly over one period, each scaled set is compressed, and
    3 R ln(alpha), the exact cost of the scaling for three coordinates per atom, is subtracted
    before averaging.

    The atoms are coded in random groups of ``group_size``, the atoms left over after the last full
    group being dropped. On exactly sampled Einstein solids the fixed cost per atom then stays the
    same from 343 to 2197 atoms; coded all together it falls by 3 J/(K mol) over that range.

    Args:
        target: Cartesian positions of the target microstate, shape ``(N, 3)``, in Å.
        reference: Cartesian positions of the reference microstate, shape ``(N, 3)``, in Å.
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å.
        epsilon: Reconstruction threshold in Å, a scalar or one value per atom.
        dither: Number of scale factors over one period; 1 switches dithering off.
        reference_cell: Lattice vectors of the reference, if they differ from ``cell`` (NPT).
        group_size: Atoms coded together; None codes all at once, as before version 0.1.0.
        seed: Seed for the grouping.
        **kwargs: Passed to :func:`compress` (and ``pooled`` to :func:`coefficient_bits`).

    Returns:
        :math:`S = R \\ln 2 \\; N_\\mathrm{bits} / N` (Eq. 2 of Fisher and Hong), averaged over the
        dither and the groups.

    Raises:
        ValueError: If there are fewer atoms than ``group_size``.
    """
    pooled = kwargs.pop("pooled", False)
    residuals = displacement_residuals(target, reference, cell, reference_cell)
    n_atoms = len(residuals)
    eps = np.broadcast_to(np.asarray(epsilon, dtype=float), (n_atoms,))
    if group_size is None:
        groups = [np.arange(n_atoms)]
    else:
        if n_atoms < group_size:
            raise ValueError(f"{n_atoms} atoms are fewer than one group of {group_size}")
        order = np.random.default_rng(seed).permutation(n_atoms)
        groups = [order[g * group_size:(g + 1) * group_size] for g in range(n_atoms // group_size)]
    values = []
    for alpha in 3.0 ** (np.arange(dither) / dither):
        for group in groups:
            _, coefficients = compress(alpha * residuals[group], eps[group], **kwargs)
            bits = coefficient_bits(coefficients, pooled=pooled)
            # The subtraction holds while the code resolves the positions; with almost nothing to
            # encode it would turn negative, so each term is floored at zero.
            values.append(max(0.0, GAS_CONSTANT * (np.log(2) * bits / len(group) - 3 * np.log(alpha))))
    return float(np.mean(values))


def trajectory_entropy(
    positions: np.ndarray,
    cell: np.ndarray,
    masses_amu: np.ndarray,
    temperature: float,
    n_pairs: int = 60,
    min_separation: int = 1,
    epsilon_scale: float = 1.0,
    seed: int = 0,
    min_groups: int = 480,
    **kwargs,
) -> dict:
    """
    Configurational entropy from random target and reference pairs of an equilibrated trajectory.

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)``, in Å, after equilibration.
        cell: Lattice vectors as rows, shape ``(3, 3)`` or ``(n_frames, 3, 3)``, in Å.
        masses_amu: Atomic masses, shape ``(N,)``, in atomic mass units.
        temperature: Temperature in K.
        n_pairs: Smallest number of target and reference pairs. It is raised so that at least
            ``min_groups`` groups of atoms are coded, which keeps the statistical error of a small
            cell (one group per pair at 256 atoms) as small as that of a large one.
        min_separation: Smallest separation of a pair in frames, so the two are decorrelated.
        epsilon_scale: Epsilon of each atom in units of its thermal de Broglie wavelength.
        seed: Seed for the pair selection.
        min_groups: Smallest number of atom groups coded in all.
        **kwargs: Passed to :func:`microstate_entropy` and :func:`compress`.

    Returns:
        Dictionary with the entropy of every pair (``samples``), their ``mean`` and ``minimum``,
        and ``gamma_lower_bound``: the location of a gamma distribution fitted to the samples, the
        bound the paper's Appendix 3 takes as the estimate. Entropies in J/(K mol of atoms).
    """
    positions = np.asarray(positions, dtype=float)
    n_frames = len(positions)
    cells = np.broadcast_to(np.asarray(cell, dtype=float), (n_frames, 3, 3))
    epsilon = epsilon_scale * thermal_de_broglie_wavelength(masses_amu, temperature)
    if n_frames - 1 < min_separation:
        raise ValueError(f"{n_frames} frames cannot give pairs {min_separation} frames apart")
    group_size = kwargs.get("group_size", GROUP_SIZE)
    if group_size is not None:
        n_pairs = max(n_pairs, -(-min_groups // max(1, positions.shape[1] // group_size)))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_pairs):
        while True:
            i, j = rng.integers(n_frames, size=2)
            if abs(i - j) >= min_separation:
                break
        samples.append(microstate_entropy(positions[i], positions[j], cells[i], epsilon, reference_cell=cells[j],
                                          seed=int(rng.integers(2**31)), **kwargs))
    samples = np.array(samples)
    lower = np.nan
    if len(samples) >= 5 and np.ptp(samples) > 0:
        _, lower, _ = stats.gamma.fit(samples)
    return {"samples": samples, "mean": float(samples.mean()), "minimum": float(samples.min()),
            "gamma_lower_bound": float(lower)}


def _log_det_bias(d: int, nu: int, complex_valued: bool = False) -> float:
    """Mean of ln det S - ln det Sigma for a sample covariance S with nu degrees of freedom (Wishart)."""
    if complex_valued:
        return sum(special.digamma(nu - i) for i in range(d)) - d * np.log(nu)
    return d * np.log(2.0 / nu) + sum(special.digamma((nu - i + 1) / 2.0) for i in range(1, d + 1))


def crystal_translations(sites: np.ndarray, cell: np.ndarray, species: np.ndarray, tol: float = None) -> tuple:
    """
    Lattice translations that carry the crystal onto itself, species included.

    Args:
        sites: Mean atomic positions, shape ``(N, 3)``, in Å.
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å.
        species: Species label of each atom, shape ``(N,)``.
        tol: Largest misfit of a mapped site in Å; a fifth of the nearest-neighbour distance by
            default, wide enough for mean sites from a short run and far below the miss of a wrong
            translation.

    Returns:
        ``(vectors, permutations)``: the translations in fractional coordinates of the cell, each
        averaged over all atoms and snapped to the nearest fraction with a denominator no larger than
        the number of atoms, shape ``(n_t, 3)``; and for each the index of the site every atom is
        carried to, shape ``(n_t, N)``.
    """
    cell = np.asarray(cell, dtype=float)
    species = np.asarray(species)
    sites = np.asarray(sites, dtype=float)
    n_atoms = len(sites)
    if tol is None:
        tol = 0.2 * _nearest_neighbour_distance(sites, cell)
    frac = (sites @ np.linalg.inv(cell)) % 1.0
    tree = cKDTree(frac, boxsize=1.0 + 1e-12)
    ftol = tol / np.linalg.norm(cell, axis=1).max()
    vectors, permutations = [], []
    for j in np.flatnonzero(species == species[0]):
        shift = frac[j] - frac[0]
        distance, index = tree.query((frac + shift) % 1.0)
        if distance.max() < ftol and np.all(species[index] == species) and len(np.unique(index)) == n_atoms:
            moved = frac[index] - frac
            moved -= np.round(moved - shift)  # unwrap against the trial shift, then average out the noise
            mean = moved.mean(axis=0)
            vectors.append([float(Fraction(x).limit_denominator(n_atoms)) for x in mean - np.round(mean)])
            permutations.append(index)
    return np.array(vectors), np.array(permutations)


def _nearest_neighbour_distance(sites: np.ndarray, cell: np.ndarray) -> float:
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    images = (sites[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
    return float(cKDTree(images).query(sites, k=2)[0][:, 1].min())


def _characters(vectors: np.ndarray) -> np.ndarray:
    """The distinct characters exp(2 pi i H.f) of a translation group given by its fractional vectors."""
    steps = []
    for column in vectors.T:
        denominators = [Fraction(x).limit_denominator(len(vectors)).denominator for x in column]
        steps.append(int(np.lcm.reduce(denominators)))
    found = {}
    grid = np.array(np.meshgrid(*[np.arange(s) for s in steps], indexing="ij")).reshape(3, -1).T
    for start in range(0, len(grid), 256):
        phases = np.exp(2j * np.pi * grid[start:start + 256] @ vectors.T)
        for row in phases:
            found.setdefault((np.round(row, 6) + 0.0).tobytes(), row)  # + 0.0 turns -0.0 into 0.0
        if len(found) == len(vectors):
            break
    return np.array(list(found.values()))


def crystal_correlation_entropy(
    positions: np.ndarray, cell: np.ndarray, species: np.ndarray = None, max_rms_fraction: float = 0.25
) -> dict:
    r"""
    Entropy shared among the atoms of a crystal, which per-atom coding counts more than once.

    :func:`microstate_entropy` codes each atom on its own and so measures the sum of the atoms'
    marginal entropies. The entropy of the crystal is lower by the total correlation of the
    displacements, taken here in the Gaussian approximation and over all ranges,

    .. math:: S_\mathrm{corr} = \frac{R}{2N} \Bigl[\sum_i \ln\det C_{ii} - \ln\det C\Bigr],

    with :math:`C_{ii}` each atom's 3x3 displacement covariance and C the 3N x 3N covariance. MD holds
    the centre of mass still, which leaves C three zero modes; they are given the variance they would
    have if the atoms moved independently, the mean variance of one coordinate, the convention of the
    Einstein solids that calibrate the compression offset. The result does not depend on the units,
    and for independent atoms it is -3R/(2N): holding the centre of mass still narrows each atom's
    spread by a factor 1 - 1/N, which lowers the compression entropy by just that much, and this term
    returns it. The crystal's translations make C block diagonal in wavevector, one
    3m x 3m block per wavevector for m atoms per primitive cell, so ln det C is exact without a model
    of which atoms interact. Each log determinant is corrected by its exact Wishart mean, real or
    complex, so Gaussian displacements from independent frames give an unbiased result; frames
    correlated in time count as fewer, so space them a few vibrational periods apart. On harmonic
    fcc, bcc and hcp crystals of 108 to 864 atoms this is within 0.03 J/(K mol) of the exact total
    correlation for 60 to 1000 frames, where a sum over neighbour pairs is up to 0.8 high.

    MD also keeps the crystal from drifting as a whole, which a crystal free in its box does, as
    thermodynamic integration by the Frenkel-Ladd route assumes. That adds, per atom,

    .. math:: S_\mathrm{cm} = \frac{R}{N} \ln\Bigl[V \Bigl(\frac{N}{2\pi e \sigma^2}\Bigr)^{3/2}\Bigr],

    the entropy of a centre of mass free in the cell volume V less that of the centre of mass of N
    independent atoms of coordinate variance :math:`\sigma^2`, the convention used above. It is 0.6
    J/(K mol) at 256 atoms and 0.1 at 2048, and agrees within 0.015 with the centre-of-mass correction
    calphy applies (12 EAM crystals). Leaving it out made the error depend on the box.

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)``, in Å, of a crystal in which no
            atom leaves its site. The cell must be periodic in the crystal's own translations.
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å (NVT).
        species: Species label of each atom; all alike if omitted. A random alloy has no
            translations that carry its species onto themselves and is refused.
        max_rms_fraction: Largest rms displacement, as a fraction of the nearest-neighbour distance,
            accepted as a solid.

    Returns:
        Dictionary with ``entropy``, the shared entropy less the centre-of-mass entropy, which is what
        to subtract from the compression entropy; its parts ``shared`` and ``centre_of_mass``; all in
        J/(K mol of atoms); ``cells`` (primitive cells in the box), ``basis`` (atoms per primitive
        cell) and ``rms_displacement`` (Å).

    Raises:
        ValueError: If the system is not a solid, has fewer than 8 frames, or its translations do
            not give one wavevector per primitive cell.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n_frames, n_atoms = positions.shape[:2]
    species = np.zeros(n_atoms, dtype=int) if species is None else np.asarray(species)
    if n_frames < 8:
        raise ValueError("need at least 8 frames")
    frac = np.linalg.solve(cell.T, positions.reshape(-1, 3).T).T.reshape(positions.shape)
    step = frac - frac[0]
    step -= np.round(step)
    track = (frac[0] + step) @ cell
    sites = track.mean(axis=0)
    u = track - sites
    u -= u.mean(axis=1, keepdims=True)

    nearest = _nearest_neighbour_distance(sites, cell)
    rms = float(np.sqrt((u**2).sum(axis=2).mean()))
    if rms > max_rms_fraction * nearest:
        raise ValueError(f"rms displacement {rms:.2f} A is {rms / nearest:.2f} of the nearest-neighbour distance; not a solid")

    vectors, permutations = crystal_translations(sites, cell, species)
    n_cells = len(vectors)
    if n_atoms % n_cells:
        raise ValueError("the crystal's translations do not tile the cell")
    basis, seen = [], np.zeros(n_atoms, dtype=bool)
    for i in range(n_atoms):
        if not seen[i]:
            basis.append(i)
            seen[permutations[:, i]] = True
    m = len(basis)
    if 3 * m >= n_frames - 1:
        raise ValueError(f"{n_frames} frames cannot resolve a primitive cell of {m} atoms; a random alloy, whose "
                         "species break the translations, has the whole box as its cell")
    characters = _characters(vectors)
    if len(characters) != n_cells:
        raise ValueError(f"found {len(characters)} wavevectors for {n_cells} primitive cells")

    nu = n_frames - 1
    ub = u[:, permutations[:, basis], :].reshape(n_frames, n_cells, 3 * m)
    uq = np.einsum("kt,ftd->fkd", characters.conj(), ub) / np.sqrt(n_cells)
    cq = np.einsum("fka,fkb->kab", uq, uq.conj()) / nu
    marginals = np.einsum("fia,fib->iab", u, u) / nu
    independent = float(np.trace(marginals, axis1=1, axis2=2).mean() / 3)
    log_det = 0.0
    for chi, block in zip(characters, cq):
        w = np.linalg.eigvalsh(block)
        if np.allclose(chi, 1.0):
            log_det += 3 * np.log(independent)  # the three centre-of-mass modes, see above
            w = w[3:]
        if len(w):
            log_det += np.sum(np.log(w)) - _log_det_bias(len(w), nu, complex_valued=not np.allclose(chi.imag, 0.0))
    marginal = np.sum(np.linalg.slogdet(marginals)[1]) - n_atoms * _log_det_bias(3, nu)
    shared = GAS_CONSTANT * 0.5 * (marginal - log_det) / n_atoms
    volume = abs(np.linalg.det(cell))
    centre_of_mass = GAS_CONSTANT / n_atoms * np.log(volume * (n_atoms / (2 * np.pi * np.e * independent)) ** 1.5)
    return {"entropy": float(shared - centre_of_mass), "shared": float(shared), "centre_of_mass": float(centre_of_mass),
            "cells": n_cells, "basis": m, "rms_displacement": rms}


def gaussian_entropy(
    positions: np.ndarray, cell: np.ndarray, masses_amu, temperature: float, species: np.ndarray = None
) -> dict:
    r"""
    Total classical entropy of a crystal in the Gaussian (quasi-harmonic) approximation, J/(K mol of atoms).

    Each atom's motion is taken as Gaussian with its measured covariance, which in the classical convention of the
    Einstein solids that calibrate the compression gives

    .. math:: S_i = \frac{3R}{2} + \frac{R}{2} \ln\det\bigl(2\pi e\, C_{ii} / \Lambda_i^2\bigr),

    :math:`\Lambda_i` the thermal de Broglie wavelength (an Einstein solid gives 3R(1 + ln(k T / hbar omega)) exactly);
    the log determinants carry the Wishart correction of :func:`crystal_correlation_entropy`, whose shared entropy less
    centre-of-mass entropy is then subtracted. Unlike the compression it needs no minimum amplitude, so it serves as the
    anchor of :func:`hermes.heat_capacity.entropy_from_heat_capacity` at moderate temperature, where the motion is
    Gaussian; near melting it reads high, like the compression route, for the same reason.

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)``, in Å, of a crystal (NVT).
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å.
        masses_amu: Atomic masses, shape ``(N,)``, in atomic mass units.
        temperature: Temperature in K.
        species: Species label of each atom; all alike if omitted.

    Returns:
        Dictionary with ``entropy`` and its parts ``marginal`` (sum of the atoms' own entropies per atom) and
        ``correlation`` (shared less centre of mass, subtracted), all in J/(K mol of atoms).
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n_frames = len(positions)
    corr = crystal_correlation_entropy(positions, cell, species)
    frac = np.linalg.solve(cell.T, positions.reshape(-1, 3).T).T.reshape(positions.shape)
    step = frac - frac[0]
    step -= np.round(step)
    track = (frac[0] + step) @ cell
    u = track - track.mean(axis=0)
    u -= u.mean(axis=1, keepdims=True)
    lam = thermal_de_broglie_wavelength(np.broadcast_to(np.asarray(masses_amu, dtype=float), (u.shape[1],)), temperature)
    cov = np.einsum("fia,fib->iab", u, u) / (n_frames - 1) / lam[:, None, None] ** 2
    log_det = np.linalg.slogdet(cov)[1] - _log_det_bias(3, n_frames - 1)
    marginal = float(np.mean(GAS_CONSTANT * (1.5 + 0.5 * (3 * np.log(2 * np.pi * np.e) + log_det))))
    return {"entropy": marginal - corr["entropy"], "marginal": marginal, "correlation": corr["entropy"]}
