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
import scipy.constants

# Third-Party Library Imports
import numpy as np
from scipy.spatial import cKDTree
from scipy import special, stats

# Gas constant in J/(K mol)
GAS_CONSTANT = scipy.constants.R
ANGSTROM = 1e-10


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


def displacement_residuals(target: np.ndarray, reference: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Residual vectors from each target atom to the nearest atom of the reference microstate.

    The nearest atom is found over the periodic images of the reference, so the mapping holds for
    any cell shape and also for liquids, where an atom's nearest reference atom need not be itself.

    Args:
        target: Cartesian positions of the target microstate, shape ``(N, 3)``, in Å.
        reference: Cartesian positions of the reference microstate, shape ``(N, 3)``, in Å.
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å.

    Returns:
        Residuals :math:`\\Delta r_i`, shape ``(N, 3)``, in Å.
    """
    cell = np.asarray(cell, dtype=float)
    # Wrap both into the cell, then search the reference and its 26 neighboring images.
    frac_t = np.linalg.solve(cell.T, np.asarray(target, dtype=float).T).T % 1.0
    frac_r = np.linalg.solve(cell.T, np.asarray(reference, dtype=float).T).T % 1.0
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
    target: np.ndarray, reference: np.ndarray, cell: np.ndarray, epsilon, dither: int = 8, **kwargs
) -> float:
    """
    Configurational entropy in J/(K mol of atoms) from one target and reference pair.

    Each ternary step shrinks the residuals by a factor of three, so the number of steps, and with
    it the bit count, jumps as the amplitude grows: on an Einstein solid the result oscillates by
    about 4 J/(K mol) with period ln 3 in the amplitude. Dithering removes it. The residuals are
    scaled by factors alpha spread log-uniformly over one period, each scaled set is compressed, and
    3 R ln(alpha), the exact cost of the scaling for three coordinates per atom, is subtracted
    before averaging.

    Args:
        target: Cartesian positions of the target microstate, shape ``(N, 3)``, in Å.
        reference: Cartesian positions of the reference microstate, shape ``(N, 3)``, in Å.
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å.
        epsilon: Reconstruction threshold in Å, a scalar or one value per atom.
        dither: Number of scale factors over one period; 1 switches dithering off.
        **kwargs: Passed to :func:`compress` (and ``pooled`` to :func:`coefficient_bits`).

    Returns:
        :math:`S = R \\ln 2 \\; N_\\mathrm{bits} / N` (Eq. 2 of Fisher and Hong), averaged over the
        dither.
    """
    pooled = kwargs.pop("pooled", False)
    residuals = displacement_residuals(target, reference, cell)
    values = []
    for alpha in 3.0 ** (np.arange(dither) / dither):
        _, coefficients = compress(alpha * residuals, epsilon, **kwargs)
        bits = coefficient_bits(coefficients, pooled=pooled)
        # The subtraction holds while the code resolves the positions; with almost nothing to encode
        # it would turn negative, so each term is floored at zero.
        values.append(max(0.0, GAS_CONSTANT * (np.log(2) * bits / len(target) - 3 * np.log(alpha))))
    return float(np.mean(values))


def trajectory_entropy(
    positions: np.ndarray,
    cell: np.ndarray,
    masses_amu: np.ndarray,
    temperature: float,
    n_pairs: int = 50,
    min_separation: int = 1,
    epsilon_scale: float = 1.0,
    seed: int = 0,
    **kwargs,
) -> dict:
    """
    Configurational entropy from random target and reference pairs of an equilibrated trajectory.

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)``, in Å, after equilibration.
        cell: Lattice vectors as rows, shape ``(3, 3)`` or ``(n_frames, 3, 3)``, in Å.
        masses_amu: Atomic masses, shape ``(N,)``, in atomic mass units.
        temperature: Temperature in K.
        n_pairs: Number of target and reference pairs.
        min_separation: Smallest separation of a pair in frames, so the two are decorrelated.
        epsilon_scale: Epsilon of each atom in units of its thermal de Broglie wavelength.
        seed: Seed for the pair selection.
        **kwargs: Passed to :func:`compress`.

    Returns:
        Dictionary with the entropy of every pair (``samples``), their ``mean`` and ``minimum``,
        and ``gamma_lower_bound``: the location of a gamma distribution fitted to the samples, the
        bound the paper's Appendix 3 takes as the estimate. Entropies in J/(K mol of atoms).
    """
    positions = np.asarray(positions, dtype=float)
    n_frames = len(positions)
    cells = np.broadcast_to(np.asarray(cell, dtype=float), (n_frames, 3, 3))
    epsilon = epsilon_scale * thermal_de_broglie_wavelength(masses_amu, temperature)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_pairs):
        while True:
            i, j = rng.integers(n_frames, size=2)
            if abs(i - j) >= min_separation:
                break
        samples.append(microstate_entropy(positions[i], positions[j], cells[i], epsilon, **kwargs))
    samples = np.array(samples)
    lower = np.nan
    if len(samples) >= 5 and np.ptp(samples) > 0:
        _, lower, _ = stats.gamma.fit(samples)
    return {"samples": samples, "mean": float(samples.mean()), "minimum": float(samples.min()),
            "gamma_lower_bound": float(lower)}


def _unbiased_log_det(covariances: np.ndarray, n_samples: int) -> np.ndarray:
    """ln det of sample covariances (ddof 1), less the Wishart mean bias, so Gaussian data give ln det Sigma on average."""
    d, nu = covariances.shape[-1], n_samples - 1
    bias = d * np.log(2.0 / nu) + sum(special.digamma((nu - i + 1) / 2.0) for i in range(1, d + 1))
    return np.linalg.slogdet(covariances)[1] - bias


def first_shell_cutoff(sites: np.ndarray, cell: np.ndarray, shell_gap: float = 0.05, neighbours: int = 24) -> float:
    """
    Distance halfway between the first neighbour shell and the next.

    Site-to-site distances are sorted and the first shell ends at the first jump of more than
    ``shell_gap`` (relative). That keeps the slightly split shell of a non-ideal hcp metal together
    (2 % in Ti) and separates bcc's first shell from its second (15 %).
    """
    cell = np.asarray(cell, dtype=float)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    images = (sites[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
    distances, _ = cKDTree(images).query(sites, k=neighbours + 1)
    d = np.sort(distances[:, 1:].ravel())
    jumps = np.flatnonzero(d[1:] > d[:-1] * (1 + shell_gap))
    if not len(jumps):
        raise ValueError("no gap after the first neighbour shell; is this a crystal?")
    return float(0.5 * (d[jumps[0]] + d[jumps[0] + 1]))


def neighbour_correlation_entropy(
    positions: np.ndarray,
    cell: np.ndarray,
    shell_gap: float = 0.05,
    max_rms_fraction: float = 0.25,
    chunk: int = 20_000,
) -> dict:
    r"""
    Entropy shared between neighbouring atoms of a solid, which per-atom coding counts twice.

    :func:`microstate_entropy` codes each atom's residual on its own, so it measures the sum of the
    atoms' marginal entropies. The entropy of the solid is lower by the total correlation of the
    displacements. This takes it in the Gaussian pair approximation over the first neighbour shell,

    .. math:: S_\mathrm{corr} = \frac{R}{N} \sum_{\langle ij \rangle} \frac{1}{2}
              \ln \frac{\det C_i \det C_j}{\det C_{ij}},

    with each pair's 3x3 and 6x6 displacement covariances taken over the frames. The log
    determinants are corrected for the finite number of frames by their exact Wishart mean, so the
    estimate is unbiased for Gaussian displacements from independent frames; correlated frames
    count as fewer, so space them at least a few vibrational periods apart. Displacements are from
    each atom's mean site, with the frame's mean displacement removed. On fcc Cu (Mendelev EAM,
    1300 K) the compression entropy less this term (5.3) gives 72.5 J/(K mol) against 72.76 from
    thermodynamic integration. The pair approximation leaves out correlations among three or more
    atoms and beyond the first shell.

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)``, in Å, of a solid in which no
            atom leaves its site.
        cell: Lattice vectors as rows, shape ``(3, 3)``, in Å (NVT).
        shell_gap: Relative distance jump that ends the first shell; see :func:`first_shell_cutoff`.
        max_rms_fraction: Largest rms displacement, as a fraction of the cutoff, accepted as a solid.
        chunk: Pairs per batch, to bound memory.

    Returns:
        Dictionary with ``entropy`` (J/(K mol of atoms), to subtract), ``pair_information`` (nats per
        pair), ``pairs_per_atom``, ``cutoff`` (Å) and ``rms_displacement`` (Å).

    Raises:
        ValueError: If the system is not a solid, has fewer frames than 8, or the cell is so small
            that an atom meets another atom's two images within the cutoff.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n_frames, n_atoms = positions.shape[:2]
    if n_frames < 8:
        raise ValueError("need at least 8 frames for a 6x6 covariance")
    frac = np.linalg.solve(cell.T, positions.reshape(-1, 3).T).T.reshape(positions.shape)
    step = frac - frac[0]
    step -= np.round(step)
    track = (frac[0] + step) @ cell
    sites = track.mean(axis=0)
    u = track - sites
    u -= u.mean(axis=1, keepdims=True)

    cutoff = first_shell_cutoff(sites, cell, shell_gap)
    rms = float(np.sqrt((u**2).sum(axis=2).mean()))
    if rms > max_rms_fraction * cutoff:
        raise ValueError(f"rms displacement {rms:.2f} A is {rms / cutoff:.2f} of the neighbour cutoff; not a solid")

    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    images = (sites[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
    pairs = []
    for i, hits in enumerate(cKDTree(images).query_ball_point(sites, cutoff)):
        partners = np.array([h % n_atoms for h in hits if h % n_atoms != i], dtype=int)
        if len(partners) != len(np.unique(partners)):
            raise ValueError("cell too small: an atom meets two images of one neighbour")
        pairs.extend((i, j) for j in partners if j > i)
    pairs = np.array(pairs, dtype=int)

    information = []
    for start in range(0, len(pairs), chunk):
        p = pairs[start:start + chunk]
        x = np.concatenate([u[:, p[:, 0]], u[:, p[:, 1]]], axis=2)  # (frames, pairs, 6)
        x = x - x.mean(axis=0)
        cov = np.einsum("fpa,fpb->pab", x, x) / (n_frames - 1)
        information.append(0.5 * (_unbiased_log_det(cov[:, :3, :3], n_frames)
                                  + _unbiased_log_det(cov[:, 3:, 3:], n_frames)
                                  - _unbiased_log_det(cov, n_frames)))
    information = np.concatenate(information)
    return {"entropy": float(GAS_CONSTANT * information.sum() / n_atoms), "pair_information": information,
            "pairs_per_atom": 2 * len(pairs) / n_atoms, "cutoff": cutoff, "rms_displacement": rms}
