"""Entropy of a liquid from its pair structure.

Compression codes each atom on its own and so counts the information atoms share; in a liquid that
shared part is large (packing) and the compression entropy comes out far too high. This module is
the liquid route instead:

    S = S_ideal + S_2
    S_ideal / R = 5/2 - sum_a x_a ln(rho x_a Lambda_a^3)
    S_2 / R = -(rho / 2) sum_ab x_a x_b  int [g_ab ln g_ab - g_ab + 1] 4 pi r^2 dr

S_ideal is the ideal gas at the liquid's density and composition (Sackur-Tetrode, ideal mixing
included) and S_2 the two-body excess entropy from the partial pair distribution functions
(R. E. Nettleton and M. S. Green, J. Chem. Phys. 29, 1365 (1958); A. Baranyai and D. J. Evans,
Phys. Rev. A 40, 3817 (1989)). Chemical short-range order enters through the partials, so S_2 also
carries the departure of the mixing entropy from ideal.

The integrand keeps the -g + 1 term so it is second order in g - 1 at large r, where a finite
canonical cell biases g by order 1/N. The grand-canonical form with the long-wavelength structure
factor needs the integral of g ln g alone to converge, and in cells of a few hundred atoms it does
not.

Left out is the residual multiparticle entropy, three-body and higher. On the Mendelev EAM for Cu
at 1300 K (864 atoms) the result is 81.0 against 82.5 J/(K mol) from thermodynamic integration on
the same potential. The residual also carries most of the chemistry: on liquid Cu-Zr at 2300 K
the pure liquids come out 2.0 to 2.2 J/(K mol) high, and the excess mixing entropy is -0.2 to +0.5
where the exact values are -1.0 to -1.7, so short-range order beyond pairs is not captured.
Everything here is classical; :mod:`hermes.quantum_correction` supplies the quantum part.
"""

# Third-Party Library Imports
import numpy as np
from scipy.spatial import cKDTree

# Local Imports
from hermes.compression_entropy import GAS_CONSTANT, thermal_de_broglie_wavelength


def _shifts() -> np.ndarray:
    return np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


def half_cell_width(cell: np.ndarray) -> float:
    """Half the smallest distance between opposite faces of the cell, in Å."""
    cell = np.asarray(cell, dtype=float)
    volume = abs(np.linalg.det(cell))
    widths = [volume / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    return 0.5 * min(widths)


def partial_pair_distributions(
    positions: np.ndarray,
    cell: np.ndarray,
    species: np.ndarray,
    r_max: float = None,
    dr: float = 0.02,
) -> tuple:
    """
    Partial pair distribution functions g_ab(r) averaged over the frames of a trajectory.

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)`` or ``(N, 3)``, in Å.
        cell: Lattice vectors as rows, shape ``(3, 3)`` or ``(n_frames, 3, 3)``, in Å.
        species: Integer species label of each atom, shape ``(N,)``.
        r_max: Largest distance in Å, at most half the smallest cell width (the default), so every
            shell is complete and no atom meets its own image.
        dr: Bin width in Å.

    Returns:
        ``(r, g, labels)``: bin centres, shape ``(n_bins,)``; partials, shape
        ``(n_species, n_species, n_bins)``, ordered as ``labels``; and the sorted species labels.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 2:
        positions = positions[None]
    n_frames, n_atoms, _ = positions.shape
    cells = np.broadcast_to(np.asarray(cell, dtype=float), (n_frames, 3, 3))
    labels, index = np.unique(np.asarray(species), return_inverse=True)
    n_species = len(labels)
    counts_per_species = np.bincount(index, minlength=n_species)

    limit = min(half_cell_width(c) for c in cells)
    r_max = limit if r_max is None else r_max
    if r_max > limit + 1e-9:
        raise ValueError(f"r_max {r_max} exceeds half the smallest cell width, {limit:.3f} Å")
    # Whole bins only: a last bin reaching past r_max would be part empty, and at large r its shell
    # volume makes that error dominate the entropy integral.
    edges = dr * np.arange(int(np.floor(r_max / dr + 1e-9)) + 1)
    hist = np.zeros((n_species, n_species, len(edges) - 1))
    density_sum = 0.0

    for x, c in zip(positions, cells):
        frac = np.linalg.solve(c.T, x.T).T % 1.0
        central = frac @ c
        images = ((frac[None] + _shifts()[:, None]).reshape(-1, 3)) @ c
        image_owner = np.tile(np.arange(n_atoms), 27)
        found = cKDTree(central).sparse_distance_matrix(cKDTree(images), r_max, output_type="ndarray")
        i, j, d = found["i"], image_owner[found["j"]], found["v"]
        keep = d > 1e-8  # drop each atom's distance to itself
        flat = (index[i[keep]] * n_species + index[j[keep]]) * (len(edges) - 1)
        bins = np.digitize(d[keep], edges) - 1
        ok = (bins >= 0) & (bins < len(edges) - 1)
        hist += np.bincount(flat[ok] + bins[ok], minlength=hist.size).reshape(hist.shape)
        density_sum += 1.0 / abs(np.linalg.det(c))

    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    # Ordered pairs (i of a, j of b); the ideal count per frame is N_a N_b shell / V (N_a - 1 for a = b).
    pair_norm = np.outer(counts_per_species, counts_per_species) - np.diag(counts_per_species)
    g = hist / (pair_norm[:, :, None] * shell[None, None, :] * density_sum)
    return r, g, labels


def ideal_gas_entropy(number_density: float, temperature: float, masses_amu, fractions) -> float:
    """
    Sackur-Tetrode entropy of an ideal gas mixture, ideal mixing included, in J/(K mol of atoms).

    Args:
        number_density: Atoms per Å³.
        temperature: Temperature in K.
        masses_amu: Mass of each species in atomic mass units.
        fractions: Mole fraction of each species.
    """
    x = np.asarray(fractions, dtype=float)
    lam = thermal_de_broglie_wavelength(masses_amu, temperature)
    present = x > 0
    return float(GAS_CONSTANT * (2.5 - np.sum(x[present] * np.log(number_density * x[present] * lam[present] ** 3))))


def two_body_excess_entropy(r: np.ndarray, g: np.ndarray, number_density: float, fractions) -> float:
    """
    Two-body excess entropy from partial pair distribution functions, in J/(K mol of atoms).

    Args:
        r: Bin centres in Å, evenly spaced.
        g: Partials, shape ``(n_species, n_species, n_bins)``.
        number_density: Atoms per Å³.
        fractions: Mole fraction of each species, in the order of ``g``.
    """
    x = np.asarray(fractions, dtype=float)
    dr = r[1] - r[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(g > 0, g * np.log(np.where(g > 0, g, 1.0)), 0.0) - g + 1.0
    integrals = np.sum(integrand * 4.0 * np.pi * r**2 * dr, axis=-1)
    return float(-GAS_CONSTANT * number_density / 2.0 * np.sum(np.outer(x, x) * integrals))


def liquid_entropy(
    positions: np.ndarray,
    cell: np.ndarray,
    species: np.ndarray,
    masses_amu: dict,
    temperature: float,
    r_max: float = None,
    dr: float = 0.02,
) -> dict:
    """
    Total classical entropy of a liquid from an equilibrated trajectory, in J/(K mol of atoms).

    Args:
        positions: Cartesian positions, shape ``(n_frames, N, 3)``, in Å, at fixed composition.
        cell: Lattice vectors as rows, shape ``(3, 3)`` or ``(n_frames, 3, 3)``, in Å.
        species: Species label of each atom, shape ``(N,)``.
        masses_amu: Mass of each species label in atomic mass units.
        temperature: Temperature in K.
        r_max: Cutoff of the pair integrals in Å; defaults to half the smallest cell width.
        dr: Bin width in Å.

    Returns:
        Dictionary with ``ideal``, ``two_body`` and ``total`` entropies, the species ``labels``,
        their ``fractions``, the ``number_density`` in Å⁻³ and the pair data ``r`` and ``g``.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 2:
        positions = positions[None]
    cells = np.broadcast_to(np.asarray(cell, dtype=float), (len(positions), 3, 3))
    species = np.asarray(species)
    r, g, labels = partial_pair_distributions(positions, cells, species, r_max=r_max, dr=dr)
    fractions = np.array([np.mean(species == s) for s in labels])
    density = float(np.mean([species.size / abs(np.linalg.det(c)) for c in cells]))
    masses = np.array([masses_amu[s] for s in labels], dtype=float)
    s_ideal = ideal_gas_entropy(density, temperature, masses, fractions)
    s_two = two_body_excess_entropy(r, g, density, fractions)
    return {"ideal": s_ideal, "two_body": s_two, "total": s_ideal + s_two, "labels": labels,
            "fractions": fractions, "number_density": density, "r": r, "g": g}


def mixing_decomposition(entropy: float, pure_entropies, fractions) -> dict:
    """
    Split the entropy of a solution as a CALPHAD model does, in the units of the input.

    ``mechanical`` is the mole-fraction average of the pure entropies (the mechanical mixture),
    ``ideal`` the ideal mixing entropy -R sum x ln x, and ``excess`` what remains, the part a
    model's excess parameters have to carry.

    Args:
        entropy: Entropy of the solution, J/(K mol of atoms).
        pure_entropies: Entropy of each pure component at the same temperature and pressure.
        fractions: Mole fraction of each component.
    """
    x = np.asarray(fractions, dtype=float)
    mechanical = float(np.dot(x, pure_entropies))
    present = x > 0
    ideal = float(-GAS_CONSTANT * np.sum(x[present] * np.log(x[present])))
    return {"mechanical": mechanical, "ideal": ideal, "excess": entropy - mechanical - ideal}
