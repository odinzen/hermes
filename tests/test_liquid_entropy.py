import numpy as np
import pytest

from hermes.compression_entropy import GAS_CONSTANT
from hermes.liquid_entropy import (
    ideal_gas_entropy,
    liquid_entropy,
    mixing_decomposition,
    partial_pair_distributions,
)

R = GAS_CONSTANT


def ideal_gas_frames(n_frames=20, n_atoms=400, box=20.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, box, (n_frames, n_atoms, 3)), np.eye(3) * box


def jittered_fcc(n_frames=6, reps=4, a=3.61, sigma=0.12, seed=1):
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    cells = np.array([[i, j, k] for i in range(reps) for j in range(reps) for k in range(reps)])
    sites = ((cells[:, None, :] + basis[None]).reshape(-1, 3)) * a
    rng = np.random.default_rng(seed)
    return sites + rng.normal(0, sigma, (n_frames, *sites.shape)), np.eye(3) * reps * a


def test_ideal_gas_has_no_excess():
    x, cell = ideal_gas_frames()
    result = liquid_entropy(x, cell, np.zeros(x.shape[1], int), {0: 40.0}, 300.0, dr=0.1)
    # Uncorrelated positions: g = 1 up to sampling noise, which biases S_2 slightly negative.
    assert abs(result["two_body"]) < 0.02 * R
    assert result["total"] == pytest.approx(result["ideal"], abs=0.02 * R)


def test_ideal_mixing_of_identical_species():
    x, cell = ideal_gas_frames()
    pure = liquid_entropy(x, cell, np.zeros(x.shape[1], int), {0: 40.0, 1: 40.0}, 300.0, dr=0.1)
    labels = np.arange(x.shape[1]) % 2
    mixed = liquid_entropy(x, cell, labels, {0: 40.0, 1: 40.0}, 300.0, dr=0.1)
    assert mixed["ideal"] - pure["ideal"] == pytest.approx(R * np.log(2), rel=1e-12)
    assert abs(mixed["two_body"] - pure["two_body"]) < 0.03 * R


def test_pair_distribution_tends_to_one_for_ideal_gas():
    x, cell = ideal_gas_frames()
    r, g, _ = partial_pair_distributions(x, cell, np.zeros(x.shape[1], int), dr=0.25)
    assert np.mean(g[0, 0][r > 2]) == pytest.approx(1.0, abs=0.01)


def test_result_does_not_depend_on_bin_width():
    # Half the cell (7.22 A) is not a multiple of these widths; a part-filled last bin used to shift
    # S_2 by about 0.1 R. Broad, liquid-like peaks, so binning itself does not blur them.
    x, cell = jittered_fcc(n_frames=40, sigma=0.35)
    results = [liquid_entropy(x, cell, np.zeros(x.shape[1], int), {0: 63.546}, 1000.0, dr=dr)
               for dr in (0.02, 0.03, 0.05)]
    for res, dr in zip(results, (0.02, 0.03, 0.05)):
        assert res["r"][-1] + dr / 2 <= 7.22 + 1e-9
    assert np.ptp([res["two_body"] for res in results]) < 0.02 * R


def test_cutoff_beyond_half_cell_is_refused():
    x, cell = ideal_gas_frames(n_frames=1)
    with pytest.raises(ValueError):
        partial_pair_distributions(x, cell, np.zeros(x.shape[1], int), r_max=10.5)


def test_mixing_decomposition():
    parts = mixing_decomposition(10.0, [6.0, 8.0], [0.25, 0.75])
    assert parts["mechanical"] == pytest.approx(7.5)
    assert parts["ideal"] == pytest.approx(-R * (0.25 * np.log(0.25) + 0.75 * np.log(0.75)))
    assert parts["mechanical"] + parts["ideal"] + parts["excess"] == pytest.approx(10.0)


def test_ideal_gas_entropy_of_argon_at_standard_conditions():
    # Sackur-Tetrode for Ar at 298.15 K and 1 bar is 154.8 J/(K mol), the measured standard entropy.
    density = 1e5 / (1.380649e-23 * 298.15) * 1e-30
    assert ideal_gas_entropy(density, 298.15, [39.948], [1.0]) == pytest.approx(154.8, abs=0.1)


def test_pair_distribution_of_a_few_atom_ideal_gas_is_one():
    # with 16 atoms, normalising by N_a instead of N_a - 1 would read 15/16
    positions, cell = ideal_gas_frames(n_frames=3000, n_atoms=16, box=10.0, seed=11)
    r, g, _ = partial_pair_distributions(positions, cell, np.zeros(16, int), dr=0.25)
    assert g[0, 0, r > 1.0].mean() == pytest.approx(1.0, abs=0.01)
