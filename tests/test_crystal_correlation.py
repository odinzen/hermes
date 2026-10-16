import numpy as np
import pytest

from hermes.compression_entropy import GAS_CONSTANT, _log_det_bias, crystal_correlation_entropy, crystal_translations


def fcc(n, a=3.6):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    return a * (cells[:, None, :] + base[None]).reshape(-1, 3), a * n * np.eye(3)


def bcc(n, a=3.0):
    base = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    return a * (cells[:, None, :] + base[None]).reshape(-1, 3), a * n * np.eye(3)


def hcp(n, a=2.95, c_over_a=1.633):
    c = c_over_a * a
    cell = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    base = np.array([[0, 0, 0], [1 / 3, 2 / 3, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = (cells[:, None, :] + base[None]).reshape(-1, 3)
    return frac @ cell, n * cell


def harmonic_covariance(sites, box, spring=1.0, einstein=0.3):
    """Displacement covariance of a lattice with longitudinal nearest-neighbour springs and an on-site term."""
    n = len(sites)
    cutoff = 1.1 * nearest_distance(sites, box)
    k = einstein * np.eye(3 * n)
    for i in range(n):
        d = sites - sites[i]
        d -= np.round(d @ np.linalg.inv(box)) @ box
        for j in np.flatnonzero((np.linalg.norm(d, axis=1) < cutoff) & (np.arange(n) != i)):
            e = d[j] / np.linalg.norm(d[j])
            block = spring * np.outer(e, e)
            k[3 * i:3 * i + 3, 3 * i:3 * i + 3] += block
            k[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= block
    # MD holds the centre of mass still, and the estimator removes each frame's mean displacement
    centre = np.eye(3 * n) - np.kron(np.ones((n, n)) / n, np.eye(3))
    return centre @ np.linalg.inv(k) @ centre, cutoff


def nearest_distance(sites, box):
    d = sites[1:] - sites[0]
    d -= np.round(d @ np.linalg.inv(box)) @ box
    return np.linalg.norm(d, axis=1).min()


def exact_total_correlation(cov):
    """Gaussian total correlation per atom, J/(K mol); the centre-of-mass modes get the mean coordinate variance."""
    n = len(cov) // 3
    marginal = sum(np.linalg.slogdet(cov[3 * k:3 * k + 3, 3 * k:3 * k + 3])[1] for k in range(n))
    centre = np.kron(np.ones((n, 1)) / np.sqrt(n), np.eye(3))
    full = cov + np.trace(cov) / (3 * n) * centre @ centre.T
    return GAS_CONSTANT * 0.5 * (marginal - np.linalg.slogdet(full)[1]) / n


def sample(sites, cov, frames, seed, scale=0.1):
    rng = np.random.default_rng(seed)
    u = rng.multivariate_normal(np.zeros(len(cov)), cov, size=frames, method="eigh").reshape(frames, -1, 3)
    return sites + scale * u


@pytest.mark.parametrize("make, cells, basis", [(lambda: fcc(3), 108, 1), (lambda: bcc(4), 128, 1),
                                                 (lambda: hcp(4), 64, 2)])
def test_translations_find_the_primitive_cell(make, cells, basis):
    sites, box = make()
    pos = sites + 1e-3 * np.random.default_rng(0).normal(size=(20,) + sites.shape)
    out = crystal_correlation_entropy(pos, box)
    assert (out["cells"], out["basis"]) == (cells, basis)


def test_triclinic_primitive_cell():
    a = 3.6
    prim = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    cells = np.array([[i, j, k] for i in range(4) for j in range(4) for k in range(4)], dtype=float)
    vectors, _ = crystal_translations(cells @ prim, 4 * prim, np.zeros(64, int))
    assert len(vectors) == 64


@pytest.mark.parametrize("make, einstein", [(lambda: fcc(4), 1e-9), (lambda: fcc(3), 0.3), (lambda: hcp(4), 1e-9),
                                            (lambda: bcc(4), 0.3)])
def test_recovers_the_exact_total_correlation(make, einstein):
    sites, box = make()
    cov, _ = harmonic_covariance(sites, box, einstein=einstein)
    exact = exact_total_correlation(cov)
    got = crystal_correlation_entropy(sample(sites, cov, 1500, seed=2), box)["shared"]
    assert got == pytest.approx(exact, abs=0.05)


def test_unbiased_with_few_frames():
    sites, box = fcc(3)
    cov, _ = harmonic_covariance(sites, box, einstein=1e-9)
    exact = exact_total_correlation(cov)
    values = [crystal_correlation_entropy(sample(sites, cov, 40, seed=s), box)["shared"] for s in range(16)]
    # without the Wishart corrections 40 frames read about 1.5 J/(K mol) low
    assert np.mean(values) == pytest.approx(exact, abs=4 * np.std(values) / np.sqrt(len(values)) + 0.03)


def test_log_det_corrections_are_unbiased_for_tiny_samples():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    sigma = a @ a.conj().T + np.eye(6)
    for complex_valued in (False, True):
        s = sigma.real if not complex_valued else sigma
        n = 10
        z = rng.normal(size=(40_000, n, 6)) + (1j * rng.normal(size=(40_000, n, 6)) if complex_valued else 0)
        if complex_valued:
            z /= np.sqrt(2)
        x = z @ np.linalg.cholesky(s).T
        if not complex_valued:
            x -= x.mean(axis=1, keepdims=True)
        nu = n - 1 if not complex_valued else n
        cov = np.einsum("tna,tnb->tab", x, x.conj()) / nu
        got = np.linalg.slogdet(cov)[1] - _log_det_bias(6, nu, complex_valued)
        assert got.mean() == pytest.approx(np.linalg.slogdet(s)[1], abs=4 * got.std() / np.sqrt(len(got)))


def test_whole_crystal_drift_and_wrapping_change_nothing():
    sites, box = fcc(3)
    cov, _ = harmonic_covariance(sites, box, einstein=1e-9)
    pos = sample(sites, cov, 200, seed=5)
    base = crystal_correlation_entropy(pos, box)["shared"]
    drift = pos + np.random.default_rng(6).normal(scale=0.5, size=(200, 1, 3))
    assert crystal_correlation_entropy(drift, box)["shared"] == pytest.approx(base, rel=1e-9)
    wrapped = np.mod(pos - 0.05, box[0, 0])  # atoms on the faces cross the boundary from frame to frame
    assert crystal_correlation_entropy(wrapped, box)["shared"] == pytest.approx(base, rel=1e-9)


def test_independent_atoms_give_back_what_the_fixed_centre_of_mass_took():
    # holding the centre of mass still narrows each atom by (1 - 1/N); the term returns that, -3R/(2N)
    sites, box = bcc(6)
    pos = sites + 0.1 * np.random.default_rng(3).normal(size=(400,) + sites.shape)
    got = crystal_correlation_entropy(pos, box)["shared"]
    assert got == pytest.approx(-1.5 * GAS_CONSTANT / len(sites), abs=0.01)


def test_refuses_a_liquid():
    sites, box = fcc(3)
    pos = np.random.default_rng(4).uniform(0, box[0, 0], size=(20,) + sites.shape)
    with pytest.raises(ValueError, match="not a solid"):
        crystal_correlation_entropy(pos, box)


def test_refuses_a_random_alloy():
    sites, box = fcc(3)
    species = np.random.default_rng(9).integers(0, 2, len(sites))
    pos = sites + 0.05 * np.random.default_rng(10).normal(size=(100,) + sites.shape)
    with pytest.raises(ValueError, match="random alloy"):
        crystal_correlation_entropy(pos, box, species)


def test_ordered_compound_keeps_its_translations():
    # CsCl (B2) order on a bcc lattice: a simple cubic lattice with a two-atom basis
    sites, box = bcc(4)
    species = np.array([0, 1] * 64)
    pos = sites + 1e-3 * np.random.default_rng(11).normal(size=(30,) + sites.shape)
    out = crystal_correlation_entropy(pos, box, species)
    assert (out["cells"], out["basis"]) == (64, 2)


def test_units_do_not_matter():
    sites, box = fcc(3)
    cov, _ = harmonic_covariance(sites, box, einstein=0.3)
    u = sample(sites, cov, 300, seed=12) - sites
    a = crystal_correlation_entropy(sites + u, box)["shared"]
    b = crystal_correlation_entropy(sites + 0.1 * u, box)["shared"]
    assert a == pytest.approx(b, rel=1e-9)


def test_centre_of_mass_entropy_of_independent_atoms():
    sites, box = bcc(6)
    sigma, n = 0.1, len(sites)
    out = crystal_correlation_entropy(sites + sigma * np.random.default_rng(13).normal(size=(400,) + sites.shape), box)
    volume = abs(np.linalg.det(box))
    expected = GAS_CONSTANT / n * np.log(volume * (n / (2 * np.pi * np.e * sigma**2)) ** 1.5)
    assert out["centre_of_mass"] == pytest.approx(expected, abs=0.005)
    assert out["entropy"] == pytest.approx(out["shared"] - out["centre_of_mass"], rel=1e-12)


@pytest.mark.parametrize("mass, frequency, temperature", [(26.98, 6e12, 300.0), (183.84, 5e12, 1500.0)])
def test_gaussian_entropy_of_an_einstein_solid(mass, frequency, temperature):
    import scipy.constants as sc
    from hermes.compression_entropy import gaussian_entropy

    omega = 2 * np.pi * frequency
    sigma = np.sqrt(sc.k * temperature / (mass * sc.atomic_mass * omega**2)) / 1e-10
    sites, box = bcc(6)
    pos = sites + sigma * np.random.default_rng(15).normal(size=(400,) + sites.shape)
    exact = 3 * GAS_CONSTANT * (1 + np.log(sc.k * temperature / (sc.hbar * omega)))
    out = gaussian_entropy(pos, box, np.full(len(sites), mass), temperature)
    # holding the centre of mass still narrows each atom by 1 - 1/N, which the marginal must show exactly
    n = len(sites)
    assert out["marginal"] == pytest.approx(exact + 1.5 * GAS_CONSTANT * np.log(1 - 1 / n), abs=0.03)
    assert out["entropy"] == pytest.approx(out["marginal"] - out["correlation"], rel=1e-12)
