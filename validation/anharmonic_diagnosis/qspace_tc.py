"""Exact Gaussian total correlation of a crystal from its translation symmetry (block-diagonal in wavevector)."""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

sys.path.insert(0, r"C:\Users\busta\Code\hermes-private\tests")
import test_crystal_correlation as t  # noqa: E402

R = 8.314462618


def translations(sites, cell, species, tol=0.05):
    """Lattice translations that map the crystal onto itself, as (fractional vectors, permutations)."""
    inv = np.linalg.inv(cell)
    frac = (sites @ inv) % 1.0
    tree = cKDTree(frac, boxsize=1.0 + 1e-12)
    ftol = tol / np.linalg.norm(cell, axis=1).min()
    vecs, perms = [], []
    for j in range(len(sites)):
        if species[j] != species[0]:
            continue
        shift = frac[j] - frac[0]
        d, idx = tree.query((frac + shift) % 1.0)
        if d.max() < ftol and np.all(species[idx] == species):
            vecs.append(shift - np.round(shift))
            perms.append(idx)
    return np.array(vecs), np.array(perms)


def wishart_log_det_bias(d, nu, complex_=False):
    if complex_:
        return sum(digamma(nu - i) for i in range(d)) - d * np.log(nu)
    return d * np.log(2.0 / nu) + sum(digamma((nu - i + 1) / 2.0) for i in range(1, d + 1))


def crystal_total_correlation(u, sites, cell, species, correct=True):
    """u: displacements (frames, N, 3) with the centre of mass removed. Returns nats per atom."""
    f, n, _ = u.shape
    u = u - u.mean(0)  # displacements from mean sites average to zero exactly
    vecs, perms = translations(sites, cell, species)
    nt = len(vecs)
    if n % nt:
        raise ValueError("translations do not tile the cell")
    # one representative per orbit = the basis of the primitive cell
    seen, basis = np.zeros(n, bool), []
    for i in range(n):
        if not seen[i]:
            basis.append(i)
            seen[perms[:, i]] = True
    m = len(basis)
    # distinct characters exp(2 pi i H . f_t) of the translation group
    denom = [int(round(1 / np.min(np.abs(c[np.abs(c) > 1e-6])))) if np.any(np.abs(vecs[:, k]) > 1e-6) else 1
             for k, c in enumerate(vecs.T)]
    grid = np.array(np.meshgrid(*[np.arange(d) for d in denom], indexing="ij")).reshape(3, -1).T
    phase = np.exp(2j * np.pi * grid @ vecs.T)  # (n_H, nt)
    _, keep = np.unique(np.round(phase * 1e6).view(np.float64).reshape(len(grid), -1), axis=0, return_index=True)
    chars = phase[np.sort(keep)]
    if len(chars) != nt:
        raise ValueError(f"found {len(chars)} distinct wavevectors for {nt} translations")
    nu = f - 1
    # U[q] for each frame: (frames, nt, 3m)
    idx = perms[:, basis]  # (nt, m): atom at basis b translated by t
    ub = u[:, idx, :].reshape(f, nt, 3 * m)  # (frames, nt, 3m)
    uq = np.einsum("kt,ftd->fkd", chars.conj(), ub) / np.sqrt(nt)
    cq = np.einsum("fka,fkb->kab", uq, uq.conj()) / nu
    logdet_joint = 0.0
    for k in range(nt):
        w = np.linalg.eigvalsh(cq[k])
        real = np.allclose(chars[k].imag, 0.0)
        if np.allclose(chars[k], 1.0):
            w, d = w[3:], 3 * m - 3  # centre-of-mass modes
        else:
            d = 3 * m
        if d:
            logdet_joint += np.sum(np.log(w)) - (wishart_log_det_bias(d, nu, complex_=not real) if correct else 0.0)
    xc = u - u.mean(0)
    marg = 0.0
    for i in range(n):
        c = xc[:, i].T @ xc[:, i] / nu
        marg += np.linalg.slogdet(c)[1] - (wishart_log_det_bias(3, nu) if correct else 0.0)
    return 0.5 * (marg - logdet_joint) / n, m, nt


if __name__ == "__main__":
    for name, (sites, box) in (("fcc 108", t.fcc(3)), ("fcc 256", t.fcc(4)), ("bcc 128", t.bcc(4)), ("hcp 128", t.hcp(4)), ("fcc 864", t.fcc(6))):
        for einstein in ((0.3,) if name.startswith("bcc") else (0.3, 1e-9)):
            cov, cutoff = t.harmonic_covariance(sites, box, einstein=einstein)
            n = len(sites)
            w = np.linalg.eigvalsh(cov)
            exact = 0.5 * (sum(np.linalg.slogdet(cov[3 * k:3 * k + 3, 3 * k:3 * k + 3])[1] for k in range(n))
                           - np.sum(np.log(w[3:]))) / n
            species = np.zeros(n, int)
            rng = np.random.default_rng(1)
            line = f"{name} einstein {einstein:.1g}: exact {R * exact:.4f}"
            for frames in (60, 200, 1000):
                vals = []
                for rep in range(4):
                    u = rng.multivariate_normal(np.zeros(3 * n), cov, size=frames, method="eigh").reshape(frames, n, 3)
                    u -= u.mean(1, keepdims=True)
                    tc, m, nt = crystal_total_correlation(u, sites, box, species)
                    vals.append(tc)
                line += f" | {frames} fr {R * np.mean(vals):.4f}+-{R * np.std(vals) / 2:.4f}"
            print(line + f" (basis {m}, {nt} cells)", flush=True)
