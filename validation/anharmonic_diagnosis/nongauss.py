"""Non-Gaussian shared information between neighbouring atoms: KSG mutual information on MD vs phase-randomised surrogates.

For each first-shell pair, the 6-channel displacement series (u_i, u_j) over the frames is compared with independent
Gaussian samples of the same size and the pair's own sample covariance, so the surrogate keeps every linear
correlation and nothing else. MD frames 0.5 ps apart are effectively independent (displacement autocorrelation
under 0.06 at every lag), which a phase-randomised surrogate would only be needed for. The Kraskov-Stoegbauer-Grassberger estimator (k = 4, max norm) is applied to both; its
bias and the effect of time-correlated frames are the same for the two, so their difference is the information the
pair shares beyond the Gaussian part.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

sys.path.insert(0, r"C:\Users\busta\Code\hermes-private")
from hermes.compression_entropy import _nearest_neighbour_distance  # noqa: E402

R = 8.314462618


def ksg(x, y, k=4):
    """KSG estimator 1 of I(X;Y) in nats."""
    xy = np.hstack([x, y])
    n = len(xy)
    eps = cKDTree(xy).query(xy, k=k + 1, p=np.inf)[0][:, -1]
    tx, ty = cKDTree(x), cKDTree(y)
    nx = np.array([len(tx.query_ball_point(x[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(ty.query_ball_point(y[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    return digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))


def surrogate(z, rng):
    """Phase-randomised copy of a multichannel series z (frames, channels), same phases for every channel."""
    f = np.fft.rfft(z, axis=0)
    phases = np.exp(2j * np.pi * rng.uniform(size=f.shape[0]))
    phases[0] = 1.0
    if z.shape[0] % 2 == 0:
        phases[-1] = 1.0
    return np.fft.irfft(f * phases[:, None], n=z.shape[0], axis=0)


def displacements(frames, cell):
    frac = np.linalg.solve(cell.T, frames.reshape(-1, 3).T).T.reshape(frames.shape)
    step = frac - frac[0]
    step -= np.round(step)
    track = (frac[0] + step) @ cell
    u = track - track.mean(0)
    return u - u.mean(1, keepdims=True), track.mean(0)


def first_shell_pairs(sites, cell, n_pairs, rng):
    d_nn = _nearest_neighbour_distance(sites, cell)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    images = (sites[None] + (shifts @ cell)[:, None]).reshape(-1, 3)
    hits = cKDTree(images).query_ball_point(sites, 1.15 * d_nn)
    pairs = sorted({(i, h % len(sites)) for i, hs in enumerate(hits) for h in hs if h % len(sites) > i})
    per_atom = 2 * len(pairs) / len(sites)
    pick = rng.choice(len(pairs), size=min(n_pairs, len(pairs)), replace=False)
    return [pairs[p] for p in pick], per_atom


def excess_information(u, pairs, rng, repeats=2):
    """Mean over pairs of KSG(data) - KSG(surrogate), nats, and its standard error."""
    diffs = []
    for i, j in pairs:
        z = np.hstack([u[:, i], u[:, j]])
        data = ksg(z[:, :3], z[:, 3:])
        c = np.cov(z.T)
        surr = np.mean([ksg(*np.hsplit(rng.multivariate_normal(np.zeros(6), c, size=len(z)), 2)) for _ in range(repeats)])
        diffs.append(data - surr)
    diffs = np.array(diffs)
    return diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(diffs))
