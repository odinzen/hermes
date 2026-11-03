"""Is each atom's own 3D motion non-Gaussian? Kozachenko-Leonenko entropy of the MD marginal vs a Gaussian surrogate with
the atom's own covariance (same n, same estimator, so the estimator bias cancels). Negative = less entropy than Gaussian.
Controls: Gaussian data must give ~0; a known non-Gaussian marginal must give its exact negentropy."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

sys.path.insert(0, r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad")
sys.path.insert(0, r"C:\Users\busta\Code\hermes-private\validation\solid_fresh")
import analyze_fresh as af  # noqa: E402
import nongauss as ng  # noqa: E402

R = 8.314462618


def kl_entropy(x, k=4):
    n, d = x.shape
    eps = cKDTree(x).query(x, k=k + 1, p=np.inf)[0][:, -1]
    return digamma(n) - digamma(k) + d * np.log(2) + d * np.mean(np.log(eps))


def negentropy(u, atoms, rng, repeats=2):
    diffs = []
    for i in atoms:
        x = u[:, i]
        g = np.mean([kl_entropy(rng.multivariate_normal(np.zeros(3), np.cov(x.T), size=len(x))) for _ in range(repeats)])
        diffs.append(kl_entropy(x) - g)
    return R * np.mean(diffs), R * np.std(diffs, ddof=1) / np.sqrt(len(diffs))


rng = np.random.default_rng(5)
# controls
x = rng.normal(size=(401, 600, 3))
print("control Gaussian: %+.3f +- %.3f J/(K mol)" % negentropy(x, range(600), rng))
x = rng.uniform(-1, 1, size=(401, 600, 3))  # uniform cube: exact negentropy 3 * (ln 2 - 0.5 ln(2 pi e / 3)) nats
exact = 3 * R * (np.log(2) - 0.5 * np.log(2 * np.pi * np.e / 3))
print("control uniform : %+.3f +- %.3f J/(K mol), exact %+.3f" % (*negentropy(x, range(600), rng), exact))
res = json.load(open(r"C:\Users\busta\Code\hermes-private\validation\solid_fresh\fresh_results.json"))
for tag in af.CASES:
    frames, cell = af.read_dump(rf"C:\Users\busta\Code\hermes-private\validation\solid_fresh\{tag}.dump")
    u, _ = ng.displacements(frames, cell)
    atoms = rng.choice(frames.shape[1], size=min(600, frames.shape[1]), replace=False)
    m, se = negentropy(u, atoms, rng)
    print(f"{tag:13s} marginal non-Gaussian entropy {m:+.3f} +- {se:.3f} J/(K mol)", flush=True)
