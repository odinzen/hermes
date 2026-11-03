"""Controls for nongauss.py: Gaussian data (independent and time-correlated frames) must give ~0, non-Gaussian data > 0."""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad")
sys.path.insert(0, r"C:\Users\busta\Code\hermes-private\tests")
import nongauss as ng  # noqa: E402
import test_crystal_correlation as t  # noqa: E402

rng = np.random.default_rng(0)
frames = 401
sites, box = t.fcc(4)
cov, _ = t.harmonic_covariance(sites, box, einstein=0.05)
pairs, per_atom = ng.first_shell_pairs(sites, box, 300, rng)

# 1. independent Gaussian frames from a correlated crystal
u = rng.multivariate_normal(np.zeros(len(cov)), cov, size=frames, method="eigh").reshape(frames, -1, 3)
d, se = ng.excess_information(u, pairs, rng)
print(f"Gaussian, independent frames : excess {d:+.4f} +- {se:.4f} nats/pair -> {ng.R * per_atom / 2 * d:+.3f} J/(K mol) per atom")

# 3. non-Gaussian: same crystal, each pair's longitudinal motion bent by a hard-core-like nonlinearity
u = rng.multivariate_normal(np.zeros(len(cov)), cov, size=frames, method="eigh").reshape(frames, -1, 3)
u = u + 0.6 * np.roll(u, 1, axis=1) ** 2 / np.sqrt(np.mean(u**2))
d, se = ng.excess_information(u, pairs, rng)
print(f"non-Gaussian dependence      : excess {d:+.4f} +- {se:.4f} nats/pair -> {ng.R * per_atom / 2 * d:+.3f} J/(K mol) per atom")
