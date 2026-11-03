"""How much of Hermes's +5 J/(K mol) on solid Cu is neighbour correlation?

Per-atom coding counts each atom's displacement as if independent, so it measures the sum of the
marginal entropies; the true entropy is lower by the total correlation (multi-information). Pair
approximation, Gaussian: each nearest-neighbour pair carries I = 1/2 ln(det S_i det S_j / det S_ij)
nats, with the 6x6 covariance pooled over all pairs in each pair's bond frame. Per atom: 6 pairs.
Displacements are from each atom's time-averaged site (solid, no diffusion).
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam")
from analyze_cuzr import read_dump  # noqa: E402

R = 8.314462618
frames, _, cell = read_dump(r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\runs\fus_solid.dump")
box = np.diag(cell)
# unwrap against frame 0, then displacements from the mean site
d = frames - frames[0]
d -= box * np.round(d / box)
traj = frames[0] + d
sites = traj.mean(0)
u = traj - sites  # (n_frames, N, 3)
u -= u.mean(1, keepdims=True)  # remove centre-of-mass drift per frame

tree = cKDTree(np.mod(sites, box), boxsize=box)
pairs = np.array(sorted(tree.query_pairs(2.9)))  # fcc NN ~2.55-2.6 A, 2nd shell ~3.7 A
print("pairs per atom", 2 * len(pairs) / len(sites))


def bond_frame(v):
    e1 = v / np.linalg.norm(v)
    a = np.array([1.0, 0, 0]) if abs(e1[0]) < 0.9 else np.array([0, 1.0, 0])
    e2 = np.cross(e1, a); e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3])


# Transverse axes of a bond are not unique, so pool only rotation-invariant blocks: use the
# longitudinal component plus the transverse plane treated isotropically.
samples = []
for i, j in pairs:
    b = sites[j] - sites[i]
    b -= box * np.round(b / box)
    Q = bond_frame(b)
    samples.append(np.concatenate([u[:, i] @ Q.T, u[:, j] @ Q.T], axis=1))
X = np.concatenate(samples)  # (n_frames * n_pairs, 6)
C = np.cov(X.T)
# symmetrise the two transverse axes (rotation about the bond) and the i<->j exchange
P = np.eye(6)[[0, 2, 1, 3, 5, 4]]
C = 0.5 * (C + P @ C @ P.T)
S = np.eye(6)[[3, 4, 5, 0, 1, 2]]
C = 0.5 * (C + S @ C @ S.T)
I_pair = 0.5 * np.log(np.linalg.det(C[:3, :3]) * np.linalg.det(C[3:, 3:]) / np.linalg.det(C))
corr_long = C[0, 3] / C[0, 0]
corr_tr = C[1, 4] / C[1, 1]
print(f"NN displacement correlation: longitudinal {corr_long:.3f}, transverse {corr_tr:.3f}")
print(f"MI per pair {I_pair:.4f} nats; pair-approx total correlation per atom = {6 * I_pair * R:.2f} J/(K mol)")
