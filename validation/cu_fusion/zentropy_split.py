"""Zentropy split of liquid and solid Cu: basins (inherent structures) as the configurations.

S = S_basin_count + <S_within_basin>.
Within-basin part: Hermes compression of each MD frame against its own quenched inherent structure.
Coding u = x - x_IS instead of a pair difference x_a - x_b halves the variance, i.e. the amplitude
is 1/sqrt(2) smaller, so 3R ln sqrt(2) is added to put it on Hermes's pair-calibrated scale, then the
5.3 offset. Correlation correction: Gaussian pair mutual information of within-basin displacements
over nearest-neighbour pairs of the inherent structure (same recipe as solid_pair_mi.py).
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam")
from analyze_cuzr import read_dump  # noqa: E402

from hermes.compression_entropy import microstate_entropy, thermal_de_broglie_wavelength  # noqa: E402
from hermes.md_configuration import CLASSICAL_ENTROPY_OFFSET  # noqa: E402

R = 8.314462618
HERE = r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\e7397efa-dcb0-4ef2-878d-eb03d3cbbcfd\scratchpad"
RUNS = r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\runs"
EXACT = {"solid": 72.76, "liquid": 82.5}


def pair_mi(u, sites, box, cutoff):
    """Pooled Gaussian MI per neighbour pair (nats) and neighbours per atom."""
    tree = cKDTree(np.mod(sites, box), boxsize=box)
    pairs = np.array(sorted(tree.query_pairs(cutoff)))
    X = []
    for i, j in pairs:
        b = sites[j] - sites[i]
        b -= box * np.round(b / box)
        e1 = b / np.linalg.norm(b)
        a = np.array([1.0, 0, 0]) if abs(e1[0]) < 0.9 else np.array([0, 1.0, 0])
        e2 = np.cross(e1, a); e2 /= np.linalg.norm(e2)
        Q = np.stack([e1, e2, np.cross(e1, e2)])
        X.append(np.concatenate([u[:, i] @ Q.T, u[:, j] @ Q.T], axis=1))
    return np.concatenate(X), 2 * len(pairs) / len(sites)


out = {}
for phase in ("solid", "liquid"):
    md, _, cell = read_dump(rf"{RUNS}\fus_{phase}.dump")
    qs, _, _ = read_dump(rf"{HERE}\quench\{phase}_is.dump")
    e_is = np.loadtxt(rf"{HERE}\quench\{phase}_is_energy.txt", usecols=2)
    box = np.diag(cell)
    eps = thermal_de_broglie_wavelength(np.full(md.shape[1], 63.546), 1300.0)
    idx = range(0, len(md), 3)
    s_in = [microstate_entropy(md[k], qs[k], cell, eps) for k in idx]
    s_in = np.array(s_in) + 3 * R * np.log(np.sqrt(2)) + CLASSICAL_ENTROPY_OFFSET

    # within-basin displacements, minimum image
    u = md - qs
    u -= box * np.round(u / box)
    Xs, nb = [], []
    for k in idx:
        X, n = pair_mi(u[k:k + 1], qs[k], box, 3.1)  # first-shell minimum of g(r) for Cu ~3.1-3.3 A
        Xs.append(X); nb.append(n)
    X = np.concatenate(Xs)
    C = np.cov(X.T)
    P = np.eye(6)[[0, 2, 1, 3, 5, 4]]; C = 0.5 * (C + P @ C @ P.T)
    S = np.eye(6)[[3, 4, 5, 0, 1, 2]]; C = 0.5 * (C + S @ C @ S.T)
    I = 0.5 * np.log(np.linalg.det(C[:3, :3]) * np.linalg.det(C[3:, 3:]) / np.linalg.det(C))
    corr = (np.mean(nb) / 2) * I * R
    rms_u = float(np.sqrt((u ** 2).sum(-1).mean()))
    out[phase] = {"S_within_basin_compression": float(s_in.mean()), "sem": float(s_in.std(ddof=1) / np.sqrt(len(s_in))),
                  "neighbours_per_atom": float(np.mean(nb)), "pair_MI_correction": float(corr),
                  "S_within_basin_corrected": float(s_in.mean() - corr), "rms_u_A": rms_u,
                  "E_IS_eV_atom": float(e_is.mean()), "E_IS_spread_meV": float(1000 * e_is.std())}
    out[phase]["basin_count_needed"] = EXACT[phase] - out[phase]["S_within_basin_corrected"]
    print(phase, json.dumps(out[phase], indent=1), flush=True)
json.dump(out, open(rf"{HERE}\zentropy_split.json", "w"), indent=1)
