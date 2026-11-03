"""Liquid Cu: ideal gas + two-body excess entropy, S = S_id + S2, S2 = -(rho/2) int [g ln g - g + 1] dV.

Exact (calphy, same potential): 82.5 J/(K mol). Total entropy incl. kinetic, classical.
"""
import sys

import numpy as np
import scipy.constants as sc

sys.path.insert(0, r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam")
from analyze_cuzr import read_dump  # noqa: E402

R = sc.R
frames, _, cell = read_dump(r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\runs\fus_liquid.dump")
box = np.diag(cell)
N = frames.shape[1]
V = np.prod(box)
rho = N / V
T, m = 1300.0, 63.546 * sc.atomic_mass
lam = sc.h / np.sqrt(2 * np.pi * m * sc.k * T) * 1e10  # A
S_id = R * (np.log(1 / (rho * lam ** 3)) + 2.5)

rmax, dr = box[0] / 2, 0.02
edges = np.arange(0, rmax + dr, dr)
hist = np.zeros(len(edges) - 1)
for x in frames:
    d = x[:, None, :] - x[None, :, :]
    d -= box * np.round(d / box)
    r = np.sqrt((d ** 2).sum(-1))[np.triu_indices(N, 1)]
    hist += np.histogram(r, edges)[0]
shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
g = 2 * hist / (len(frames) * N * rho * shell)
rc = 0.5 * (edges[1:] + edges[:-1])
with np.errstate(divide="ignore", invalid="ignore"):
    f = np.where(g > 0, g * np.log(g), 0.0) - g + 1
S2 = -R * rho / 2 * np.sum(f * 4 * np.pi * rc ** 2 * dr)
print(f"rho {rho:.4f} /A3, lambda {lam:.4f} A, first peak g={g.max():.2f} at {rc[g.argmax()]:.2f} A")
print(f"S_ideal {S_id:.2f}, S2 {S2:.2f}, S_id+S2 {S_id + S2:.2f} vs exact 82.5 J/(K mol)")
