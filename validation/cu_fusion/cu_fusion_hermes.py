"""Hermes compression entropy on the 2026-10-02 Cu fusion red-team trajectories.

Mendelev EAM Cu, 864 atoms, 1300 K, NVT frames every 0.5 ps for 60 ps.
Exact (calphy): solid 72.76, liquid 82.5 (from the red-team record), dS_fus 9.78 J/(K mol).
Hermes total S = compression entropy + CLASSICAL_ENTROPY_OFFSET (5.3).
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam")
from analyze_cuzr import read_dump  # noqa: E402

from hermes.compression_entropy import trajectory_entropy  # noqa: E402
from hermes.md_configuration import CLASSICAL_ENTROPY_OFFSET  # noqa: E402

RUNS = r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\runs"
N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MIN_SEP = 10  # frames, 5 ps

out = {}
for phase in ("solid", "liquid"):
    frames, types, cell = read_dump(rf"{RUNS}\fus_{phase}.dump")
    masses = np.full(frames.shape[1], 63.546)
    t0 = time.time()
    r = trajectory_entropy(frames, cell, masses, 1300.0, n_pairs=N_PAIRS, min_separation=MIN_SEP)
    out[phase] = {"n_frames": len(frames), "n_atoms": frames.shape[1],
                  "S_mean": r["mean"] + CLASSICAL_ENTROPY_OFFSET,
                  "S_sem": float(r["samples"].std(ddof=1) / np.sqrt(len(r["samples"]))) if N_PAIRS > 1 else None,
                  "S_min": r["minimum"] + CLASSICAL_ENTROPY_OFFSET,
                  "S_gamma_lower": r["gamma_lower_bound"] + CLASSICAL_ENTROPY_OFFSET,
                  "seconds": round(time.time() - t0, 1)}
    print(phase, out[phase], flush=True)

exact = {"solid": 72.76, "liquid": 82.5, "dS_fus": 9.78}
out["dS_fus_mean"] = out["liquid"]["S_mean"] - out["solid"]["S_mean"]
out["exact"] = exact
out["ratio_dS"] = out["dS_fus_mean"] / exact["dS_fus"]
print(json.dumps({k: out[k] for k in ("dS_fus_mean", "ratio_dS")}, indent=1))
json.dump(out, open(rf"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\e7397efa-dcb0-4ef2-878d-eb03d3cbbcfd\scratchpad\cu_fusion_hermes_{N_PAIRS}.json", "w"), indent=1)
