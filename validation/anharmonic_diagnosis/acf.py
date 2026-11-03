import sys
import numpy as np
sys.path.insert(0, r"C:\Users\busta\Code\hermes-private\validation\solid_fresh")
sys.path.insert(0, ".")
import analyze_fresh as af
import nongauss as ng
for tag in ("ni_1000_0864", "mg_0550_0800", "fe_1000_1024"):
    frames, cell = af.read_dump(rf"C:\Users\busta\Code\hermes-private\validation\solid_fresh\{tag}.dump")
    u, _ = ng.displacements(frames, cell)
    x = u.reshape(len(u), -1)
    acf = [np.mean(np.sum(x[: len(x) - lag] * x[lag:], axis=1)) / np.mean(np.sum(x * x, axis=1)) for lag in range(0, 13)]
    print(tag, " ".join(f"{a:+.3f}" for a in acf), "(lags 0..6 ps in 0.5 ps steps)", flush=True)
