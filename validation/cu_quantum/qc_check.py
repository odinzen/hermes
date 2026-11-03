import numpy as np, sys
from hermes.quantum_correction import vibrational_density_of_states, harmonic_entropy, quantum_correction
def read_vel(path, n=864):
    rows = [ln.split() for ln in open(path) if not ln.startswith("ITEM")]
    v = np.array([r for r in rows if len(r) == 4], float)[:, 1:]
    return v.reshape(-1, n, 3)
RUNS = r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\runs"
for name, path, T in (("solid 298", r"qc\cu298.vel", 298.15), ("solid 1300", rf"{RUNS}\vel_solid.vel", 1300.0),
                      ("liquid 1300", rf"{RUNS}\vel_liquid.vel", 1300.0)):
    v = read_vel(path)
    nu, g = vibrational_density_of_states(v, 0.004)
    print(f"{name}: frames {len(v)}, S_q {harmonic_entropy(nu, g, T):.2f}, S_cl {harmonic_entropy(nu, g, T, quantum=False):.2f}, "
          f"correction {quantum_correction(nu, g, T):.3f}, peak {nu[g.argmax()]:.2f} THz, g(0) {g[0]:.4f}", flush=True)
