"""Diagnostic: current Hermes and the pure Gaussian (quasi-harmonic from covariance) estimate against exact on the frozen
set, where each metal was run at two temperatures a factor of two apart."""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\busta\Code\hermes-private")
sys.path.insert(0, r"C:\Users\busta\Code\hermes-private\validation\solid_fresh")
import analyze_fresh as an  # noqa: E402

an.HERE = r"C:\Users\busta\Code\hermes-private\validation\solid_exact"
CASES = {"fe_0700": ("bcc", 55.845, 700.0), "fe_1400": ("bcc", 55.845, 1400.0), "w_1200": ("bcc", 183.84, 1200.0),
         "w_2400": ("bcc", 183.84, 2400.0), "mg_0370": ("hcp", 24.305, 370.0), "mg_0740": ("hcp", 24.305, 740.0),
         "zr_0450": ("hcp", 91.224, 450.0), "zr_0900": ("hcp", 91.224, 900.0), "al_0370": ("fcc", 26.98154, 370.0),
         "al_0740": ("fcc", 26.98154, 740.0), "ni_0700": ("fcc", 58.71, 700.0), "ni_1400": ("fcc", 58.71, 1400.0)}
from hermes.compression_entropy import crystal_correlation_entropy, thermal_de_broglie_wavelength, trajectory_entropy  # noqa: E402
from hermes.md_configuration import CLASSICAL_ENTROPY_OFFSET  # noqa: E402

R = 8.314462618
rows = {}
for tag, (lattice, mass, temperature) in CASES.items():
    exact, _ = an.exact_entropy(tag, temperature)
    frames, cell = an.read_dump(rf"C:\Users\busta\Code\hermes-private\validation\solid_exact\{tag}.dump")
    corr = crystal_correlation_entropy(frames, cell)
    comp = trajectory_entropy(frames, cell, np.full(frames.shape[1], mass), temperature, n_pairs=60, min_separation=10)
    hermes = comp["mean"] + CLASSICAL_ENTROPY_OFFSET - corr["entropy"]
    frac = np.linalg.solve(cell.T, frames.reshape(-1, 3).T).T.reshape(frames.shape)
    step = frac - frac[0]
    step -= np.round(step)
    track = (frac[0] + step) @ cell
    u = track - track.mean(0)
    u -= u.mean(1, keepdims=True)
    c = np.einsum("fia,fib->iab", u, u) / (len(u) - 1)
    lam = thermal_de_broglie_wavelength([mass], temperature)[0]
    gauss_marginal = R * np.mean(0.5 * (3 * np.log(2 * np.pi * np.e) + np.linalg.slogdet(c / lam**2)[1])) + 1.5 * R
    quasi_harmonic = gauss_marginal - corr["entropy"]
    rows[tag] = (temperature, exact, hermes - exact, quasi_harmonic - exact)
    print(f"{tag:8s} {temperature:6.0f} K  exact {exact:6.2f}  Hermes error {hermes - exact:+.2f}  Gaussian-QH error {quasi_harmonic - exact:+.2f}",
          flush=True)
print()
for metal in ("fe", "w", "mg", "zr", "al", "ni"):
    (lo, hi) = sorted(t for t in rows if t.startswith(metal + "_"))
    print(f"{metal:3s} QH error {rows[lo][3]:+.2f} at {rows[lo][0]:.0f} K -> {rows[hi][3]:+.2f} at {rows[hi][0]:.0f} K   "
          f"ratio {rows[hi][3] / rows[lo][3] if abs(rows[lo][3]) > 0.05 else float('nan'):.2f}")
json.dump(rows, open(r"C:\Users\busta\AppData\Local\Temp\claude\C--Users-busta-Code\901a81a4-bab3-4f41-bc37-cb7028beffb3\scratchpad\tdep.json", "w"))
