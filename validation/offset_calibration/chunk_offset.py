"""Compress fixed groups of 256 atoms: is the Einstein offset then independent of N and flat in amplitude?"""
import sys

import numpy as np
import scipy.constants

sys.path.insert(0, r"C:\Users\busta\Code\hermes-private")
import hermes.compression_entropy as ce  # noqa: E402

R = scipy.constants.R
CHUNK = 256


def chunked_entropy(target, reference, cell, eps, rng):
    res = ce.displacement_residuals(target, reference, cell)
    order = rng.permutation(len(res))
    vals = []
    for g in range(len(res) // CHUNK):
        sub = res[order[g * CHUNK:(g + 1) * CHUNK]]
        per = []
        for alpha in 3.0 ** (np.arange(8) / 8):
            _, coef = ce.compress(alpha * sub, eps)
            per.append(max(0.0, R * (np.log(2) * ce.coefficient_bits(coef) / CHUNK - 3 * np.log(alpha))))
        vals.append(np.mean(per))
    return float(np.mean(vals))


if __name__ == "__main__":
    mass, temperature = 26.98, 1500.0
    lam = ce.thermal_de_broglie_wavelength([mass], temperature)[0]
    rng = np.random.default_rng(3)
    for freq in (12e12, 6e12, 3e12):
        omega = 2 * np.pi * freq
        sigma = np.sqrt(scipy.constants.k * temperature / (mass * scipy.constants.atomic_mass * omega**2)) / 1e-10
        exact = 3 * R * (1 + np.log(scipy.constants.k * temperature / (scipy.constants.hbar * omega)))
        for n in (7, 10, 13):
            sites = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)], float) * 4.05
            cell = np.eye(3) * n * 4.05
            reps = max(2, 6000 // n**3)
            vals = [chunked_entropy(sites + rng.normal(0, sigma, sites.shape), sites + rng.normal(0, sigma, sites.shape), cell, lam, rng)
                    for _ in range(reps)]
            print(f"sigma/lam {sigma / lam:.2f} N {n**3:5d}: offset {exact - np.mean(vals):5.2f} +- {np.std(vals, ddof=1) / np.sqrt(reps):.2f}",
                  flush=True)
