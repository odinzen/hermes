"""Quantum correction to a classical entropy, from the vibrational spectrum of an MD run.

Classical MD gives every vibrational mode k_B of heat capacity at any temperature, so its entropy
falls below the quantum one, without bound as T goes to 0 (the classical oscillator entropy
1 - ln x is unbounded below). Treating each mode of the spectrum as a harmonic oscillator, the
correction is

    S_q - S_cl = R int g(nu) [s_q(x) - s_cl(x)] d nu,    x = h nu / k_B T
    s_q = x / (e^x - 1) - ln(1 - e^-x),    s_cl = 1 - ln x

with g the vibrational density of states normalised to 3 modes per atom. The difference is positive
and goes as x^2/24 for small x, so the diffusive, zero-frequency weight of a liquid contributes nothing and the
correction is well defined there too. Anharmonicity enters only through the spectrum itself, the
usual approximation; the classical entropy it is added to keeps its full anharmonic content.

On the Mendelev EAM for fcc Cu at 298.15 K (864 atoms, 20 ps) the quantum harmonic entropy of the
spectrum is 32.98 J/(K mol) against 32.94 for the lattice (CODATA S = 33.150 J/(K mol), less the
electronic gamma T = 0.21); the classical value is 32.30. At 1300 K the correction is 0.03.
"""

# Standard Library Imports
import scipy.constants

# Third-Party Library Imports
import numpy as np

GAS_CONSTANT = scipy.constants.R
# h / k_B in K per THz
H_OVER_KB = scipy.constants.h * 1e12 / scipy.constants.k


def vibrational_density_of_states(
    velocities: np.ndarray, timestep_ps: float, masses_amu=None, segment_ps: float = 5.0
) -> tuple:
    """
    Vibrational density of states from the power spectrum of the atomic velocities.

    The trajectory is cut into half-overlapping, Hann-windowed segments whose spectra are averaged.
    Each atom is weighted by its mass, so the spectrum is that of the kinetic energy.

    Args:
        velocities: Velocities, shape ``(n_frames, N, 3)``, any unit.
        timestep_ps: Time between frames in ps.
        masses_amu: Atomic masses, shape ``(N,)``; equal masses if omitted.
        segment_ps: Segment length in ps, which sets the frequency resolution 1 / segment_ps.

    Returns:
        ``(nu, dos)``: frequencies in THz and the density of states in 1/THz, integrating to 3.
    """
    v = np.asarray(velocities, dtype=float)
    n_frames = len(v)
    length = int(round(segment_ps / timestep_ps))
    if length > n_frames:
        raise ValueError(f"segment of {length} frames is longer than the trajectory ({n_frames})")
    weight = np.ones(v.shape[1]) if masses_amu is None else np.asarray(masses_amu, dtype=float)
    window = np.hanning(length)[:, None, None]
    power, count = 0.0, 0
    for start in range(0, n_frames - length + 1, max(1, length // 2)):
        spectrum = np.abs(np.fft.rfft(v[start:start + length] * window, axis=0)) ** 2
        power = power + np.einsum("fnk,n->f", spectrum, weight)
        count += 1
    nu = np.fft.rfftfreq(length, d=timestep_ps)
    dos = power / count
    return nu, 3.0 * dos / np.trapezoid(dos, nu)


def _mode_entropies(nu: np.ndarray, temperature: float) -> tuple:
    x = H_OVER_KB * np.asarray(nu, dtype=float) / temperature
    s_q = np.zeros_like(x)
    s_cl = np.zeros_like(x)
    m = x > 0
    s_q[m] = x[m] / np.expm1(x[m]) - np.log(-np.expm1(-x[m]))
    s_cl[m] = 1.0 - np.log(x[m])
    return s_q, s_cl, m


def harmonic_entropy(nu: np.ndarray, dos: np.ndarray, temperature: float, quantum: bool = True) -> float:
    """
    Entropy of the modes of a density of states as harmonic oscillators, in J/(K mol of atoms).

    Args:
        nu: Frequencies in THz.
        dos: Density of states in 1/THz, integrating to 3 per atom.
        temperature: Temperature in K.
        quantum: Quantum (Bose-Einstein) oscillators if True, classical if False.
    """
    s_q, s_cl, m = _mode_entropies(nu, temperature)
    s = s_q if quantum else s_cl
    return float(GAS_CONSTANT * np.trapezoid(np.where(m, dos * s, 0.0), nu))


def quantum_correction(nu: np.ndarray, dos: np.ndarray, temperature: float) -> float:
    """
    Quantum minus classical harmonic entropy of a density of states, in J/(K mol of atoms).

    Add it to a classical entropy (compression, liquid route or thermodynamic integration) of the
    same system at the same temperature. It is positive and tends to zero at high temperature.
    """
    s_q, s_cl, m = _mode_entropies(nu, temperature)
    return float(GAS_CONSTANT * np.trapezoid(np.where(m, dos * (s_q - s_cl), 0.0), nu))
