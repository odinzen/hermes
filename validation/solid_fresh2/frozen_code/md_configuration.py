"""Configurations from molecular dynamics, with entropy from compressibility.

Builds a :class:`~hermes.configuration.Configuration` from classical MD averages on a grid of
temperatures and volumes, so a configuration's intra-configurational entropy can come from MD
(anharmonic, any temperature at which classical MD holds) instead of phonons. The entropy is the
compression entropy of :mod:`hermes.compression_entropy` plus a fixed offset that turns it into
the total classical entropy; F, its volume derivatives and the heat capacity follow from the MD
energies and pressures:

    S = S_compression + offset - S_correlation (+ S_electronic)
    E = <U> + 3/2 N k_B T
    F = E - T S,   dF/dV = -<P>,   d2F/dV2 = -d<P>/dV,   C_V = dE/dT

The offset was measured on exactly sampled Einstein solids, whose classical entropy is
3 R (1 + ln(k_B T / hbar omega)), coded as :func:`hermes.compression_entropy.microstate_entropy` codes
an MD frame: epsilon at the thermal de Broglie wavelength, eight-fold dither, groups of 256 atoms.
Over 60 solids (masses 12 to 180, amplitudes of 1 to 4.5 wavelengths, 343 to 2197 atoms) the
compression entropy falls below the exact value by 5.83 +- 0.04 J/(K mol of atoms), spread 0.31,
with no trend in mass, amplitude or box size. It contains the kinetic entropy and the fixed cost of
the code. Below about 0.8 wavelengths the compressor cannot resolve the positions and the entropy
falls short by more, which is also the temperature range in which classical MD itself fails; use
phonons there.

The compression codes each atom on its own, so in a solid it counts more than once the entropy the
atoms share (5 to 6 J/(K mol) in fcc Cu at 1300 K). That is measured by
:func:`hermes.compression_entropy.crystal_correlation_entropy` and passed as
``correlation_entropies``; the Einstein solids have none, their atoms being independent.
"""

# Third-Party Library Imports
import numpy as np
import scipy.constants

# Local Imports
from hermes.configuration import Configuration

# J/(K mol of atoms) added to the compression entropy to give the total classical entropy
CLASSICAL_ENTROPY_OFFSET = 5.83
EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208
BOLTZMANN_CONSTANT = scipy.constants.Boltzmann / scipy.constants.electron_volt
# J/(K mol) per eV/K of one particle
J_PER_MOL_PER_EV = scipy.constants.Avogadro * scipy.constants.electron_volt


def configuration_from_md(
    name: str,
    multiplicity: int,
    number_of_atoms: int,
    volumes: np.ndarray,
    temperatures: np.ndarray,
    potential_energies: np.ndarray,
    pressures: np.ndarray,
    compression_entropies: np.ndarray,
    electronic_entropies: np.ndarray = None,
    offset: float = CLASSICAL_ENTROPY_OFFSET,
    correlation_entropies: np.ndarray = None,
) -> Configuration:
    """
    A Configuration from MD averages and compression entropies on a temperature and volume grid.

    Args:
        name: Name or label of the configuration.
        multiplicity: Degeneracy factor used in statistical weighting.
        number_of_atoms: Number of atoms in the MD cell.
        volumes: Cell volumes, shape ``(n_volumes,)``, in Å³, increasing.
        temperatures: Temperatures, shape ``(n_temperatures,)``, in K, increasing and above 0 K.
        potential_energies: Mean potential energy of the cell, shape
            ``(n_temperatures, n_volumes)``, in eV.
        pressures: Mean pressure including the kinetic term, same shape, in GPa.
        compression_entropies: Entropy from :func:`hermes.compression_entropy.trajectory_entropy`,
            same shape, in J/(K mol of atoms).
        electronic_entropies: Electronic entropy of the cell, same shape, in eV/K. Defaults to None.
        offset: J/(K mol of atoms) added to the compression entropy; see the module notes.
        correlation_entropies: Entropy the atoms share, from
            :func:`hermes.compression_entropy.crystal_correlation_entropy`, same shape, in
            J/(K mol of atoms), subtracted. Leave it out only for a check.

    Returns:
        Configuration with F, dF/dV, d2F/dV2, S and C_V per cell, in eV, eV/Å³, eV/Å⁶ and eV/K.

    Raises:
        ValueError: If a grid is not increasing or includes 0 K, or an array has the wrong shape.
    """
    volumes = np.asarray(volumes, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    if np.any(np.diff(volumes) <= 0) or np.any(np.diff(temperatures) <= 0):
        raise ValueError("volumes and temperatures must be strictly increasing")
    if temperatures[0] <= 0:
        raise ValueError("classical MD has no 0 K point; start the temperature grid above 0 K")
    shape = (len(temperatures), len(volumes))
    arrays = {"potential_energies": potential_energies, "pressures": pressures,
              "compression_entropies": compression_entropies}
    if electronic_entropies is not None:
        arrays["electronic_entropies"] = electronic_entropies
    if correlation_entropies is not None:
        arrays["correlation_entropies"] = correlation_entropies
    for key, value in arrays.items():
        if np.shape(value) != shape:
            raise ValueError(f"{key} must have shape {shape} (n_temperatures, n_volumes)")

    per_mole = np.asarray(compression_entropies, dtype=float) + offset
    if correlation_entropies is not None:
        per_mole = per_mole - np.asarray(correlation_entropies, dtype=float)
    entropies = per_mole * number_of_atoms / J_PER_MOL_PER_EV
    if electronic_entropies is not None:
        entropies = entropies + np.asarray(electronic_entropies, dtype=float)
    energies = np.asarray(potential_energies, dtype=float) + 1.5 * number_of_atoms * BOLTZMANN_CONSTANT * temperatures[:, None]
    pressure = np.asarray(pressures, dtype=float) / EV_PER_CUBIC_ANGSTROM_TO_GPA
    config = Configuration(
        name=name,
        multiplicity=multiplicity,
        number_of_atoms=number_of_atoms,
        volumes=volumes,
        temperatures=temperatures,
        helmholtz_energies=energies - temperatures[:, None] * entropies,
        helmholtz_energies_dV=-pressure,
        helmholtz_energies_d2V2=-np.gradient(pressure, volumes, axis=1),
        entropies=entropies,
        heat_capacities=np.gradient(energies, temperatures, axis=0),
    )
    return config
