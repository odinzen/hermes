Changelog
=========

Hermes 0.2.0 (2026-10-22)
-------------------------

- Added ``hermes.crystal_field``: Schottky entropy, heat capacity and Gibbs energy of the crystal-field levels of
  open-4f lanthanide ions, from measured level energies.
- Added ``hermes.magnetic``: magnetic entropy and Gibbs energy by the Inden / Hillert-Jarl model, with an explicit
  choice of the energy it is added to, so that an ordered 0 K cell's ordering energy is not counted twice.
- Added ``hermes.electronic``: electronic entropy and free energy of a density of states, the chemical potential
  solved at each temperature.
- All three are checked against exact results only; a test against measured entropies is pre-registered, not yet run.

Hermes 0.1.0 (2026-10-16)
-------------------------

- Forked from pyzentropy 0.1.1 (Phases Research Lab, MIT) and renamed Hermes.
- Added ``hermes.compression_entropy``: entropy from the compressibility of MD snapshots, after Fisher and
  Hong (arXiv:2512.02221), calibrated on an exactly sampled Einstein solid.
- Added ``hermes.md_configuration.configuration_from_md``: Configurations from MD energies, pressures and
  compression entropies.
- Added ``hermes.liquid_entropy``: liquid entropy from partial pair distribution functions, and a
  mechanical / ideal / excess split of solution entropies.
- Added ``hermes.quantum_correction``: quantum correction from the MD vibrational spectrum.
- Added ``hermes.heat_capacity``, ``hermes.concentration_fluctuations``, ``hermes.excess_model`` and
  ``hermes.formation``.
- Cu and Cu-Zr checks against thermodynamic integration in ``validation/``.

The history of pyzentropy itself is at https://github.com/PhasesResearchLab/pyzentropy/releases/.
