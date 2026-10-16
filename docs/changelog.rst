Changelog
=========

Hermes 0.1.0 (2026-10-08)
-------------------------

- Forked from pyzentropy 0.1.1 (Phases Research Lab, MIT) and renamed Hermes.
- Added ``hermes.compression_entropy``: entropy from the compressibility of MD snapshots, after Fisher and
  Hong (arXiv:2512.02221), calibrated on an exactly sampled Einstein solid.
- Added ``hermes.md_configuration.configuration_from_md``: Configurations from MD energies, pressures and
  compression entropies.

Unreleased
----------

- Added ``hermes.liquid_entropy``: liquid entropy from partial pair distribution functions, and a
  mechanical / ideal / excess split of solution entropies.
- Added ``hermes.quantum_correction``: quantum correction from the MD vibrational spectrum.
- Cu and Cu-Zr checks against thermodynamic integration in ``validation/``.

The history of pyzentropy itself is at https://github.com/PhasesResearchLab/pyzentropy/releases/.
