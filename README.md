# Hermes

An all-in-one entropy tool: Hermes measures the entropy of a phase from simulation, term by term, and
turns it into CALPHAD parameters.

MD trajectories from any potential (machine-learned or classical) and DFT inputs go in; out come the
entropy of each contribution, the total, and Redlich-Kister excess parameters with uncertainties, written
as TDB lines for pycalphad or as datasets for ESPEI.

Hermes started from [pyzentropy](https://github.com/PhasesResearchLab/pyzentropy) 0.1.1 by Nigel L. E.
Hew (Phases Research Lab, Penn State; MIT license), whose zentropy core it keeps:
S = -k_B sum p_k ln p_k + sum p_k S_k combines the configurations of a phase, each with its own Helmholtz
energy F_k(T, V), into the phase's free energy, entropy and heat capacity. The name follows Fisher and
Hong, who treat entropy as a message: the number of bits needed to tell a receiver where every atom is.

## Entropy terms

| Term | Measured from | Module | Held-out accuracy (see docs/validation.md) |
|---|---|---|---|
| Configurational: orderings, magnetic states, defects that can be listed | -k_B sum p_k ln p_k over configurations | `hermes.system`, `hermes.configuration` (zentropy core) | as pyzentropy |
| Vibrational, classical, anharmonic included (crystals) | MD positions: compression [1] or Gaussian, less the shared entropy; near melting, carried up by the heat capacity | `hermes.compression_entropy`, `hermes.heat_capacity`, `hermes.md_configuration` | 9 metals near melting within 0.52 J/(K mol) |
| Vibrational, quantum part | MD velocity spectrum | `hermes.quantum_correction` | Cu at 298 K within 0.04 J/(K mol) of CODATA |
| Liquid | partial pair distribution functions [3, 4] | `hermes.liquid_entropy` | within 2 % for pure liquids |
| Ideal mixing | composition | `hermes.liquid_entropy` | exact |
| Excess mixing (chemical short-range order) | long-wavelength composition fluctuations [5] | `hermes.concentration_fluctuations`, `hermes.excess_model` | failed its held-out test; 7 to 25 % too small on Cu-Zr |
| Electronic | DFT density of states, Fermi-Dirac occupation | `hermes.electronic` | exact cases only; no test against measurement yet |
| Crystal field, open-4f lanthanide ions | measured crystal-field levels | `hermes.crystal_field` | exact cases only; test against measurement pre-registered, not run |
| Magnetic, transition-metal ions | ordering temperature and moment, Inden / Hillert-Jarl [8, 9] | `hermes.magnetic` | exact cases only; test against measurement pre-registered, not run |

**Crystals.** Coding each atom on its own counts more than once the entropy the atoms share, so Hermes subtracts the
total correlation of the displacements, taken exactly from the crystal's translations
(`crystal_correlation_entropy`), and adds the entropy of the crystal's centre of mass, which MD holds still. The
compression is a clean-room implementation of Fisher and Hong [1], written from the preprint alone, independent of the
authors, whose code was not released; it codes atoms in groups of 256 so that its calibrated offset, 5.83 J/(K mol)
from 60 exactly sampled Einstein solids, does not depend on the box. `gaussian_entropy` takes each atom's motion as
Gaussian instead and needs no minimum amplitude. Both read high near melting, where correlation among three or more
atoms is no longer Gaussian, so there `entropy_from_heat_capacity` anchors the entropy at about a third of the melting
point and carries it up with the enthalpy of NPT runs: S(T) = S(T0) + int C_P / T dT.

**Mixtures.** `hermes.liquid_entropy.mixing_decomposition` splits a solution's entropy into mechanical mixture, ideal
mixing and excess, as a CALPHAD model does. `hermes.excess_model` fits L_k = a_k + b_k T to MD mixing enthalpies and
composition fluctuations together; candidate models and their selection by modified AICc follow ESPEI [6], and the
fit being linear, its uncertainties are exact without sampling. Supply about ten compositions or more.

**Terms atomic motion does not carry.** MD sees the atoms move, not the electrons or spins, so three terms are
added from other inputs, never fitted to the data they are meant to predict. `hermes.crystal_field` gives the
Schottky entropy, heat capacity and Gibbs energy of the crystal-field levels of a lanthanide ion, from level energies
measured by spectroscopy or neutron scattering; with the whole ground multiplet it reaches R ln(2J+1) at high
temperature. `hermes.magnetic` gives the magnetic terms of the Inden / Hillert-Jarl model from T_C or T_N, the mean
moment and the structure factor, rising to R ln(2S+1) for a spin-only ion. An MLIP or DFT energy of an ordered cell
already holds the ordering energy, so the module asks which energy its term is added to (`reference="ordered"`,
`"disordered"` or `"calphad"`) and has no default. `hermes.electronic` gives the electronic entropy and free energy of
a density of states, such as the Materials Project's, with the chemical potential solved at each temperature; it is
gamma T for a metal and vanishes for a wide-gap insulator. Their limits: crystal-field levels and the density of states
are held fixed in temperature and volume; the magnetic model knows spins only, so orbital moments (Co2+, high-spin
Fe2+) carry entropy it misses, and it takes the 0 K cell's magnetic order to be the ground state; a band gap that DFT
closes gives a spurious electronic entropy. None has yet been tested against measured entropies.

## CALPHAD output

- Redlich-Kister parameters as TDB lines for pycalphad (`ExcessModel.tdb_parameters`) and mixing data as ESPEI
  datasets (`espei_datasets`).
- Formation enthalpy, entropy and heat capacity of a compound, or the lattice stability of an element, as ESPEI
  datasets (`hermes.formation.espei_formation_datasets`). They are taken against the elements computed with the same
  potential and route, never against laboratory references. Fit them with
  `generate_parameters(..., ridge_alpha=1e-100)`: with ESPEI's default, scikit-learn's LinearRegression (version 1.9)
  drops the 1/T term of the heat capacity without warning.

An ESPEI fit of each and a pycalphad read-back are in the test suite.

## Validation

Every route is checked against exact references, thermodynamic integration with calphy [7] on the same interatomic
potential, in tests whose pass rule was written before the runs existed. All of them are reported in
[docs/validation.md](docs/validation.md), including the five that failed and what each failure changed. In brief:

- **Crystals near melting** (9 metals, heat-capacity route): within 0.52 J/(K mol), mean +0.35. The direct route reads
  up to 1.4 high there.
- **Box size:** Cu at 256, 864 and 2048 atoms agrees to 0.13.
- **Excess mixing entropy** (liquid Cu-Zr): within 0.28 of exact at ten compositions but 7 to 25 % too small, and the
  test failed on its second criterion.

The test suite has 231 tests (six need pycalphad, four of them ESPEI too). Of 114 errors planted in the modules one at
a time, it catches 112; the other two change no result. `tools/check_references.py` checks every reference against its
registry record.

## Notebook

[notebooks/EntropyWithHermes](notebooks/EntropyWithHermes/EntropyWithHermes.ipynb) takes fcc Cu on an EAM potential
at 480 K and 1150 K, reads its entropy from the MD trajectories, carries the 480 K value up with the heat capacity, and
compares each with thermodynamic integration on the same potential. The data are included; running it needs matplotlib
as well. `tools/build_notebook.py` rebuilds it. `examples/codespace/Fe3Pt.ipynb` shows the inherited zentropy core.

## Install

Hermes is not on PyPI (the `hermes` name there belongs to an unrelated project). Clone this repository
and install from source:

    python -m pip install -e .

Tests: `python -m pytest -o addopts="" -q`. The pycalphad and ESPEI round-trip tests run where those
packages are installed and are skipped otherwise.

## License

MIT, as pyzentropy. Hew's copyright notice is kept in LICENSE.txt; the Hermes additions are copyright
Odinzen LLC.

## Citation

Hermes on Zenodo:

- version 0.2.0: https://doi.org/10.5281/zenodo.23573417
- all versions: https://doi.org/10.5281/zenodo.23423225

See `CITATION.cff`. If you use `hermes.compression_entropy`, please cite the method's authors [1]; for the zentropy
machinery, cite pyzentropy [2]; for the liquid and mixing routes, [3-6]; for the magnetic term, [8, 9].

## References

1. D. Fisher and Q.-J. Hong, "Thermodynamic Entropy as Information -- A compression-based demonstration
   of the Shannon-Boltzmann equivalence in condensed matter", arXiv:2512.02221 (2025).
   https://arxiv.org/abs/2512.02221
2. N. L. E. Hew, pyzentropy, version 0.1.1 (GitHub tag 0.1.1; Hermes started four tutorial commits later,
   at bd92123). All versions on Zenodo, where each is labelled 0.1.0: https://doi.org/10.5281/zenodo.18262305
3. R. E. Nettleton and M. S. Green, "Expression in Terms of Molecular Distribution Functions for the
   Entropy Density in an Infinite System", J. Chem. Phys. 29, 1365 (1958). https://doi.org/10.1063/1.1744724
4. A. Baranyai and D. J. Evans, "Direct entropy calculation from computer simulation of liquids",
   Phys. Rev. A 40, 3817 (1989). https://doi.org/10.1103/PhysRevA.40.3817
5. A. B. Bhatia and D. E. Thornton, "Structural Aspects of the Electrical Resistivity of Binary Alloys",
   Phys. Rev. B 2, 3004 (1970). https://doi.org/10.1103/PhysRevB.2.3004
6. B. Bocklund, R. Otis, A. Egorov, A. Obaied, I. Roslyakova and Z.-K. Liu, "ESPEI for efficient
   thermodynamic database development, modification, and uncertainty quantification: application to
   Cu-Mg", MRS Communications 9, 618 (2019). https://doi.org/10.1557/mrc.2019.59
7. S. Menon, Y. Lysogorskiy, J. Rogal and R. Drautz, "Automated free-energy calculation from atomistic simulations",
   Phys. Rev. Materials 5, 103801 (2021). https://doi.org/10.1103/PhysRevMaterials.5.103801
8. M. Hillert and M. Jarl, "A model for alloying in ferromagnetic metals", Calphad 2, 227 (1978).
   https://doi.org/10.1016/0364-5916(78)90011-1
9. G. Inden, "The role of magnetism in the calculation of phase diagrams", Physica B+C 103, 82 (1981).
   https://doi.org/10.1016/0378-4363(81)91004-4
