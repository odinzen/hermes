# Validation

How each part of Hermes is checked, against what, and how close it comes. Entropies are in J/(K mol of atoms). Exact
references for MD are thermodynamic integration with calphy [1] on the same interatomic potential, so a comparison
tests Hermes and nothing else. The validation scripts and run outputs are available from the authors on request.

Four kinds of check stand behind this page:

- **Exact cases** in the test suite (`python -m pytest -o addopts="" -q`, 168 tests, four of which need pycalphad and
  ESPEI), each named below.
- **Held-out tests**: the pass rule, the cases and a fingerprint of the code were written down before any of the test
  runs existed, and nothing was changed after the results were read. All of them are listed, failures included.
- **Planted bugs**: 77 deliberate errors (wrong factors, signs, units, dropped terms) were put into the modules one at a
  time, and the test suite catches 75. The other two change no result: leaving the wavevector phases unconjugated only
  swaps each wavevector with its mirror image, and dividing the covariances by n instead of n - 1 rescales both sides
  of the shared-entropy difference alike.
- **References**: `tools/check_references.py` compares every cited work with its Crossref, DataCite or arXiv record,
  full author list in order, title, year, volume and first page.

## Solid entropy

Near melting use the heat-capacity route. At about a third of the melting point the direct Gaussian estimate reads
+0.1 to +0.5 high (the anchors of `solid_fresh3`), and the error grows toward melting.

| Route | Calls | Held-out result |
|---|---|---|
| Heat capacity: Gaussian anchor at a third of the melting point, then the enthalpy of NPT runs up to T | `gaussian_entropy`, `entropy_from_heat_capacity` | 9 metals at 0.81 to 0.86 of the melting point: all within 0.52, mean +0.35 |
| Direct: compression + offset - shared entropy | `trajectory_entropy`, `crystal_correlation_entropy` | 0.1 to 0.5 high at a third of the melting point; up to 1.4 high near it |

### Held-out tests, in order

| Test | What it tested | Verdict |
|---|---|---|
| `solid_exact` | compression less a sum over neighbour pairs, 12 crystals (bcc Fe, W; hcp Mg, Zr; fcc Al, Ni; two temperatures each) | FAIL: 11 within 1.0, hcp Mg 370 K at -1.05 |
| `solid_fresh` | compression in groups of 256 atoms, exact shared entropy; 12 crystals including Ni at 256, 864 and 2048 atoms | FAIL: all within 0.66, but the 256-atom box 0.77 from the larger ones |
| `solid_fresh2` | adds the centre-of-mass entropy and equal sampling of small cells; 12 new crystals | FAIL: box size now independent (Cu 256 to 2048 atoms within 0.13), but three crystals near melting 1.15 to 1.38 high |
| `solid_fresh3` | the heat-capacity route; 9 metals near melting | PASS: all within 0.52 |

What each failure taught, and what changed:

- **The fixed offset depended on the box.** Coded all at once, the compression's fixed cost per atom falls from 8.0
  J/(K mol) at 64 atoms to 4.7 at 2197, because its first step is set by the largest residual. Atoms are now coded in
  groups of 256, and the offset, measured on 60 exactly sampled Einstein solids (masses 12 to 180, amplitudes 1 to 4.5
  thermal wavelengths, 343 to 2197 atoms), is 5.83 +- 0.04 with no trend in mass, amplitude or box size.
- **A sum over neighbour pairs over-counts what atoms share.** On harmonic crystals with exact covariances the pair sum
  exceeds the exact Gaussian total correlation by 17 to 28 % at 864 atoms. Hermes now takes the exact total correlation
  from the crystal's translations, block-diagonal in wavevector; on fcc, bcc and hcp harmonic crystals it is within
  0.03 of exact for 60 to 1000 frames, where the pair sum is up to 0.8 off.
- **MD holds the centre of mass still; Frenkel-Ladd integration does not.** Hermes adds the centre-of-mass entropy
  (R/N) ln[V (N / 2 pi e sigma^2)^(3/2)], which agrees within 0.015 with calphy's correction on 12 crystals. Without it
  the error depended on the box, 0.6 at 256 atoms against 0.1 at 2048.
- **Near melting the motion is not Gaussian.** Each atom's own motion and every neighbour pair are Gaussian to within
  0.05 (Kraskov and Kozachenko-Leonenko estimators against covariance-matched Gaussian surrogates), but the error of a
  Gaussian description grows with temperature in all six metals tested, from -0.6 to +0.3 at a fifth to two-fifths of
  the melting point to +0.1 to +1.0 at twice those temperatures. The missing part is correlation among three or more atoms. The heat-capacity route carries an
  accurate value upward by thermodynamics instead; in `solid_fresh3` the final error equals the anchor's own error
  within 0.10, so the integration is essentially exact and the remaining +0.35 is the anchor reading slightly high.

### Exact cases in the suite

| Check | Agreement | Test |
|---|---|---|
| Gaussian entropy of an Einstein solid against 3R(1 + ln(kT / hbar omega)) | 0.005 | `tests/test_crystal_correlation.py::test_gaussian_entropy_of_an_einstein_solid` |
| Shared entropy against the exact total correlation of harmonic fcc, bcc and hcp crystals | 0.05 | `tests/test_crystal_correlation.py::test_recovers_the_exact_total_correlation` |
| Small-sample (Wishart) corrections unbiased, real and complex | within noise | `tests/test_crystal_correlation.py::test_log_det_corrections_are_unbiased_for_tiny_samples` |
| Centre-of-mass entropy of independent atoms against its formula | 0.005 | `tests/test_crystal_correlation.py::test_centre_of_mass_entropy_of_independent_atoms` |
| Compression offset the same at 343 and 1728 atoms | 0.6 | `tests/test_compression_entropy.py::test_offset_does_not_depend_on_the_box` |
| Doubling the amplitude adds 3 R ln 2 | 6 % | `tests/test_compression_entropy.py::test_doubling_the_amplitude_adds_three_bits_per_atom` |
| Heat-capacity integration against a classical harmonic solid and direct numerical integration | 1e-8 | `tests/test_heat_capacity.py::test_anharmonic_enthalpy_against_direct_integration` |

## Liquid entropy

| | Hermes | Exact |
|---|---|---|
| liquid Cu 1300 K (Mendelev EAM [2], 864 atoms), liquid route | 81.0 | 82.5 |
| liquid Cu, Zr 2300 K, pure ends | +2.0 to +2.2 high | |
| liquid Cu-Zr 2300 K, excess mixing entropy, liquid route | -0.2 to +0.5 | -1.0 to -1.7 |

The liquid route takes two-body correlations only, so it misses the chemical short-range order of an alloy. The
compression route is not for liquids: it overstates the entropy of fusion of Cu 2.2 times.

## Excess mixing entropy from composition fluctuations

Two held-out tests on liquid Cu-Zr at 2300 K (Mendelev EAM [2], 6912 atoms), both failed:

- **Four compositions:** with eight data points the modified AICc kept no entropy term and reported zero excess entropy.
- **Ten compositions:** the fitted excess entropy is within 0.28 of exact everywhere (rule 0.5), but S_cc(0) misses the
  curvature of the exact excess Gibbs energy at x_Zr = 0.20 (0.162 against 0.182 to 0.186). Hermes underestimates the
  size of the excess entropy at every composition, by 7 to 25 %.

| x_Zr | 0.10 | 0.20 | 0.28 | 0.36 | 0.43 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hermes | -0.53 | -1.01 | -1.29 | -1.46 | -1.51 | -1.46 | -1.24 | -0.88 | -0.47 | -0.13 |
| Exact | -0.71 | -1.27 | -1.44 | -1.72 | -1.79 | -1.71 | -1.52 | -1.01 | -0.51 | -0.20 |

## Quantum correction

fcc Cu at 298.15 K, quantum harmonic entropy from the MD velocity spectrum: 32.98 against 32.94 (CODATA, lattice
part). One system only.

## CALPHAD output

| Output | Check | Test |
|---|---|---|
| Redlich-Kister liquid parameters as TDB lines | pycalphad reads back G, H and S of mixing to 1e-6 | `tests/test_calphad_interop.py::test_pycalphad_reads_the_tdb_parameters_as_meant` |
| Mixing data as ESPEI datasets | an ESPEI fit gives ideal plus excess entropy to 1e-3 | `tests/test_calphad_interop.py::test_espei_fit_of_excess_entropy_gives_ideal_plus_excess` |
| Formation enthalpy, entropy and heat capacity as ESPEI datasets | an ESPEI fit recovers the formation Gibbs energy exactly, and pycalphad reads the compound's entropy back as the SGTE elements plus it, to 1e-3 | `tests/test_formation.py::test_espei_fit_reproduces_the_formation_properties` |

Fit formation data with `generate_parameters(..., ridge_alpha=1e-100)`. With ESPEI's default (None) scikit-learn's
LinearRegression (version 1.9) drops the badly scaled 1/T term of the heat capacity without warning; this was found
while building the export and has been reported to ESPEI.

## Limits

- **Solids near melting:** use the heat-capacity route; the direct route reads up to 1.4 high there.
- **Random alloys:** the shared entropy needs the crystal's translations, which random species break; they are refused.
- **Excess mixing entropy:** underestimated by 7 to 25 % on Cu-Zr; untested on other systems.
- **Classical MD below about 0.8 thermal wavelengths of vibration:** the compression cannot resolve the positions; use
  the Gaussian route or phonons, and `hermes.quantum_correction` for the quantum part.

## Potentials used

Ni, Pd, Ag, Pt, Au, Cu: Foiles, Baskes and Daw [3] (`*_u3.eam`); Cu: Mishin et al. [4] (`Cu_mishin1.eam.alloy`); W: Zhou
et al. [5] (`W_zhou.eam.alloy`); Fe: Mendelev et al. [6] (`Fe_mm.eam.fs`); Mg: Sun et al. [7] (`Mg_mm.eam.fs`); Zr:
Mendelev and Ackland [8] (`Zr_mm.eam.fs`); Cu-Zr: Mendelev et al. [2] (`CuZr_mm.eam.fs`); Al and V: the LAMMPS files
`Al_mm.eam.fs` and `VFe_mm.eam.fs` (M. I. Mendelev, 2007), whose headers give no publication.

## References

1. S. Menon, Y. Lysogorskiy, J. Rogal and R. Drautz, "Automated free-energy calculation from atomistic simulations",
   Phys. Rev. Materials 5, 103801 (2021). https://doi.org/10.1103/PhysRevMaterials.5.103801
2. M. I. Mendelev, M. J. Kramer, R. T. Ott, D. J. Sordelet, D. Yagodin and P. Popel, "Development of suitable
   interatomic potentials for simulation of liquid and amorphous Cu-Zr alloys", Philos. Mag. 89, 967 (2009).
   https://doi.org/10.1080/14786430902832773
3. S. M. Foiles, M. I. Baskes and M. S. Daw, "Embedded-atom-method functions for the fcc metals Cu, Ag, Au, Ni, Pd, Pt,
   and their alloys", Phys. Rev. B 33, 7983 (1986). https://doi.org/10.1103/PhysRevB.33.7983
4. Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter and J. D. Kress, "Structural stability and lattice
   defects in copper: Ab initio, tight-binding, and embedded-atom calculations", Phys. Rev. B 63, 224106 (2001).
   https://doi.org/10.1103/PhysRevB.63.224106
5. X. W. Zhou, H. N. G. Wadley, R. A. Johnson, D. J. Larson, N. Tabat, A. Cerezo, A. K. Petford-Long, G. D. W. Smith,
   P. H. Clifton, R. L. Martens and T. F. Kelly, "Atomic scale structure of sputtered metal multilayers", Acta Mater.
   49, 4005 (2001). https://doi.org/10.1016/S1359-6454(01)00287-7
6. M. I. Mendelev, S. Han, D. J. Srolovitz, G. J. Ackland, D. Y. Sun and M. Asta, "Development of new interatomic
   potentials appropriate for crystalline and liquid iron", Philos. Mag. 83, 3977 (2003).
   https://doi.org/10.1080/14786430310001613264
7. D. Y. Sun, M. I. Mendelev, C. A. Becker, K. Kudin, T. Haxhimali, M. Asta, J. J. Hoyt, A. Karma and D. J. Srolovitz,
   "Crystal-melt interfacial free energies in hcp metals: A molecular dynamics study of Mg", Phys. Rev. B 73, 024116
   (2006). https://doi.org/10.1103/PhysRevB.73.024116
8. M. I. Mendelev and G. J. Ackland, "Development of an interatomic potential for the simulation of phase
   transformations in zirconium", Philos. Mag. Lett. 87, 349 (2007). https://doi.org/10.1080/09500830701191393
