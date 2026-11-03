# Validation

How each part of Hermes is checked, against what, and how close it comes. Entropies are in J/(K mol of atoms). Exact
references for MD are thermodynamic integration with calphy [1] on the same interatomic potential, so a comparison
tests Hermes and nothing else. The rule, scripts, fingerprints and output of every test are in `validation/` (see
its README); the MD trajectories are not kept in the repository.

Four kinds of check stand behind this page:

- **Exact cases** in the test suite (`python -m pytest -o addopts="" -q`, 242 tests, six of which need pycalphad and
  four of those ESPEI too), each named below.
- **Held-out tests**: the pass rule, the cases and a fingerprint of the code were written down before any of the test
  runs existed, and nothing was changed after the results were read. All of them are listed, failures included.
- **Planted bugs**: 126 deliberate errors (wrong factors, signs, units, dropped terms) were put into the modules one at a
  time, and the test suite catches 123. The other three change no result: leaving the wavevector phases unconjugated only
  swaps each wavevector with its mirror image, dividing the covariances by n instead of n - 1 rescales both sides
  of the shared-entropy difference alike, and taking each wavevector's length without its periodic images only
  leaves out mirror copies of the shortest ones in every box tested (cubic, hexagonal and triclinic); it guards
  boxes whose shortest wavevector mixes signs.
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
- **A box is not a bulk crystal.** Fed exact Gaussian samples of harmonic MgO (MACE-MP-0 force constants, 298 K),
  the box value lies 1.75, 0.55 and 0.24 above the bulk entropy (a dense phonon mesh) at 64, 216 and 512 atoms: the
  centre-of-mass entropy belongs to the box, and the box lacks the vibrations longer than itself. The bulk value
  (`limit="bulk"`) is within 0.04 of exact at all three sizes and within 0.08 on six more oxides of 216 to 270 atoms
  (corundum, ilmenite, La2O3, rock salt). The exact references of the tests above, thermodynamic integration on the
  same box, are box values, so those tests compare like with like; a comparison with calorimetry needs the bulk value.
  This came out of the closed-shell oxide test (below) and was settled against exact references, not measurements.
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
| Bulk limit against the infinite harmonic fcc crystal (Brillouin-zone mean of ln det C) | 0.15 at 108 atoms, 0.05 at 256 | `tests/test_crystal_correlation.py::test_bulk_limit_recovers_the_infinite_crystal` |
| The box value exceeds the bulk value by its centre-of-mass entropy | sign and size | `tests/test_crystal_correlation.py::test_box_and_bulk_differ_by_the_finite_box` |
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

## Crystal-field, magnetic and electronic terms

These terms come from measured or DFT inputs, not from MD, and are added to the vibrational entropy. They pass their
exact cases; their test against measured entropies failed (below), largely because the vibrational entropy they are
added to was itself too high. Use them with that in mind.

| Check | Agreement | Test |
|---|---|---|
| Two-level Schottky anomaly: heat capacity and entropy against the closed form, peak 0.4392 R at 0.4168 delta | 1e-10 | `tests/test_crystal_field.py::test_two_level_schottky_anomaly` |
| Crystal field of Nd3+ (five doublets) reaches R ln(2J+1) = R ln 10 at high temperature | 1e-5 | `tests/test_crystal_field.py::test_high_temperature_limit_is_r_ln_2j_plus_1` |
| Ground-level degeneracy R ln g_0 counted or left out, as chosen | 1e-12 | `tests/test_crystal_field.py::test_low_temperature_limit_is_the_ground_degeneracy` |
| cm^-1 and meV converted with h, c, k_B and e written out | 1e-9 | `tests/test_crystal_field.py::test_entropy_from_first_principles_constants` |
| Hund's rules give J of all fifteen trivalent lanthanides | exact | `tests/test_crystal_field.py::test_hund_rules_give_the_lanthanide_ground_multiplets` |
| Hillert-Jarl Gibbs energy against the published polynomials, bcc and other p | 1e-9 | `tests/test_magnetic.py::test_gibbs_energy_against_the_published_form` |
| Magnetic heat capacity integrates to R ln(beta + 1), with the fraction p of the ordering enthalpy above T* | 1e-9 | `tests/test_magnetic.py::test_ordering_releases_r_ln_beta_plus_1_and_p_of_the_enthalpy_lies_above_t_star` |
| Spin-only ion (Mn2+) reaches R ln(2S+1) = R ln 6 | 1e-9 | `tests/test_magnetic.py::test_spin_only_paramagnetic_limit` |
| The three references: no ordering enthalpy on an ordered cell, all of it on a disordered one | exact | `tests/test_magnetic.py::test_the_three_references` |
| pycalphad's magnetic model, fcc and bcc Fe (its R = 8.3145 allowed for) | 1e-9 | `tests/test_magnetic.py::test_pycalphad_magnetic_model_agrees` |
| Free-electron metal against the Sommerfeld gamma T | 5e-4 (the (k_B T / E_F)^2 correction) | `tests/test_electronic.py::test_free_electron_metal_gives_gamma_t` |
| gamma = 2.357 mJ/(K^2 mol) per state/eV/atom from k_B, N_A and e written out | 1e-6 | `tests/test_electronic.py::test_sommerfeld_coefficient_from_first_principles_constants` |
| Wide-gap insulator (5 eV) at 1500 K: no electronic entropy | below 1e-5 | `tests/test_electronic.py::test_wide_gap_insulator_has_no_electronic_entropy` |
| Two levels across a gap, against the closed form | 1e-4 | `tests/test_electronic.py::test_two_levels_across_a_gap` |
| A gapless density of states opened to a measured 3 eV gap: no electronic entropy at 1000 K | below 1e-4 | `tests/test_electronic_gap.py::test_a_gapless_density_of_states_opened_to_a_wide_gap_has_no_electronic_entropy` |
| A 0.4 eV gap opened to 1.2 eV equals a density of states built with 1.2 eV | 1 % | `tests/test_electronic_gap.py::test_opening_a_gap_matches_a_density_of_states_built_with_that_gap` |
| A gap already larger than the measured one is left alone | 1e-12 | `tests/test_electronic_gap.py::test_a_larger_gap_is_left_alone` |

The pycalphad comparison found that pycalphad 0.11.1 returns no magnetic energy at exactly T = T_C, because its
piecewise conditions are strict on both sides; one degree either side it agrees.

**Test against measurement.** The rule, cases, pass criteria and code fingerprint were fixed on 2026-10-22, before any
data were scored (`validation/extra_terms`). It passes only if both arms pass; both failed.

*Lanthanide orthophosphates LnPO4: failed.* The crystal-field term against the excess entropy left when the lattice
part (LaPO4 for the monazites, LuPO4 for the xenotimes) is removed from the measured entropy, in J/(K mol of Ln), pass
within 1.0 at 298.15 K and at 1000 K. The levels (one neutron or optical source per ion) were fixed before any
phosphate heat capacity was read; the measured entropies come from 14 calorimetric papers. None of the seven scored
compounds passes:

| Compound | Error at 298.15 K | Error at 1000 K |
|---|---|---|
| NdPO4 | +2.72 | no data |
| EuPO4 | +0.68 | -2.96 |
| GdPO4 | +1.83 | +3.29 |
| HoPO4 | -4.98 | no data |
| ErPO4 | -0.56 | -2.70 |
| TmPO4 | -8.13 | -12.04 |
| YbPO4 | +0.03 | -3.12 |

GdPO4 shows where most of this comes from. Its crystal-field entropy is R ln 8 = 17.29 with no levels involved, yet the
measured GdPO4 less LaPO4 leaves 15.46 at 298.15 K and 14.00 at 1000 K: subtracting a neighbour's lattice entropy is
itself uncertain by 2 to 3, more than the tolerance, and YPO4 in place of LuPO4 makes the xenotimes worse. Most of the
TmPO4 error is a magnetic term R ln 2 the calorimetry paper assumed rather than measured, in a crystal whose lowest
level is a singlet; without it the errors are -2.4 and -6.3 (reported, outside the verdict). TbPO4 was not scored (its
only calorimetry is of the monoclinic form, its levels of the tetragonal one), nor DyPO4 (no low-temperature data), nor
Ce, Pr and Sm (no measured levels in a phosphate). A better test needs a lattice reference good to a fraction of a
J/(K mol), such as the vibrational entropy of both crystals from the same route.

*Magnetic oxides NiO, CoO, MnO, Fe2O3 and Cr2O3: failed.* Vibrational (MACE-MP-0, the route of the closed-shell oxide
test below) plus magnetic plus electronic entropy against the measured S(298.15 K) and S(1000 K), in J/(K mol of atoms).
Every prediction is high, by +4.9 to +9.9 at 298.15 K and +6.0 to +9.4 at 1000 K, and CoO, predicted in advance to
fall short because Co2+ carries an orbital moment, came out high as well. The vibrational part alone is already 1.9 to
4.7 high on the closed-shell oxides, so this arm says little about the magnetic term itself. Recorded before scoring:
the Materials Project densities of states used for CoO, Fe2O3 and Cr2O3 are nearly gapless, which gives them a spurious
electronic entropy of up to 4 at 1000 K; `band_gap` (0.3) now opens them to the measured gap.

| Compound | Error at 298.15 K | Error at 1000 K |
|---|---|---|
| NiO | +7.79 | no data |
| MnO | +6.72 | +6.49 |
| CoO | +9.94 | +9.43 |
| Fe2O3 | +4.86 | +5.95 |
| Cr2O3 | +6.08 | +8.12 |

The full prediction for the seven phosphates (vibrational from MD plus crystal field plus electronic) was reported
outside the verdict: within 1.6 for Ho, Er and Tm, +2.6 for Nd and Gd, +5.8 for Eu and +10.5 for Yb, whose crystal lost
its symmetry at 1000 K.

## Vibrational entropy of oxides against calorimetry

Three held-out tests put the vibrational route, run on universal machine-learned potentials (MACE foundation models,
MIT licence), against measured entropies of closed-shell oxides, in J/(K mol of atoms). All three failed. Each
prediction is the Gaussian anchor at 298.15 K plus the quantum correction, carried to 1000 K by the enthalpy of an NPT
ladder; the measured values come from JANAF [9], Robie and Hemingway [10] and Konings et al. [11].

| Test | Potential, cases | Verdict |
|---|---|---|
| `oxide_vibration` (frozen 2026-10-23) | MACE-MP-0, box value (Hermes 0.2); MgO, Al2O3, TiO2, Y2O3, Lu2O3, La2O3, MgTiO3 | FAIL: all high, +1.9 to +4.7 at 298.15 K (mean +3.3) |
| `oxide_heldout` (frozen 2026-10-28) | MACE MPA-0, bulk value; 11 oxides none of the earlier work had seen | FAIL on absolute entropy, +1.05 to +2.94 at 298.15 K (mean +1.84) and +0.37 to +3.20 at 1000 K, all high; polymorph differences of Al2SiO5 HOLD, within 0.41 of JANAF |
| `polymorph_heldout` (frozen 2026-11-02) | MACE MPA-0, bulk value less an offset fixed from the 19 earlier oxides (+1.87 at 298.15 K, +2.20 at 1000 K); 13 new compounds in four polymorph families | VOID (FAIL): four crystals changed structure in the MD, which left one family to compare |

What each taught:

- **The box is not the bulk crystal.** The first test compared box values with calorimetry. That is wrong by 0.4 to 0.6
  at these sizes, and Hermes 0.3 makes the choice explicit (`limit="box"` or `"bulk"`, above).
- **The excess belongs to the potential.** Between the first two tests, eight development oxides were run with three
  MIT potentials (`validation/sol_bench`, outside every verdict). Mean bulk error at 298.15 K: MACE-MP-0 +2.83 (7
  oxides), MP-0b3 +2.22, MPA-0 +1.90 (8 each), so MPA-0 was taken for the held-out tests. All three over-expand the
  crystals, by 4.7 to 5.9 % in volume on average, and the volume error does not rank their entropy errors.
- **A fixed offset works at 1000 K, not at 298.15 K.** After subtracting the offset, the third test's residuals have an
  RMS of 0.41 over the five compounds scored at 1000 K (rule 1.0) but 1.04 over the nine at 298.15 K, where aragonite
  reads 1.90 high. In the classical limit a softened spectrum shifts a compound's entropy by an amount that no longer
  changes with temperature; at room temperature the quantum part makes the error depend on the spectrum.
- **Polymorph differences do not carry over in general.** The Al2SiO5 differences held, but aragonite less calcite
  came out +1.85 against a measured -0.74, the wrong sign. The SiO2, CaSiO3 and MgSiO3 families could not be judged:
  MPA-0 turned alpha-quartz into the beta structure at 298.15 K (9 % too large in volume) and changed the symmetry of
  pseudowollastonite, clinoenstatite and CaTiO3 perovskite at 1000 K.

**Where the error is.** The harmonic entropy of each potential's own vibrational spectrum, read from the same MD, is
already above the measured value in 23 of the 24 compounds of the two MPA-0 tests. Finite-displacement phonons of
MPA-0, which use no Hermes code, give the same high values (calcite 18.85 against 18.77 from the MD spectrum; CaO,
at the potential's relaxed volume, 21.34 against 21.12; `validation/polymorph_heldout/post_scoring`). Holding the crystal at its measured volume brings
calcite within 0.2 of the measurement but leaves CaO 1.75 and aragonite 1.5 high. The potentials' force constants are
too soft, and Hermes reports them as they are. Hermes's Gaussian route adds 0.1 to 1.4 above that harmonic value, the
anharmonic part; on exact Gaussian samples the route is within 0.08 of the exact bulk entropy, but this anharmonic
part has not been checked against an exact reference on these oxides.

## Limits

- **Solids near melting:** use the heat-capacity route; the direct route reads up to 1.4 high there.
- **Random alloys:** the shared entropy needs the crystal's translations, which random species break; they are refused.
- **Excess mixing entropy:** underestimated by 7 to 25 % on Cu-Zr; untested on other systems.
- **Crystal-field term:** failed its test against measured phosphate entropies, where subtracting a neighbour's
  lattice entropy is itself uncertain by 2 to 3 J/(K mol). Magnetic and electronic terms: exact cases pass; their test
  against measurement failed with a vibrational part that was itself too high. The levels and the density of states
  are held fixed in temperature and volume; the magnetic model knows spins only and takes the 0 K cell's magnetic
  order to be the ground state.
- **Universal machine-learned potentials:** the MACE foundation models tested give oxide vibrational entropies 1 to 3
  J/(K mol of atoms) too high at 298.15 K, from force constants that are too soft, and can move a crystal into another
  structure (alpha- to beta-quartz at 298.15 K). Hermes measures the entropy of the potential it is given; for
  comparison with calorimetry, use a potential whose phonons have been checked against measurement.
- **Classical MD below about 0.8 thermal wavelengths of vibration:** the compression cannot resolve the positions; use
  the Gaussian route or phonons, and `hermes.quantum_correction` for the quantum part.

## Potentials used

Ni, Pd, Ag, Pt, Au, Cu: Foiles, Baskes and Daw [3] (`*_u3.eam`); Cu: Mishin et al. [4] (`Cu_mishin1.eam.alloy`); W: Zhou
et al. [5] (`W_zhou.eam.alloy`); Fe: Mendelev et al. [6] (`Fe_mm.eam.fs`); Mg: Sun et al. [7] (`Mg_mm.eam.fs`); Zr:
Mendelev and Ackland [8] (`Zr_mm.eam.fs`); Cu-Zr: Mendelev et al. [2] (`CuZr_mm.eam.fs`); Al and V: the LAMMPS files
`Al_mm.eam.fs` and `VFe_mm.eam.fs` (M. I. Mendelev, 2007), whose headers give no publication. Oxides: the MACE
foundation models MACE-MP-0 medium, MACE-MP-0b3 medium and MACE MPA-0 medium, MIT licence
(https://github.com/ACEsuit/mace-foundations), with the SHA-256 of each weights file recorded in the tests'
`PREREGISTERED.txt`.

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
9. M. W. Chase, Jr., NIST-JANAF Thermochemical Tables, fourth edition, J. Phys. Chem. Ref. Data Monograph 9 (1998).
10. R. A. Robie and B. S. Hemingway, "Thermodynamic properties of minerals and related substances at 298.15 K and 1 bar
   (10^5 pascals) pressure and at higher temperatures", U.S. Geological Survey Bulletin 2131 (1995).
   https://doi.org/10.3133/b2131
11. R. J. M. Konings, O. Benes, A. Kovacs, D. Manara, D. Sedmidubsky, L. Gorokhov, V. S. Iorish, V. Yungman,
   E. Shenyavskaya and E. Osina, "The Thermodynamic Properties of the f-Elements and their Compounds. Part 2. The
   Lanthanide and Actinide Oxides", J. Phys. Chem. Ref. Data 43, 013101 (2014). https://doi.org/10.1063/1.4825256
