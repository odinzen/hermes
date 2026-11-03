# Crystal-field, magnetic and electronic terms against measured entropies

Status: FROZEN 2026-10-22 on Michael's approval; not run. Proposed the same day. No measured entropy or heat capacity
was read while writing it (the Juno manifests were searched only for which records exist and their sources, never
their values). The only change between proposal and freezing is the one the proposal allowed: the Neel temperatures
were replaced by cited values (MnO 118 -> 119 K, CoO 291 -> 289 K, Cr2O3 307 -> 308.5 K; NiO and Fe2O3 unchanged).

`PREREGISTERED.txt` holds the date and the SHA-256 of this file and of the code below (line endings normalised to LF).
Input files that do not exist yet (`levels.json`, the LnPO4 data) are hashed into it, with a date, when they are
written and before any thermal data are scored. Nothing in this file changes after freezing, whatever the results; a
fix found by this test gets a new test on new data.

## Code under test

Commit 7c46c9e on branch `claude/extra-entropy-terms-7d6972`:

| Module | SHA-256 (LF) |
|---|---|
| `hermes/crystal_field.py` | 0c4694a7425256a9a253ef16f281e74dfbde2667e0fee429b7bd435bb4428f96 |
| `hermes/magnetic.py` | 9fbe13e7227a9568208e52c692432fc0098ef9d771972204567134e34a69ee28 |
| `hermes/electronic.py` | 3b9ebc37f87a4f83025bd48b4e1d24d9c17b40f41ae22573a9dfa242e6883e85 |

The vibrational part is not new code: it is taken by the route, potential and settings frozen for the closed-shell
oxide test (MgO, Al2O3, rutile, Y2O3, Sc2O3, La2O3, Lu2O3, SrTiO3, MgTiO3, Mg3(PO4)2), applied unchanged. That test
is separate and is not affected by this one.

All entropies below are per mole of atoms unless said otherwise, in J/(K mol).

## Arm A: magnetic oxides

### Cases and inputs, fixed now

| Oxide | Ion | S | beta = 2S | T_N (K) | p | ions / atoms per formula |
|---|---|---|---|---|---|---|
| Oxide | Ion | S | beta = 2S | T_N (K) | source of T_N | p | ions / atoms per formula |
|---|---|---|---|---|---|---|---|
| NiO | Ni2+ | 1 | 2 | 523 | [1], neutron diffraction, 523 +- 2 K | 0.28 | 1 / 2 |
| MnO | Mn2+ | 5/2 | 5 | 119 | [1], neutron diffraction, 119 +- 1 K | 0.28 | 1 / 2 |
| CoO | Co2+ | 3/2 | 3 | 289 | [2], 15.8 C on a single crystal | 0.28 | 1 / 2 |
| alpha-Fe2O3 | Fe3+ | 5/2 | 5 | 950 | [3] via [4] | 0.28 | 2 / 5 |
| Cr2O3 | Cr3+ | 3/2 | 3 | 308.5 | [2], 35.38 C on a single crystal | 0.28 | 2 / 5 |

The values were read from the text of [1] (abstract), [2] (full text) and [4] (full text, which places the hematite
transition of [3] at about 950 K; the full text of [3] was not available). The moment is the spin-only 2S, not a
fitted or measured moment, and p is Inden's value for non-bcc structures.

1. A. M. Balagurov, I. A. Bobrikov, S. V. Sumnikov, V. Yu. Yushankhai and N. A. Mironova-Ulmane, "Magnetostructural
   phase transitions in NiO and MnO: Neutron diffraction data", JETP Lett. 104, 88 (2016).
   https://doi.org/10.1134/S0021364016140071
2. C. Glorieux, J. Caerels and J. Thoen, "Photoacoustic and photopyroelectric investigation of the antiferromagnets
   CoO and Cr2O3 in the neighbourhood of the Neel Temperature", J. Phys. IV 4, C7-267 (1994).
   https://doi.org/10.1051/jp4:1994763
3. F. Gronvold and E. J. Samuelsen, "Heat capacity and thermodynamic properties of alpha-Fe2O3 in the region
   300-1050 K. Antiferromagnetic transition", J. Phys. Chem. Solids 36, 249 (1975).
   https://doi.org/10.1016/0022-3697(75)90017-7
4. B. S. Hemingway, American Mineralogist 75, 781 (1990). https://msaweb.org/AmMin/AM75/AM75_781.pdf

Prediction: S_pred(T) = S_vib(T) + S_mag(T) + S_el(T), with

- S_vib from the frozen closed-shell oxide route, run on the Materials Project structure of each oxide;
- S_mag = `magnetic_properties(T, T_N, 2S, 0.28, reference="ordered", ions_per_formula=..., atoms_per_formula=...)`
  (the entropy is the same for every reference; only the Gibbs energy differs);
- S_el = `electronic_entropy` of the Materials Project GGA+U density of states of the same material (summed spins,
  `fermi_energy=dos.efermi`, `atoms=len(dos.structure)`); expected near zero and reported.

### Measured values, selection fixed now

From `juno/assessments/Refractory-Oxides/manifest.json`, read-only, the crystalline phase stable at the temperature:

- S(298.15 K): the mean of every evaluated compilation in the manifest for that oxide (TKV, NIST-JANAF, CRC, NEA,
  Gamsjager as present), their spread reported as the data uncertainty. No compilation is dropped.
- S(1000 K): (1) S tabulated directly by NIST-JANAF, if ingested into the manifest before scoring; else (2)
  S(298.15) + int C_p / T dT from the Robie and Hemingway 1995 heat-capacity model where the manifest holds it and
  its stated range covers 298-1000 K; else not scored, reported as "no data". Tabulated C_p at a few hundred kelvin
  spacing is not integrated: it misses the Neel peak of Fe2O3 at 950 K.

### Pass criteria

- A1 (primary): for NiO, MnO, Fe2O3 and Cr2O3, |S_pred - S_meas| <= 1.0 at 298.15 K, and at 1000 K where scored.
  All must hold.
- A2 (reported): for each of the four, whether adding S_mag reduces |S_pred - S_meas| at 298.15 K.
- A3 (prediction, reported, not in the verdict): CoO falls short by more than 1.0 at 1000 K. Co2+ on an octahedral
  site has an orbital moment; its 4T1g term holds up to R ln 12 against the spin-only R ln 4, at most R ln 3 = 9.1
  per mole of Co (4.6 per mole of atoms) more. If CoO passes A1's tolerance instead, the report says the prediction
  failed.

If the closed-shell oxide test of the vibrational route fails, Arm A is still scored and reported, labelled as limited
by the vibrational route; A2 is then the magnetic term's own check.

## Arm B: lanthanide orthophosphates LnPO4

### Data to collect first

The Juno manifests hold only formation enthalpies for LnPO4 (Ushakov et al. 2001, `Phosphate-minerals`); no S(298.15)
or C_p. Before scoring, a session that does not see Hermes predictions ingests, with provenance, the calorimetric
S(298.15) and S(T) or C_p(T) of LaPO4, LuPO4 and YPO4 and of every open-4f LnPO4 with published data (primary
low-temperature adiabatic calorimetry, such as the Konings and Gavrichev groups' work). For each it records, from the
source's text, whether the reported S(298.15) includes the entropy released by low-temperature magnetic order or the
ground level's degeneracy R ln g_0 (measured through, or added by extrapolation) or not.

Crystal-field levels: one set per ion, chosen before any thermal data are read, written to `levels.json` with
citations, and hashed at freezing. Priority: inelastic neutron scattering or optical spectroscopy on that LnPO4; else
the ion doped into LaPO4 (monazite members Ce to Gd) or LuPO4 or YPO4 (xenotime members Tb to Yb). Every level of the
ground J multiplet, which `free_ion_entropy` and the sum of degeneracies must agree on; for Sm3+ and Eu3+ the levels
of the first excited multiplet as well. Gd3+ (8S7/2, splitting below 1 cm^-1) is one eight-fold level, a built-in
check: R ln 8 = 17.29 per mole of Gd at any temperature above a few kelvin. Pm is left out.

### Lattice part, fixed now

The excess entropy is the measured entropy less a lattice part from closed-shell members of the same structure, as
Konings and coworkers separate it:

- monazite members (Ce, Pr, Nd, Sm, Eu, Gd): S_lat(T) = S_LaPO4(T);
- xenotime members (Tb, Dy, Ho, Er, Tm, Yb): S_lat(T) = S_LuPO4(T), the closest mass; S_YPO4(T) is the second
  estimate, and the difference between the two is reported as the lattice uncertainty.

### Pass criteria

- B1 (primary): for every open-4f LnPO4 with data, |S_cf,pred - (S_meas - S_lat)| <= 1.0 J/(K mol of Ln) at
  298.15 K, and at the highest temperature up to 1000 K where the source gives S(T) or C_p(T). S_cf,pred is
  `crystal_field_properties(levels, degeneracies, T)["entropy"]` with `include_ground_degeneracy` set by the
  source's convention as recorded at ingestion (True if it includes R ln g_0). All must hold.
- B2 (reported): the full prediction S_vib + S_cf + S_el against S_meas per mole of atoms, with the frozen
  vibrational route on the Materials Project structure.

## Verdict

PASS only if A1 and B1 both hold. Anything else is FAIL, reported with every number, as earlier Hermes tests were.

## What this test does not cover

- Whether the MLIP's vibrations of a magnetic oxide are right: the MLIP has no spins, and its training data carry the
  magnetic order the Materials Project calculation used (often ferromagnetic for an antiferromagnet).
- Gibbs energies: only entropies are scored. The ordering enthalpy, which the `reference` choice governs, is not.
- The electronic term on a metal; both arms are insulators, where it is expected to be near zero.
