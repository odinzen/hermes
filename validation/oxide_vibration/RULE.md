# Vibrational entropy of closed-shell oxides against measured entropies

Status: FROZEN 2026-10-23 on Michael's approval ("I want you to do it"). Not run. No measured entropy or heat capacity was read while writing
it; the Juno manifests were searched only for which records exist, never their values.

This is the test `validation/extra_terms/RULE.md` (Arm A) points to for its vibrational part. It fixes the route,
potential and settings that Arm A then applies unchanged to NiO, MnO, CoO, Fe2O3 and Cr2O3. It is scored and reported
on its own, pass or fail, before Arm A is run.

When frozen, `PREREGISTERED.txt` holds the date and the SHA-256 (line endings normalised to LF) of this file, of the
code below, and of the potential. The run and analysis scripts, the structure list (`structures.json`) and the
measured-data ingest are written after freezing, hashed into it with a date, and all of that happens before any
production MD starts. Nothing in this file changes after freezing, whatever the results; a fix found by this test
gets a new test on new data.

## Code and potential under test

| Item | SHA-256 (LF) |
|---|---|
| `hermes/compression_entropy.py` (`gaussian_entropy`, `crystal_correlation_entropy`) | 58e95b9206d6d287ce4ca1ad5faf18fd7d6a671d27f34a48edb1430643552fb6 |
| `hermes/heat_capacity.py` (`entropy_from_heat_capacity`) | 541d7a3483bf84adc8949d6a492450e954284d4ff050bf7b6730e11838552b80 |
| `hermes/quantum_correction.py` (`vibrational_density_of_states`, `quantum_correction`) | 8d968a9b7c40ef36cbed0a64b6fe25339abb39de40d9b8deeb9db510928bbdf8 |
| MACE-MP-0 medium [1], `20231203mace128L1_epoch199model` (binary, not normalised) | 01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845 |

MD by ASE 3.29.0 with mace-torch 0.3.16 (WSL venv `~/scratch/mace-gpu`), float32, cuEquivariance kernels on the GPU,
or the same settings on the CPU if a box does not fit in GPU memory. Before production, forces with and without
cuEquivariance are compared on one CaO frame; they must agree within 1 meV/A or cuEquivariance is not used.

## Cases, fixed now

The polymorph stable at 298.15 K and 1 bar, taken from the Materials Project entry with that space group and the
lowest energy above the hull. The mp-id of each is written to `structures.json` at setup.

| Oxide | Polymorph | Space group |
|---|---|---|
| MgO | periclase | Fm-3m |
| Al2O3 | corundum | R-3c |
| TiO2 | rutile | P4_2/mnm |
| Y2O3 | bixbyite (C-type) | Ia-3 |
| Sc2O3 | bixbyite | Ia-3 |
| Lu2O3 | bixbyite | Ia-3 |
| La2O3 | A-type | P-3m1 |
| SrTiO3 | cubic perovskite | Pm-3m |
| MgTiO3 | geikielite (ilmenite) | R-3 |
| Mg3(PO4)2 | farringtonite | P2_1/n |

All are closed-shell (Ti4+ d0, La3+ 4f0, Lu3+ 4f14) and insulators with gaps of 3 eV or more, so no magnetic,
crystal-field or electronic term is added. CaO is the smoke-test case for plumbing; its prediction is not compared
with any measurement before the verdict.

Box: the conventional standard cell repeated n_a x n_b x n_c, each n the smallest giving a perpendicular width of at
least 12 A (twice the potential's 6 A cutoff), with repeats then added along the narrowest width until the box holds
at least 200 atoms.

## Route, fixed now

Classical MD, then the quantum part from the vibrational spectrum. T0 = 298.15 K, T1 = 1000 K. Timestep 1 fs for all
runs. Seed 4928.

1. Ladder: NPT (ASE `MTKNPT`, fully anisotropic cell, P = 0, thermostat damping 0.1 ps, barostat damping 1 ps) at
   seven temperatures evenly from T0 to T1, each 10 ps to settle and 30 ps averaged. H = <PE> + 3/2 k_B T + <P V> per
   atom, the same as the third fresh test. The T0 and T1 runs also give the mean cell for step 2.
2. Anchor and top: NVT (ASE `Bussi`, damping 0.1 ps) at the mean cell of T0 and of T1, 10 ps to settle, then 100 ps at
   T0 and 30 ps at T1, with positions every 0.25 ps (401 and 121 frames) and velocities every 4 fs.
3. S_cl(T0) = `gaussian_entropy(positions, cell, masses, T0, species)["entropy"]`.
4. S_cl(T1) = `entropy_from_heat_capacity(ladder T, H, T0, S_cl(T0), at=T1)["entropy"]` (cubic fit).
5. QC(T) = `quantum_correction(*vibrational_density_of_states(velocities, 0.004, masses, segment_ps=5.0), T)` at T0
   and T1.
6. S_vib(T) = S_cl(T) + QC(T), in J/(K mol of atoms).

Classical MD gives classical thermal expansion, without zero-point expansion; this is left in, as in any quasi-classical
route, and is reported only through the result.

## Measured values, selection fixed now

The same selection Arm A uses, from `juno/assessments/Refractory-Oxides/manifest.json`, read-only, for the crystalline
phase named above:

- S(298.15 K): the mean of every evaluated compilation in the manifest for that oxide, their spread reported as the
  data uncertainty. No compilation is dropped.
- S(1000 K): (1) S tabulated directly by NIST-JANAF, if in the manifest before scoring; else (2) S(298.15) +
  int C_p / T dT from the Robie and Hemingway 1995 heat-capacity model where the manifest holds it and its stated range
  covers 298-1000 K; else not scored, reported as "no data".

The manifest has no entropy records yet for Sc2O3, Lu2O3, SrTiO3, MgTiO3 and Mg3(PO4)2, nor S(298.15) for MgO. Before
any production MD starts, they are ingested with provenance from the evaluated compilations NIST-JANAF [2], Robie and
Hemingway [3] and, for the lanthanide oxides, Konings et al. [4], every one of those that tabulates the oxide, and the
ingest is hashed into `PREREGISTERED.txt`. An oxide still without S(298.15) after the ingest is not scored and is
reported as "no data". Measured values per formula unit are divided by the atoms per formula unit.

## Validity, fixed now

A case is valid if `gaussian_entropy` accepts the T0 trajectory as a crystal and the time-averaged structure at T0
and at T1 keeps the space group above (spglib, symprec 0.1 A). An invalid case is excluded and reported with the
reason. With more than two cases excluded the test is void. A case with no data is neither valid nor excluded; with
fewer than six cases scored the test is void.

## Pass criteria

- C1 (primary): for every valid case with data, |S_vib - S_meas| <= 1.0 J/(K mol of atoms) at 298.15 K, and at
  1000 K where scored. All must hold.
- Reported, outside the verdict: the sign and mean of the errors (a potential whose vibrations are too soft reads
  high); QC at each temperature; the purely harmonic quantum entropy of the T0 spectrum (`harmonic_entropy`) as a
  second estimate at T0; the fit residual of H(T); the data spread.

## Verdict

PASS only if C1 holds and the test is not void. Anything else is FAIL, reported with every number.

## What this test does not cover

- Magnetic oxides: the potential has no spins. Arm A applies this route to them unchanged.
- Gibbs energies and enthalpies: only entropies are scored.
- Whether a failure comes from the route or from the potential. The route alone was checked against thermodynamic
  integration on metals (third fresh test); this test checks route and potential together against measurement.

## References

1. I. Batatia et al., "A foundation model for atomistic materials chemistry", arXiv:2401.00096 (2023).
2. M. W. Chase Jr., NIST-JANAF Thermochemical Tables, 4th ed., J. Phys. Chem. Ref. Data Monograph 9 (1998).
   https://doi.org/10.18434/T42S31
3. R. A. Robie and B. S. Hemingway, "Thermodynamic properties of minerals and related substances at 298.15 K and
   1 bar (10^5 pascals) pressure and at higher temperatures", U.S. Geological Survey Bulletin 2131 (1995).
   https://doi.org/10.3133/b2131
4. R. J. M. Konings, O. Benes, A. Kovacs, D. Manara, D. Sedmidubsky, L. Gorokhov, V. S. Iorish, V. Yungman,
   E. Shenyavskaya and E. Osina, "The thermodynamic properties of the f-elements and their compounds. Part 2. The
   lanthanide and actinide oxides", J. Phys. Chem. Ref. Data 43, 013101 (2014). https://doi.org/10.1063/1.4825256
