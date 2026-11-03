# Vibrational entropy of held-out closed-shell oxides against measured entropies

Status: FROZEN 2026-10-28 on Michael Bustamante's approval, C2 included (`PREREGISTERED.txt`). Not run. The
measured entropies of these compounds were ingested into Juno by script without being shown (juno 21bbc60); nobody
has read them.

The first closed-shell oxide test (`validation/oxide_vibration/RULE.md`, frozen 2026-10-23) failed. Working on its
cases afterwards, outside every verdict, found two things in Hermes, both settled against exact references rather
than measurements and released as Hermes 0.3: the entropy of the simulated box differs from that of a macroscopic
crystal (`limit="bulk"`), and a density of states whose gap DFT closes needs the measured gap (`band_gap`, not used
here, as every case below is a wide-gap insulator). It also found that the potential then used, MACE-MP-0, makes oxide
crystals 2 to 7 % too large and too soft. This test applies the corrected route and the potential chosen below, once,
to oxides none of that work has seen.

When frozen, `PREREGISTERED.txt` holds the date and the SHA-256 (line endings normalised to LF) of this file, of the
code below, of the potential and of the run and analysis scripts, all before any MD of these cases starts. Nothing in
this file changes after freezing, whatever the results.

## Code under test

Hermes 0.3: `hermes/compression_entropy.py` (`gaussian_entropy` with `limit="bulk"`), `hermes/heat_capacity.py`
(`entropy_from_heat_capacity`), `hermes/quantum_correction.py` (`vibrational_density_of_states`,
`quantum_correction`), hermes-private commit 817f412, SHA-256 with line endings normalised to LF:

    hermes/compression_entropy.py  512e1cf35bdd0b0e5728fe8c07f8dd1f707e5891e0f793a74a4e25176a246c77
    hermes/heat_capacity.py        541d7a3483bf84adc8949d6a492450e954284d4ff050bf7b6730e11838552b80
    hermes/quantum_correction.py   8d968a9b7c40ef36cbed0a64b6fe25339abb39de40d9b8deeb9db510928bbdf8

The last two are unchanged from the first test.

## Potential, chosen before any MD of these cases

MACE MPA-0 medium (MIT licence), weights `macempa0mediummodel`, SHA-256
75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638. Chosen 2026-10-28 from the three MIT-licensed MACE
foundation models, MACE-MP-0 medium, MPA-0 medium and MP-0b3 medium, each run on every development case of the first
test, Arm A and B2 with the MD settings below (`validation/sol_bench/`, commit ad277d5, every result of all three
reported in `bench_output.txt`). The choice is the smallest mean entropy error at 298.15 K by this route
(`limit="bulk"`) over the closed-shell development oxides of the first test whose trajectory keeps its space group:
MPA-0 +1.90 (8 oxides), MP-0b3 +2.22 (8), MACE-MP-0 +2.83 J/(K mol of atoms) (7; its Mg3(PO4)2 turns P-1 at 1000 K).
MPA-0's errors there run from +0.94 (Lu2O3) to +3.25 (La2O3). It also has the smallest mean error on the magnetic
oxides of Arm A, and MP-0b3 on the phosphates of B2, each over the cases that keep their symmetry. None of the three
has an imaginary harmonic mode at its own 298.15 K MD volume in any of MgO, Al2O3, TiO2, NiO, MnO, CoO, Fe2O3 and
Cr2O3, except rutile. No compound below is among the development cases, so the choice used no measurement of them.

A volume criterion drafted 2026-10-25, the smallest mean |V_MD / V_meas - 1| at 298.15 K over those eight oxides, was
set aside before freezing. It picks MACE-MP-0 (4.70 %, against 4.91 % for MP-0b3 and 5.90 % for MPA-0), but the volume
error does not follow the entropy error, MPA-0 having the largest volumes and the smallest entropy errors.

MD by ASE 3.29.0 with mace-torch 0.3.16 and cuEquivariance 0.12.0, float32, on Sol A100 GPUs by a package built like
`validation/sol_bench/`, gated as before by the cuEquivariance parity and the MPA-0 force check against the local GPU.

## Cases, fixed now

Chosen 2026-10-25 by structure type and the availability of a NIST-JANAF crystal table, before any value was read:
closed-shell, no phase change or cation disorder between 298.15 and 1000 K, none in the development set. ZnO, HfO2,
CeO2, SnO2 and ThO2 were left out for want of a JANAF table; quartz for its alpha-beta transition at 846 K; MgAl2O4
for its cation disorder.

| Case | Oxide | Polymorph | Space group | Materials Project | Box (atoms) | JANAF table |
|---|---|---|---|---|---|---|
| beo | BeO | bromellite (wurtzite) | P6_3mc | mp-2542 | 432 | Be-044 |
| cao | CaO | lime (rock salt) | Fm-3m | mp-2605 | 216 | Ca-027 |
| sro | SrO | rock salt | Fm-3m | mp-2472 | 216 | O-013 |
| bao | BaO | rock salt | Fm-3m | mp-1342 | 216 | Ba-034 |
| li2o | Li2O | antifluorite | Fm-3m | mp-1960 | 324 | Li-014 |
| zro2 | ZrO2 | baddeleyite | P2_1/c | mp-2858 | 324 | O-052 |
| mg2sio4 | Mg2SiO4 | forsterite | Pnma | mp-2895 | 504 | Mg-028 |
| zrsio4 | ZrSiO4 | zircon | I4_1/amd | mp-4820 | 288 | O-072 |
| andalusite | Al2SiO5 | andalusite | Pnnm | mp-4753 | 384 | Al-102 |
| kyanite | Al2SiO5 | kyanite | P-1 | mp-5065 | 384 | Al-103 |
| sillimanite | Al2SiO5 | sillimanite | Pnma (Pbnm) | mp-4934 | 384 | Al-104 |

Each is the Materials Project entry in that space group with the lowest energy above the hull (`structures.json`),
in the box of the first test: the conventional cell repeated until every perpendicular width is at least 12 A and the
box holds at least 200 atoms. CaO was the plumbing smoke case of the first test, run with every stage shortened; its
prediction was printed then but never compared with a measurement. No magnetic, crystal-field or electronic term is
added.

## Route, fixed now

As the first test, steps 1 to 6, with one change: step 3 is
`gaussian_entropy(positions, cell, masses, T0, species, limit="bulk")["entropy"]`. T0 = 298.15 K, T1 = 1000 K,
timestep 1 fs, seed 4928, the same NPT ladder and NVT lengths.

## Measured values, fixed now

The NIST-JANAF table above, read by script from `artifacts/Refractory-Oxides/janaf/` into
`juno/assessments/Refractory-Oxides/manifest.json` (records `meas:nist_janaf:heldout_*`, juno 21bbc60): S at 298.15 K
and at 1000 K, divided by the atoms per formula unit. One source for every case and both temperatures, so no
compilation is chosen case by case. The three Al2SiO5 polymorphs are told apart by their phase label.

## Validity, fixed now

As the first test: `gaussian_entropy` accepts the T0 trajectory as a crystal, and the time-averaged structure at T0
and T1 keeps the space group above (spglib, symprec 0.1 A). An invalid case is excluded and reported with the reason.
With more than two cases excluded the test is void.

## Pass criteria

- C1 (primary): for every valid case, |S_vib - S_meas| <= 1.0 J/(K mol of atoms) at 298.15 K and at 1000 K. All must
  hold.
- C2 (primary): the three pairwise entropy differences among the Al2SiO5 polymorphs (kyanite - andalusite,
  sillimanite - andalusite, sillimanite - kyanite), at both temperatures, each within 0.5 J/(K mol of atoms) of
  JANAF's. If any of the three polymorphs is invalid, C2 is not applied.
- Reported, outside the verdict: the mean and sign of the errors; the box value (`limit="box"`) beside the bulk value;
  QC at each temperature; the harmonic entropy of the T0 spectrum; the fit residual of H(T).

## Verdict

PASS only if C1 and C2 hold and the test is not void. Anything else is FAIL, reported with every number.

## What this test does not cover

- Magnetic oxides, crystal-field and electronic terms: none of these cases has them.
- Whether another potential would do better: the benchmark reports all three candidates on the development cases;
  only MPA-0 is run here.
- Unseen structures: these crystals, like most Materials Project entries, are in the potential's training data as
  energies and forces. What is held out is the measured entropy.

## References

1. M. W. Chase Jr., NIST-JANAF Thermochemical Tables, 4th ed., J. Phys. Chem. Ref. Data Monograph 9 (1998).
   https://doi.org/10.18434/T42S31
