# Polymorph entropy differences and a fixed offset, on a second set of held-out oxides

Status: FROZEN 2026-11-02 on Michael Bustamante's approval (`PREREGISTERED.txt`). Not run. The measured entropies
of these compounds were ingested into Juno by script without being shown (juno 29db75f); nobody has read them.

The held-out oxide test (`validation/oxide_heldout/RULE.md`, frozen 2026-10-28) failed on absolute entropies and
passed on polymorph differences. MPA-0 put every one of its 11 oxides too high, by +1.05 to +2.94 J/(K mol of atoms)
at 298.15 K, while the three entropy differences among the Al2SiO5 polymorphs came within 0.41 of JANAF's. Two claims
follow from that, and neither is yet tested on anything unseen.

1. Entropy differences between polymorphs transfer, in families other than Al2SiO5.
2. The excess in absolute entropy is a constant of the potential, so that subtracting a value fixed beforehand brings
   new compounds within about 1 J/(K mol of atoms) of the measurement.

This test puts both to compounds none of the earlier work has seen.

When frozen, `PREREGISTERED.txt` holds the date and the SHA-256 (line endings normalised to LF) of this file, of the
code below, of the potential, of the run and analysis scripts and of the ingested measured records, all before any MD
of these cases starts. Nothing in this file changes after freezing, whatever the results.

## Code under test and potential

Unchanged from the held-out test, with the same SHA-256 (line endings normalised to LF):

    hermes/compression_entropy.py  512e1cf35bdd0b0e5728fe8c07f8dd1f707e5891e0f793a74a4e25176a246c77
    hermes/heat_capacity.py        541d7a3483bf84adc8949d6a492450e954284d4ff050bf7b6730e11838552b80
    hermes/quantum_correction.py   8d968a9b7c40ef36cbed0a64b6fe25339abb39de40d9b8deeb9db510928bbdf8

MACE MPA-0 medium (MIT licence), weights `macempa0mediummodel`, SHA-256
75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638. `validation/oxide_vibration/run_oxides.py`
unchanged (8874d9a2...). MD on Sol A100s by a package built like `validation/sol_heldout/`, gated by the
cuEquivariance parity and the MPA-0 force check against the local GPU.

## Offset, fixed now from the 19 compounds already scored

The offset is the mean MPA-0 error, `limit="bulk"`, over every closed-shell oxide scored so far whose trajectory kept
its space group: the eight development oxides of the benchmark (`validation/sol_bench/bench_results.json`) and the 11
held-out oxides (`validation/oxide_heldout/heldout_results.json`).

| T (K) | Compounds | Offset D(T) | SD | RMS after subtracting D | Largest abs residual |
|---|---|---|---|---|---|
| 298.15 | 19 | +1.87 | 0.76 | 0.74 | 1.38 (La2O3) |
| 1000 | 16 | +2.20 | 0.67 | 0.65 | 1.83 (SrO) |

J/(K mol of atoms). Three development oxides (Y2O3, Lu2O3, La2O3) have no measured value at 1000 K. The phosphates
and the magnetic oxides are left out, as they carry crystal-field or magnetic terms.

## Cases, fixed now

Chosen 2026-11-02 by structure type and the availability of a high-temperature table in Robie and Hemingway (1995),
before any value was read. Closed-shell, no cation disorder, none in the development or held-out sets. Each polymorph
family has every member measured in that one compilation.

| Case | Formula | Mineral | Space group | Materials Project | Box (atoms) | Scored at | R&H page |
|---|---|---|---|---|---|---|---|
| quartz | SiO2 | alpha-quartz | P3_121 | mp-7000 | 243 | 298.15 K | 214 |
| coesite | SiO2 | coesite | C2/c | mp-11684 | 384 | 298.15 K | 216 |
| stishovite | SiO2 | stishovite | P4_2/mnm | mp-6947 | 270 | 298.15 K | 19 |
| calcite | CaCO3 | calcite | R-3c | mp-3953 | 270 | 298.15 K | 268 |
| aragonite | CaCO3 | aragonite | Pnma | mp-4626 | 360 | 298.15 K | 267 |
| wollastonite | CaSiO3 | wollastonite | P-1 | mp-4429 | 240 | 298.15, 1000 K | 342 |
| pseudowollastonite | CaSiO3 | pseudowollastonite | C2/c | mp-561086 | 480 | 298.15, 1000 K | 343 |
| enstatite | MgSiO3 | orthoenstatite | Pbca | mp-5834 | 480 | 298.15, 1000 K | 352 |
| clinoenstatite | MgSiO3 | low clinoenstatite | P2_1/c | mp-3470 | 480 | 298.15, 1000 K | 353 |
| perovskite | CaTiO3 | perovskite | Pnma | mp-4019 | 360 | 298.15, 1000 K | 232 |
| chrysoberyl | BeAl2O4 | chrysoberyl | Pnma | mp-3081 | 504 | 298.15, 1000 K | 229 |
| diopside | CaMgSi2O6 | diopside | C2/c | mp-6766 | 480 | 298.15, 1000 K | 347 |
| jadeite | NaAlSi2O6 | jadeite | C2/c | mp-6124 | 480 | 298.15, 1000 K | 356 |

Structures are the Materials Project entries named above, written as the conventional standard cell, in the box of
the earlier tests (every perpendicular width at least 12 A, at least 200 atoms). The entries are named rather than
picked by the earlier rule, the lowest energy above the hull in the space group, because that rule picks other
frameworks for coesite (mp-12787, 36 atoms) and jadeite (mp-1020661, 160 atoms). Each named entry has the mineral's
atom count, and its cell within 2 % of the measured one (the usual PBE expansion).

SiO2 and CaCO3 are scored at 298.15 K only. Quartz turns to beta-quartz at 846 K, aragonite turns to calcite on
heating, and stishovite has only a 298.15 K value. The MD still runs the full ladder, since `run_oxides.py` is
unchanged.

Left out: phenakite (Be2SiO4, a 1008-atom box) and the garnets grossular and pyrope (1280 atoms), for cost;
cristobalite and tridymite, for transitions and disorder near room temperature; anatase, because rutile is a
development case; the Mg2SiO4 high-pressure polymorphs, not in Robie and Hemingway; ZnO and SnO2, single compounds
whose entropies are already in the Juno manifest, when other single compounds serve the offset without that history;
akermanite, for its incommensurate modulation; MgAl2O4, for its cation disorder. No magnetic, crystal-field or electronic term is added.

## Route, fixed now

As the held-out test: Gaussian anchor at T0 = 298.15 K with `limit="bulk"` plus QC, H(T) from the NPT ladder carrying
it to T1 = 1000 K plus QC there. Timestep 1 fs, seed 4928, the same NPT ladder and NVT lengths.

## Measured values, fixed now

R. A. Robie and B. S. Hemingway (1995), one compilation for every case: S at 298.15 K and, where scored, at 1000 K,
per formula unit, divided by the atoms per formula unit. They are read by script from the text layer of the PDF into
`juno/assessments/Refractory-Oxides/manifest.json` as records `meas:robie_hemingway_1995:heldout2_*`, without being
shown. The script checks every row it reads against the table's own identity, S = (H - H298)/T - (G - H298)/T, to
the table's rounding, and the 298.15 K row against the 298.15 K summary table, reporting pass or fail only. A case
that fails is dropped before freezing, its value still unseen.

Several of these values have been used before, outside Hermes. The Hephaestus CaO-SiO2, MgO-SiO2 and clinopyroxene
assessments take Robie and Hemingway's tables for wollastonite, pseudowollastonite, enstatite and diopside as
endmember data; quartz's JANAF entropy has been in the Juno manifest since the Refractory-Oxides assessment; and the
metastability paper used the enthalpy differences of the SiO2, MgSiO3 and CaSiO3 pairs. None has been compared with
an MLIP entropy. Blindness to published compilations cannot be complete, so the protection is that nothing
adjustable remains after freezing: the potential, code, offset, tolerances and cases are all fixed here.

## Validity, fixed now

As the earlier tests: `gaussian_entropy` accepts the T0 trajectory as a crystal, and the time-averaged structure keeps
the space group above (spglib, symprec 0.1 A) at each temperature where the case is scored. An invalid case is
excluded and reported with the reason. The test is void with more than three of the 13 cases excluded, or with fewer
than two polymorph families left for C2.

## Pass criteria

Residual r = S_vib - D(T) - S_meas, J/(K mol of atoms).

- C1 (primary): at each temperature, over the valid cases scored there, the RMS of r is at most 1.0 and every |r| is
  at most 2.0.
- C2 (primary): within each polymorph family, every pairwise entropy difference is within 0.5 of Robie and
  Hemingway's, at each temperature the family is scored: coesite - quartz, stishovite - quartz, stishovite - coesite,
  aragonite - calcite (298.15 K); pseudowollastonite - wollastonite, clinoenstatite - enstatite (298.15 and 1000 K).
  Eight comparisons, all must hold. The offset cancels in a difference. A family with an invalid member is not
  applied.
- Reported, outside the verdict: the uncorrected errors, their mean and sign against D(T); the box value beside the
  bulk value; QC at each temperature; the harmonic entropy of the T0 spectrum; the fit residual of H(T); the MD molar
  volume at 298.15 K against Robie and Hemingway's.

## Verdict

PASS only if C1 and C2 hold and the test is not void. Anything else is FAIL, reported with every number. C1 and C2
are each reported as holding or failing.

## What this test does not cover

- Magnetic oxides, crystal-field and electronic terms.
- Another potential: only MPA-0 is run, and the offset belongs to it.
- Unseen structures: these crystals are in the potential's training data as energies and forces. What is held out is
  the measured entropy.
- Whether the offset holds outside closed-shell oxides and carbonates.

## References

1. R. A. Robie and B. S. Hemingway, Thermodynamic properties of minerals and related substances at 298.15 K and 1 bar
   (10^5 pascals) pressure and at higher temperatures, U.S. Geological Survey Bulletin 2131 (1995).
   https://doi.org/10.3133/b2131
