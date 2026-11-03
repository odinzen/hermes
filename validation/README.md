# Validation runs

Each held-out test keeps its rule (`analyze_*.py`), the record of when it was frozen and against which code
(`PREREGISTERED.txt`), the scripts that ran it, and its output. Trajectories are not kept; the energy averages and
calphy reports the exact references come from are. Results are summarised in `docs/validation.md`.

| Folder | Test | Verdict | Code it ran on |
|---|---|---|---|
| `cuzr_scc_heldout/rep` | excess mixing, four compositions | FAIL | commit fc44717 (`hermes/excess_model.py`, `hermes/concentration_fluctuations.py`) |
| `cuzr_scc_heldout/new` | excess mixing, ten compositions | FAIL | commit fc44717 |
| `solid_exact` | solid, neighbour-pair correction | FAIL | `solid_exact/frozen_code/` |
| `solid_fresh` | solid, groups of 256 and exact shared entropy | FAIL | `solid_fresh/frozen_code/` |
| `solid_fresh2` | solid, with the centre-of-mass term | FAIL | `solid_fresh2/frozen_code/` |
| `solid_fresh3` | solid near melting, heat-capacity route | PASS | `solid_fresh3/frozen_code/` (= commit 34bfd32) |
| `extra_terms` | crystal-field, magnetic and electronic terms against measured entropies | FAIL (both arms) | `PREREGISTERED.txt` (commit 7c46c9e modules) |
| `oxide_vibration` | vibrational entropy of closed-shell oxides, MACE-MP-0, against calorimetry | FAIL | `PREREGISTERED.txt` |
| `oxide_heldout` | 11 unseen oxides, MACE MPA-0, absolute entropy and Al2SiO5 polymorph differences | FAIL (polymorph differences hold) | `PREREGISTERED.txt` (Hermes 0.3) |
| `polymorph_heldout` | 13 unseen compounds, MACE MPA-0 less a fixed offset, four polymorph families | VOID (FAIL) | `PREREGISTERED.txt` (Hermes 0.3) |

Each `frozen_code/` holds the modules exactly as frozen; their SHA-256, with line endings normalised to LF, matches
`PREREGISTERED.txt`. The analyses check those hashes against the package they import and refuse to run otherwise, so to
rerun one, point its `HERMES` path at a folder holding `hermes/` with these files, as
`solid_exact/analyze_solids_snapshot_run.py` does. The first two `frozen_code/` folders were rebuilt after the fact
by undoing later edits; the hashes prove they are the frozen versions. `offset_calibration/` holds the Einstein-solid
calibration of the compression offset, and `anharmonic_diagnosis/` the checks that located the error near melting.

The oxide tests ran on GPUs: locally, and on Arizona State University's Sol cluster from the packages in `sol/`,
`sol_bench/`, `sol_heldout/` and `sol_polymorph/` (each builds a folder to upload, checks the potential's forces on the
cluster against the local GPU before any case starts, and writes results beside it). `sol_bench/` is the benchmark of
three potentials on the development oxides, outside every verdict; `explore/` and `polymorph_heldout/post_scoring/`
hold the checks that located the oxide error in the potentials (finite-displacement phonons, exact Gaussian samples,
box size), all run after scoring and outside the verdicts. Measured values were read into a separate data store
before each freeze and are not in the repository except as printed in each test's output after scoring; the
scorers' paths to that store and to the authors' machines will not resolve elsewhere. Molecular dynamics trajectories
are not kept here.

The scripts here are MIT, as the rest of Hermes; the result files in `validation/` are released under CC BY 4.0.
