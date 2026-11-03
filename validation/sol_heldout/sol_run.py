"""Sol entry point for the held-out oxide test (../oxide_heldout/RULE.md, frozen 2026-10-28): the 11 held-out cases with
MACE MPA-0. Runs ../oxide_vibration/run_oxides.py unchanged (a byte-identical copy sits in the package) with the MPA-0
weights from this package, and writes every output to the results folder beside it.

    python sol_run.py check RESULTS                      parity gate and force check against the local GPU, each model
    python sol_run.py case MODEL TAG RESULTS [--smoke]   one case into RESULTS/MODEL/runs/TAG, resuming by stage
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_oxides as ro  # noqa: E402

MODELS = {  # name: (weights file, SHA-256), the potential frozen in RULE.md; MIT licensed
    "mpa-0": ("macempa0mediummodel", "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"),
}
RUN_OXIDES = "8874d9a258624d6d4dbd044763bcbcf904e31528226780c3f75c9ada56b6ce25"
# Same frame and tolerance as the cuEquivariance gate; the energy tolerance is 0.1 meV/atom.
FORCE_TOL, ENERGY_TOL = 1e-3, 1e-4


def use(model):
    ro.MODEL = HERE / "weights" / MODELS[model][0]


def check_frozen():
    if hashlib.sha256((HERE / "run_oxides.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest() != RUN_OXIDES:
        raise SystemExit("run_oxides.py differs from the frozen file; refusing to run")
    for name, (weights, want) in MODELS.items():
        if hashlib.sha256((HERE / "weights" / weights).read_bytes()).hexdigest() != want:
            raise SystemExit(f"the {name} weights differ from the recorded file; refusing to run")


def reference_frame():
    """The parity frame of run_oxides.parity(): CaO box, seeded velocities, seeded 0.05 A rattle."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = ro.build_box("cao")
    MaxwellBoltzmannDistribution(atoms, temperature_K=ro.T0, rng=np.random.default_rng(ro.SEED))
    atoms.rattle(0.05, seed=ro.SEED)
    return atoms


def check(results):
    import torch
    report, passed = {}, True
    for model in MODELS:
        use(model)
        parity_ok = ro.parity()
        shutil.move(HERE / "parity.json", results / f"parity_{model}.json")
        ref = np.load(HERE / f"reference_forces_{model}.npz")
        atoms = reference_frame()
        if not np.allclose(atoms.get_positions(), ref["positions"], atol=1e-6):
            raise SystemExit("the reference frame differs from the one computed locally")
        atoms.calc = ro.calculator(cueq=True)
        df = float(np.abs(atoms.get_forces() - ref["forces"]).max())
        de = float(abs(atoms.get_potential_energy() - float(ref["energy"])) / len(atoms))
        ok = parity_ok and df <= FORCE_TOL and de <= ENERGY_TOL
        report[model] = {"cueq_parity": json.loads((results / f"parity_{model}.json").read_text()),
                         "max_force_difference_eV_A": df, "energy_difference_eV_atom": de,
                         "local_device": str(ref["device"]), "sol_device": torch.cuda.get_device_name(0),
                         "verdict": "PASS" if ok else "FAIL"}
        passed &= ok
    report["verdict"] = "PASS" if passed else "FAIL"
    (results / "check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return passed


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    check_frozen()
    if mode == "check":
        results = Path(args[0])
        results.mkdir(parents=True, exist_ok=True)
        sys.exit(0 if check(results) else 1)
    if mode == "case":
        model, tag, results = args[0], args[1], Path(args[2])
        gate = results / "check.json"
        if "--smoke" not in args and not (gate.exists() and json.loads(gate.read_text())["verdict"] == "PASS"):
            raise SystemExit("no passing check.json in the results folder; run the check first")
        use(model)
        ro.run_case(tag, ro.SMOKE if "--smoke" in args else ro.FULL, cueq=True, root=results / model / "runs")
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
