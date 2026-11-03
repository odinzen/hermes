"""Sol entry point for the validation MD. Runs ../oxide_vibration/run_oxides.py unchanged (a byte-identical copy sits
in the package), with the model weights from this package and every output in the results folder beside it.

    python sol_run.py check RESULTS            cuEquivariance parity gate and the cross-machine force check
    python sol_run.py case TAG RESULTS         one production case into RESULTS/runs/TAG, resuming by stage
    python sol_run.py case TAG RESULTS --smoke the same with the smoke lengths (rehearsal only)
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

ro.MODEL = HERE / "weights" / ro.MODEL.name
FROZEN = {
    "run_oxides.py": "8874d9a258624d6d4dbd044763bcbcf904e31528226780c3f75c9ada56b6ce25",
    f"weights/{ro.MODEL.name}": "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845",
}
# Same frame and tolerance as the cuEquivariance gate; the energy tolerance is 0.1 meV/atom.
FORCE_TOL, ENERGY_TOL = 1e-3, 1e-4


def check_frozen():
    for rel, want in FROZEN.items():
        data = (HERE / rel).read_bytes()
        if rel.endswith(".py"):
            data = data.replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != want:
            raise SystemExit(f"{rel} differs from the frozen file; refusing to run")


def reference_frame():
    """The parity frame of run_oxides.parity(): CaO box, seeded velocities, seeded 0.05 A rattle."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = ro.build_box("cao")
    MaxwellBoltzmannDistribution(atoms, temperature_K=ro.T0, rng=np.random.default_rng(ro.SEED))
    atoms.rattle(0.05, seed=ro.SEED)
    return atoms


def check(results):
    passed = ro.parity()
    shutil.move(HERE / "parity.json", results / "parity_sol.json")
    ref = np.load(HERE / "reference_forces.npz")
    atoms = reference_frame()
    if not np.allclose(atoms.get_positions(), ref["positions"], atol=1e-6):
        raise SystemExit("the reference frame differs from the one computed locally")
    atoms.calc = ro.calculator(cueq=True)
    df = float(np.abs(atoms.get_forces() - ref["forces"]).max())
    de = float(abs(atoms.get_potential_energy() - float(ref["energy"])) / len(atoms))
    import torch
    result = {"cueq_parity": json.loads((results / "parity_sol.json").read_text()),
              "vs_local": {"max_force_difference_eV_A": df, "force_tolerance": FORCE_TOL,
                           "energy_difference_eV_atom": de, "energy_tolerance": ENERGY_TOL,
                           "local_device": str(ref["device"]), "sol_device": torch.cuda.get_device_name(0)}}
    result["verdict"] = "PASS" if passed and df <= FORCE_TOL and de <= ENERGY_TOL else "FAIL"
    (results / "check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result["verdict"] == "PASS"


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    check_frozen()
    if mode == "check":
        results = Path(args[0])
        results.mkdir(parents=True, exist_ok=True)
        sys.exit(0 if check(results) else 1)
    if mode == "case":
        tag, results = args[0], Path(args[1])
        gate = results / "check.json"
        if "--smoke" not in args and not (gate.exists() and json.loads(gate.read_text())["verdict"] == "PASS"):
            raise SystemExit("no passing check.json in the results folder; run the check first")
        ro.run_case(tag, ro.SMOKE if "--smoke" in args else ro.FULL, cueq=True, root=results / "runs")
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
