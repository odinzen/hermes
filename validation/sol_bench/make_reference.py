"""Forces and energy of the parity frame on the local GPU for each benchmark model, for the cross-machine check sol_run.py
makes on Sol. Runs in the WSL MACE venv; writes reference_forces_<model>.npz."""
from pathlib import Path

import numpy as np
import torch

import sol_run
from sol_run import ro

for model, (weights, _) in sol_run.MODELS.items():
    ro.MODEL = Path.home() / ".cache/mace" / weights
    atoms = sol_run.reference_frame()
    atoms.calc = ro.calculator(cueq=True)
    np.savez(Path(__file__).resolve().parent / f"reference_forces_{model}.npz", positions=atoms.get_positions(),
             forces=atoms.get_forces(), energy=atoms.get_potential_energy(), device=torch.cuda.get_device_name(0))
    print(model, len(atoms), "atoms on", torch.cuda.get_device_name(0))
