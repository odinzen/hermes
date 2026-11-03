"""Forces and energy of the parity frame on the local GPU, for the cross-machine check sol_run.py makes on Sol.
Runs in the WSL MACE venv the frozen rule names; writes reference_forces.npz."""
from pathlib import Path

import numpy as np
import torch

import sol_run
from sol_run import ro

ro.MODEL = Path.home() / ".cache/mace" / ro.MODEL.name
atoms = sol_run.reference_frame()
atoms.calc = ro.calculator(cueq=True)
np.savez(Path(__file__).resolve().parent / "reference_forces.npz", positions=atoms.get_positions(),
         forces=atoms.get_forces(), energy=atoms.get_potential_energy(), device=torch.cuda.get_device_name(0))
print(len(atoms), "atoms, E =", atoms.get_potential_energy(), "on", torch.cuda.get_device_name(0))
