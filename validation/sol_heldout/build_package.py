"""Assemble the held-out Sol package folder from this directory, the unchanged run_oxides.py, the structures of the cases and
the model weights. Text files are written with LF endings. Run in WSL:  python3 build_package.py DEST"""
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIB = HERE.parent / "oxide_vibration"
MACE_CACHE = Path.home() / ".cache/mace"
TEXT = ["sol_config.sh", "env.yml", "cases.txt", "setup_env.sh", "submit.sh", "run_check.slurm", "run_case.slurm",
        "test_local.sh", "sol_run.py", "CHECKLIST.md"]

dest = Path(sys.argv[1])
if dest.exists():
    raise SystemExit(f"{dest} exists; move it to the Recycle Bin first so only one package is there")
(dest / "structures").mkdir(parents=True)
(dest / "weights").mkdir()
for name in TEXT:
    (dest / name).write_bytes((HERE / name).read_bytes().replace(b"\r\n", b"\n"))
(dest / "run_oxides.py").write_bytes((VIB / "run_oxides.py").read_bytes().replace(b"\r\n", b"\n"))
tags = {line.split()[1] for line in (HERE / "cases.txt").read_text().splitlines() if line.strip()}
tags = sorted(tags | {"cao"})  # CaO is also the check frame
for tag in tags:
    shutil.copy(VIB / "structures" / f"{tag}.cif", dest / "structures")
# the model table of sol_run.py, read as text: importing it would need the MD packages
MODELS = re.findall(r'"([\w.-]+)": \("(\w+)", "[0-9a-f]{64}"\)', (HERE / "sol_run.py").read_text())
for model, weights in MODELS:
    shutil.copy(HERE / f"reference_forces_{model}.npz", dest)
    shutil.copy(MACE_CACHE / weights, dest / "weights")
for f in dest.glob("*.sh"):
    f.chmod(0o755)
for f in dest.glob("*.slurm"):
    f.chmod(0o755)
print(f"{dest}: {len(tags)} structures, {sum(1 for _ in dest.rglob('*') if _.is_file())} files")
