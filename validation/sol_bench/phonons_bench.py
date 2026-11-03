"""Harmonic phonons of each benchmark model's own 298.15 K crystal, outside every verdict: the imaginary-mode check of
the potential choice in ../oxide_heldout/RULE.md, and the harmonic entropy at the MD cell and at the measured volume.
Phonon routine of ../explore/phonons.py. Runs in the WSL MACE venv:

    python phonons_bench.py MODEL RUNS TAG [TAG ...]     MODEL is mp-0, mpa-0 or mp-0b3 -> phonons_<model>.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "explore"))
import phonons as ph  # noqa: E402
import volumes  # noqa: E402

CACHE = Path.home() / ".cache/mace"  # where build_package.py takes the Sol copies from
WEIGHTS = {"mp-0": (ph.ro.MODEL, "01bfe221"), "mpa-0": (CACHE / "macempa0mediummodel", "75428afe"),
           "mp-0b3": (CACHE / "macemp0b3mediummodel", "2f2be696")}


def md_unit_cell(tag, runs):
    """The conventional cell with the shape and size of the 298.15 K NPT average of that model's production run."""
    unit = ph.ro.read(ph.VIB / "structures" / f"{tag}.cif")
    widths, n = ph.ro.repeats(unit.cell)
    while len(unit) * n.prod() < ph.ro.MIN_ATOMS:
        n[np.argmin(n * widths)] += 1
    ladder = json.loads((runs / tag / "ladder.json").read_text())
    unit.set_cell(np.array(ladder[0]["mean_cell"]) / n[:, None], scale_atoms=True)
    return unit, n


def main():
    model, runs, tags = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    from mace.calculators import mace_mp
    weights, want = WEIGHTS[model]
    if not hashlib.sha256(Path(weights).read_bytes()).hexdigest().startswith(want):
        raise SystemExit(f"{weights} is not the {model} file the Sol runs used")
    calc = mace_mp(model=str(weights), device="cuda", default_dtype="float64", enable_cueq=False)
    out_file = HERE / f"phonons_{model}.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for tag in tags:
        apf = volumes.OXIDES[tag][1]
        unit, n = md_unit_cell(tag, runs)
        md = ph.relax_positions(unit, calc)
        v_md = md.get_volume() / len(md) * apf * volumes.NA_CM3
        v_exp = volumes.measured_volume(tag)
        scaled = unit.copy()
        scaled.set_cell(unit.cell.array * (v_exp / v_md) ** (1 / 3), scale_atoms=True)
        results[tag] = {"md_volume": v_md, "exp_volume": v_exp, "at_md_cell": ph.harmonic(md, n, calc),
                        "at_exp_volume": ph.harmonic(ph.relax_positions(scaled, calc), n, calc)}
        r = results[tag]
        print(f"{model} {tag}: V_MD {100 * (v_md / v_exp - 1):+.2f} %  min THz {r['at_md_cell']['min_THz']:.3f}  "
              f"S_q(MD) {r['at_md_cell']['S_q']:.2f}  S_q(Vexp) {r['at_exp_volume']['S_q']:.2f}", flush=True)
        out_file.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
