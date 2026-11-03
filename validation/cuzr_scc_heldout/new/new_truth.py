"""Exact S_ex and H_mix at the new compositions, by the formula of station2 analyze_cuzr.py.

S = (<U> + 3/2 kT - G) / T with G from calphy and <U> from the 864-atom NVT run at the zero-pressure volume;
mixing quantities are taken against the pure ends of the station2 run (same potential, T and protocol). Before
the new points, the station2 x050 point is recomputed from its own files and must match its JSON.
"""
import glob
import json
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
STATION2 = r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam"
EV = 96485.33212  # J/mol per eV/atom
KB = 8.617333262e-5
R = 8.314462618
T = 2300.0
NEW = {"010": 86, "028": 242, "043": 372, "060": 518, "080": 691, "090": 778}


def mean_energy(path):
    rows = [ln.split() for ln in open(path) if not ln.startswith("#")]
    return float(rows[-1][1])


def calphy_g(folder, tag):
    (d,) = [d for d in glob.glob(os.path.join(folder, f"x{tag}-fe-*")) if os.path.isdir(d)]
    return float(yaml.safe_load(open(os.path.join(d, "report.yaml")))["results"]["free_energy"])


def mixing(u, g, x, ref):
    s = (u + 1.5 * KB * T - g) / T * EV
    pure = ref["per_composition"]
    s_mix = s - (1 - x) * pure["000"]["S_truth"] - x * pure["100"]["S_truth"]
    ideal = -R * (x * np.log(x) + (1 - x) * np.log(1 - x))
    h_mix = (u - (1 - x) * pure["000"]["U_eV_atom"] - x * pure["100"]["U_eV_atom"]) * EV
    return s_mix - ideal, h_mix


def main():
    ref = json.load(open(os.path.join(STATION2, "results", "cuzr_liquid_mixing.json")))
    x050 = ref["per_composition"]["050"]
    s_ex, h_mix = mixing(mean_energy(os.path.join(STATION2, "runs", "x050_thermo.dat")), calphy_g(STATION2, "050"),
                         0.5, ref)
    want_s = ref["dS_mix"]["truth_calphy"]["050"] - ref["dS_mix"]["ideal"]["050"]
    want_h = ref["dS_mix"]["mixing_enthalpy_kJ"]["050"] * 1000
    assert abs(s_ex - want_s) < 1e-9 and abs(h_mix - want_h) < 1e-6, (s_ex, want_s, h_mix, want_h)
    print(f"check x050: S_ex {s_ex:.4f} = {want_s:.4f}, H_mix {h_mix:.1f} = {want_h:.1f}  (U {x050['U_eV_atom']})")

    out = {"T_K": T, "units": "S_ex J/(mol-atom K), H_mix J/mol-atom", "x_Zr": {}, "U_eV_atom": {}, "G_eV_atom": {},
           "S_ex": {}, "H_mix_J": {}}
    for t, n_zr in NEW.items():
        x = n_zr / 864
        u, g = mean_energy(os.path.join(HERE, f"x{t}_thermo.dat")), calphy_g(HERE, t)
        s_ex, h_mix = mixing(u, g, x, ref)
        for key, val in (("x_Zr", x), ("U_eV_atom", u), ("G_eV_atom", g), ("S_ex", s_ex), ("H_mix_J", h_mix)):
            out[key][t] = val
        print(f"x{t} (x_Zr {x:.4f}): U {u:.4f}, G {g:.5f} eV/atom, S_ex {s_ex:6.3f}, H_mix {h_mix:8.1f}")
    json.dump(out, open(os.path.join(HERE, "truth.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
