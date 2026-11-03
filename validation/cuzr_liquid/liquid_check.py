"""hermes.liquid_entropy against calphy on the Mendelev EAM: Cu 1300 K, Cu-Zr liquids 2300 K."""
import json, sys, time
import numpy as np
sys.path.insert(0, r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam")
from analyze_cuzr import read_dump
from hermes.liquid_entropy import liquid_entropy
RUNS = r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\runs"
truth = json.load(open(r"C:\Users\busta\Code\_wt_station2_llzo\entropy\binary_cuzr_eam\results\cuzr_liquid_mixing.json"))
masses = {0: 63.546, 1: 91.224}
out = {}
t0 = time.time()
fr, ty, cell = read_dump(rf"{RUNS}\fus_liquid.dump")
r = liquid_entropy(fr, cell, ty, masses, 1300.0)
out["Cu_1300"] = {"S": r["total"], "S2": r["two_body"], "exact": 82.542}
print("Cu 1300 K", round(r["total"], 2), "exact 82.54", f"{time.time()-t0:.0f}s", flush=True)
for tag, v in truth["per_composition"].items():
    fr, ty, cell = read_dump(rf"{RUNS}\x{tag}.dump")
    r = liquid_entropy(fr, cell, ty, masses, 2300.0)
    out[tag] = {"x_Zr": v["x_Zr"], "S": r["total"], "S2": r["two_body"], "S_ideal": r["ideal"], "exact": v["S_truth"]}
    print(tag, round(v["x_Zr"], 3), "S", round(r["total"], 2), "exact", round(v["S_truth"], 2), "diff", round(r["total"] - v["S_truth"], 2), flush=True)
pc = {t: o for t, o in out.items() if t != "Cu_1300"}
def mix(k):
    return {t: o[k] - (1 - o["x_Zr"]) * pc["000"][k] - o["x_Zr"] * pc["100"][k] for t, o in pc.items()}
m, e = mix("S"), mix("exact")
ideal = truth["dS_mix"]["ideal"]
print("x_Zr   dS_mix hermes   exact   ideal")
for t in pc:
    print(f"{pc[t]['x_Zr']:.3f}  {m[t]:7.2f}  {e[t]:7.2f}  {ideal[t]:7.2f}")
out["dS_mix"] = {"hermes": m, "exact": e, "ideal": ideal}
json.dump(out, open("liquid_check.json", "w"), indent=1)
