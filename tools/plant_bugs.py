"""Plant each listed error in a copy of the package and check that the test suite catches it.

The working tree is never touched: the package and tests are copied to a temporary folder, one error is put in, the
matching tests are run, and the file is restored before the next. Two errors are expected to survive because they
change no result (see docs/validation.md).

    python tools/plant_bugs.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = tempfile.mkdtemp(prefix="hermes-planted-")
PY = sys.executable
shutil.copytree(os.path.join(SRC, "hermes"), os.path.join(WORK, "hermes"))
os.makedirs(os.path.join(WORK, "tests"))
for f in os.listdir(os.path.join(SRC, "tests")):
    if f.endswith(".py"):
        shutil.copy(os.path.join(SRC, "tests", f), os.path.join(WORK, "tests", f))
for root, _, files in os.walk(WORK):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            data = open(p, "rb").read().replace(bytes([13, 10]), bytes([10]))
            open(p, "wb").write(data)

LIQ, QC, CF = "hermes/liquid_entropy.py", "hermes/quantum_correction.py", "hermes/concentration_fluctuations.py"
EM, MD, CE = "hermes/excess_model.py", "hermes/md_configuration.py", "hermes/compression_entropy.py"
HC, FO = "hermes/heat_capacity.py", "hermes/formation.py"
TESTS = {LIQ: ["tests/test_liquid_entropy.py"], QC: ["tests/test_quantum_correction.py"],
         CF: ["tests/test_concentration_fluctuations.py"], EM: ["tests/test_excess_model.py"],
         MD: ["tests/test_compression_entropy.py"], HC: ["tests/test_heat_capacity.py"], FO: ["tests/test_formation.py"],
         CE: ["tests/test_compression_entropy.py", "tests/test_crystal_correlation.py"]}
NL = chr(10)
MUTANTS = [
    (LIQ, "ideal 5/2 -> 3/2", "(2.5 - np.sum", "(1.5 - np.sum"),
    (LIQ, "S2 missing 1/2", "number_density / 2.0 * np.sum", "number_density * np.sum"),
    (LIQ, "integrand without -g+1", "- g + 1.0", ""),
    (LIQ, "N_a - 1 -> N_a", " - np.diag(counts_per_species)", ""),
    (LIQ, "self distance kept", "keep = d > 1e-8", "keep = d >= 0"),
    (LIQ, "shell r^3 -> r^2", "(edges[1:] ** 3 - edges[:-1] ** 3)", "(edges[1:] ** 2 - edges[:-1] ** 2)"),
    (LIQ, "part-filled last bin", "int(np.floor(r_max / dr + 1e-9)) + 1", "int(np.ceil(r_max / dr)) + 1"),
    (LIQ, "ideal mixing sign", "ideal = float(-GAS_CONSTANT", "ideal = float(GAS_CONSTANT"),
    (LIQ, "ideal gas ignores fraction in log", "np.log(number_density * x[present] * lam", "np.log(number_density * lam"),
    (QC, "s_q Bose term", "x[m] / np.expm1(x[m])", "x[m] / np.exp(x[m])"),
    (QC, "s_cl drops 1", "s_cl[m] = 1.0 - np.log(x[m])", "s_cl[m] = -np.log(x[m])"),
    (QC, "masses ignored", "weight = np.ones(v.shape[1]) if masses_amu is None else", "weight = np.ones(v.shape[1]) if True else"),
    (QC, "dos normalised to 1", "return nu, 3.0 * dos", "return nu, 1.0 * dos"),
    (QC, "h/kB unit THz -> GHz", "scipy.constants.h * 1e12", "scipy.constants.h * 1e9"),
    (QC, "correction sign", "dos * (s_q - s_cl)", "dos * (s_cl - s_q)"),
    (CF, "weight sign", "weight = np.where(is_b, -(1.0 - x_b), x_b)", "weight = np.where(is_b, (1.0 - x_b), x_b)"),
    (CF, "no 1/N", "for x in positions]) / len(weight)", "for x in positions])"),
    (CF, "no 2 pi", "b = 2 * np.pi * np.linalg.inv", "b = np.linalg.inv"),
    (CF, "+k and -k both kept", "grid = grid[first > 0]", "grid = grid[first != 0]"),
    (CF, "fit in k not k^2", "k[use] ** 2], axis=1)", "k[use]], axis=1)"),
    (CF, "curvature f'' sign", "f, f1, f2 = x * (1 - x), 1 - 2 * x, -2.0", "f, f1, f2 = x * (1 - x), 1 - 2 * x, 2.0"),
    (CF, "ideal part not removed", "target = rt * (1.0 / s - 1.0 / (x * (1 - x)))", "target = rt * (1.0 / s)"),
    (CF, "S_ex sign", "(np.asarray(mixing_enthalpies, dtype=float) - g_ex) / temperature", "(g_ex - np.asarray(mixing_enthalpies, dtype=float)) / temperature"),
    (EM, "RK p1 sign", "p1 = -2.0 * k * y ** (k - 1)", "p1 = 2.0 * k * y ** (k - 1)"),
    (EM, "S factor of b", '{"a": np.zeros_like(t), "b": -np.ones_like(t)', '{"a": np.zeros_like(t), "b": np.ones_like(t)'),
    (EM, "H factor of c", '"b": np.zeros_like(t), "c": -t}', '"b": np.zeros_like(t), "c": t}'),
    (EM, "S_cc sigma propagation", "sigma.append(rt * data.scc_error / data.scc**2)", "sigma.append(rt * data.scc_error / data.scc)"),
    (EM, "AICc small-sample term", "/ (n - pk - 1.0)", "/ (n - pk + 1.0)"),
    (EM, "no chi2 inflation", "* max(1.0, chi2_reduced)", "* 1.0"),
    (EM, "curvature target ideal part", "target.append(rt * (1.0 / data.scc - 1.0 / (x * (1 - x))))", "target.append(rt * (1.0 / data.scc))"),
    (EM, "idmix flag dropped", '            "excluded_model_contributions": ["idmix"],' + NL, ""),
    (EM, "per-order candidates lost", "for per_order in itertools.product(temp_sets, repeat=order)]", "for per_order in [[temp_sets[-1]] * order]]"),
    (EM, "undetermined fit gets zero variance", "if np.linalg.matrix_rank(aw) < k:", "if False:"),
    (EM, "component order unchecked", "    if [c.upper() for c in components] != sorted(c.upper() for c in components):", "    if False:"),
    (MD, "offset sign", "compression_entropies, dtype=float) + offset", "compression_entropies, dtype=float) - offset"),
    (MD, "kinetic 3/2 -> 3", "energies = np.asarray(potential_energies, dtype=float) + 1.5", "energies = np.asarray(potential_energies, dtype=float) + 3.0"),
    (MD, "dF/dV sign", "helmholtz_energies_dV=-pressure", "helmholtz_energies_dV=pressure"),
    (MD, "correlation added", "per_mole = per_mole - np.asarray(correlation_entropies", "per_mole = per_mole + np.asarray(correlation_entropies"),
    (MD, "old offset 5.3", "CLASSICAL_ENTROPY_OFFSET = 5.83", "CLASSICAL_ENTROPY_OFFSET = 5.3"),
    (CE, "dither cost 3 ln a -> 2 ln a", "- 3 * np.log(alpha))", "- 2 * np.log(alpha))"),
    (CE, "de Broglie sqrt(2 pi) -> sqrt(pi)", "np.sqrt(2 * np.pi * m * scipy.constants.k", "np.sqrt(np.pi * m * scipy.constants.k"),
    (CE, "step 2/3 -> 1/2", "step = 2.0 / 3.0 * spread[axis]", "step = 0.5 * spread[axis]"),
    (CE, "bits in ln not log2", "return float(-(p * np.log2(p)).sum())", "return float(-(p * np.log(p)).sum())"),
    (CE, "pooled coding by default", "def coefficient_bits(coefficients: np.ndarray, pooled: bool = False)", "def coefficient_bits(coefficients: np.ndarray, pooled: bool = True)"),
    (CE, "NPT reference read in target cell", "    ref_cell = cell if reference_cell is None else np.asarray(reference_cell, dtype=float)" + NL, "    ref_cell = cell" + NL),
    (CE, "infinite loop guard off", "    if n_frames - 1 < min_separation:", "    if False:"),
    (CE, "complex blocks get real Wishart bias", "complex_valued=not np.allclose(chi.imag, 0.0))", "complex_valued=False)"),
    (CE, "centre-of-mass modes dropped", "            log_det += 3 * np.log(independent)  # the three centre-of-mass modes, see above" + NL, ""),
    (CE, "characters not conjugated (expected equivalent)", 'np.einsum("kt,ftd->fkd", characters.conj(), ub)', 'np.einsum("kt,ftd->fkd", characters, ub)'),
    (CE, "Fourier norm sqrt(n) -> n", "/ np.sqrt(n_cells)", "/ n_cells"),
    (CE, "marginal bias dropped", " - n_atoms * _log_det_bias(3, nu)", ""),
    (CE, "nu = frames (expected equivalent)", "    nu = n_frames - 1" + NL + "    ub =", "    nu = n_frames" + NL + "    ub ="),
    (CE, "partial last group kept", "for g in range(n_atoms // group_size)]", "for g in range(-(-n_atoms // group_size))]"),
    (CE, "species ignored in translations", "np.all(species[index] == species) and ", ""),
    (CE, "translation tolerance 0.2 -> 0.02", "tol = 0.2 * _nearest_neighbour_distance", "tol = 0.02 * _nearest_neighbour_distance"),
    (CE, "solid check off", "if rms > max_rms_fraction * nearest:", "if False:"),
    (CE, "per-frame COM kept", "    u = track - sites" + NL + "    u -= u.mean(axis=1, keepdims=True)" + NL, "    u = track - sites" + NL),
    (CE, "centre-of-mass sign", 'entropy": float(shared - centre_of_mass)', 'entropy": float(shared + centre_of_mass)'),
    (CE, "centre-of-mass without e", "(2 * np.pi * np.e * independent)", "(2 * np.pi * independent)"),
    (CE, "volume per atom not total", "np.log(volume * (n_atoms", "np.log(volume / n_atoms * (n_atoms"),
    (CE, "min_groups ignored", "n_pairs = max(n_pairs, -(-min_groups", "n_pairs = max(n_pairs, 0 * (-min_groups"),
    (CE, "gaussian: no kinetic 3/2", "GAS_CONSTANT * (1.5 + 0.5 * (3", "GAS_CONSTANT * (0.0 + 0.5 * (3"),
    (CE, "gaussian: 2 pi without e", "3 * np.log(2 * np.pi * np.e) + log_det", "3 * np.log(2 * np.pi) + log_det"),
    (CE, "gaussian: lambda not squared", "/ lam[:, None, None] ** 2", "/ lam[:, None, None]"),
    (CE, "gaussian: correlation added", 'return {"entropy": marginal - corr["entropy"]', 'return {"entropy": marginal + corr["entropy"]'),
    (CE, "gaussian: no Wishart correction", "log_det = np.linalg.slogdet(cov)[1] - _log_det_bias(3, n_frames - 1)", "log_det = np.linalg.slogdet(cov)[1]"),
    (HC, "k/(k-1) factor lost", "out = out + k / (k - 1) * coefficients[k]", "out = out + coefficients[k]"),
    (HC, "anchor added twice", "anchor_entropy + integral_of_c_over_t(at) - integral_of_c_over_t(anchor_temperature)", "anchor_entropy + integral_of_c_over_t(at)"),
    (HC, "log term from a0", "out = coefficients[1] * np.log(t1)", "out = coefficients[0] * np.log(t1)"),
    (HC, "linear fit only", "np.polynomial.polynomial.polyfit(t, h, degree)", "np.polynomial.polynomial.polyfit(t, h, 1)"),
    (HC, "extrapolation allowed", "if np.any(at < lo - 1e-9) or np.any(at > hi + 1e-9) or not", "if False and not"),
    (FO, "references added", "formation = formation - x * getattr(ref, name)", "formation = formation + x * getattr(ref, name)"),
    (FO, "fractions not normalised", "return {element: n / total for element, n in counts.items()}", "return dict(counts)"),
    (FO, "vacancies counted as atoms", 'if species.upper() != "VA":', "if True:"),
    (FO, "HM and SM swapped", 'QUANTITIES = {"HM": "enthalpy", "SM": "entropy", "CPM": "heat_capacity"}', 'QUANTITIES = {"HM": "entropy", "SM": "enthalpy", "CPM": "heat_capacity"}'),
    (FO, "temperature check off", "if ref.temperatures.shape != phase.temperatures.shape or not np.allclose(ref.temperatures, phase.temperatures):", "if False:"),
    (FO, "values laid out flat", '"values": [[[float(v)] for v in values]],', '"values": [[float(v) for v in values]],'),
]
check = subprocess.run([PY, "-c", "import hermes, sys; print(hermes.__file__)"], cwd=WORK, capture_output=True, text=True)
assert WORK in check.stdout, check.stdout
originals, tally = {}, {"CAUGHT": 0, "SURVIVED": 0, "NOT FOUND": 0}
for path, name, old, new in MUTANTS:
    full = os.path.join(WORK, path)
    text = originals.setdefault(path, open(full, encoding="utf-8", newline="").read())
    if text.count(old) != 1:
        tally["NOT FOUND"] += 1
        print(f"{path.split('/')[1]:24s} {name:46s} PATTERN NOT FOUND ({text.count(old)})", flush=True)
        continue
    open(full, "w", encoding="utf-8", newline="").write(text.replace(old, new))
    try:
        r = subprocess.run([PY, "-m", "pytest", "-o", "addopts=", "-q", "-x", "-p", "no:warnings", "-p", "no:cacheprovider"]
                           + TESTS[path], cwd=WORK, capture_output=True, text=True, timeout=900)
        status = "CAUGHT" if r.returncode else "SURVIVED"
    except subprocess.TimeoutExpired:
        status = "CAUGHT"
    open(full, "w", encoding="utf-8", newline="").write(text)
    tally[status] += 1
    print(f"{path.split('/')[1]:24s} {name:46s} {status}", flush=True)
print(f"{len(MUTANTS)} planted: {tally}")
