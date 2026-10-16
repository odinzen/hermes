"""Redlich-Kister excess model of a binary liquid fitted to MD mixing data.

Two kinds of data constrain the interaction parameters L_k(T) = a_k + b_k T (+ c_k T ln T) of

    G_ex = x_A x_B sum_k L_k(T) (x_A - x_B)^k

- mixing enthalpies from the MD energies, H_ex = x_A x_B sum_k (a_k - c_k T)(x_A - x_B)^k, which are
  nearly exact but blind to b_k;
- the long-wavelength concentration fluctuations of :mod:`hermes.concentration_fluctuations`, which
  give the curvature d^2 G_ex / dx^2 = R T (1 / S_cc(0) - 1 / (x_A x_B)) and so a_k + b_k T. They
  are noisy, but they are the only one of the two that carries the excess entropy b_k.

Both are linear in the parameters, so the fit is weighted least squares and, with flat priors and
Gaussian errors, its posterior is Gaussian and exact; no sampling is needed (ESPEI samples with MCMC
because phase-equilibrium likelihoods are not linear). Candidate models and their selection follow
ESPEI (B. Bocklund et al., MRS Communications 9, 618 (2019); MIT): Redlich-Kister orders are grown
successively, each order takes its own successive temperature feature set, every combination is
fitted, and the modified AICc picks the model,

    mAICc = n ln(RSS / n) + 2 p k + (2 p^2 k^2 + 2 p k) / (n - p k - 1),

with k parameters, n data and penalty factor p, as in ``espei.parameter_selection.selection``.
Uncertainties are inflated by the reduced chi-square when it exceeds one.

x is the mole fraction of B, the second component in alphabetical order, so (x_A - x_B) = 1 - 2x
as in a TDB file. Energies in J/mol of atoms.
"""

# Standard Library Imports
import itertools
import warnings
from dataclasses import dataclass, field

# Third-Party Library Imports
import numpy as np

# Local Imports
from hermes.compression_entropy import GAS_CONSTANT

TEMPERATURE_FEATURES = ("a", "b", "c")  # 1, T, T ln T


def _rk_basis(x: np.ndarray, k: int, derivative: int = 0) -> np.ndarray:
    """x(1-x)(1-2x)^k or its first or second derivative in x."""
    y = 1.0 - 2.0 * x
    f, f1, f2 = x * (1 - x), 1 - 2 * x, -2.0 * np.ones_like(x)
    p = y**k
    p1 = -2.0 * k * y ** (k - 1) if k >= 1 else np.zeros_like(x)
    p2 = 4.0 * k * (k - 1) * y ** (k - 2) if k >= 2 else np.zeros_like(x)
    return (f * p, f1 * p + f * p1, f2 * p + 2 * f1 * p1 + f * p2)[derivative]


def _temperature_factor(feature: str, t: np.ndarray, quantity: str) -> np.ndarray:
    """Multiplier of a_k, b_k or c_k in G, H = G - T dG/dT or S = -dG/dT."""
    if quantity == "G":
        return {"a": np.ones_like(t), "b": t, "c": t * np.log(t)}[feature]
    if quantity == "H":
        return {"a": np.ones_like(t), "b": np.zeros_like(t), "c": -t}[feature]
    if quantity == "S":
        return {"a": np.zeros_like(t), "b": -np.ones_like(t), "c": -(np.log(t) + 1.0)}[feature]
    raise ValueError(quantity)


@dataclass
class MixingData:
    """Mixing enthalpies and S_cc(0) values with their standard errors, at (x, T) points."""

    enthalpy_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    enthalpy_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    enthalpy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    enthalpy_error: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scc_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scc_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scc_error: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            setattr(self, name, np.atleast_1d(np.asarray(getattr(self, name), dtype=float)))


@dataclass
class ExcessModel:
    """A fitted Redlich-Kister excess model with the Gaussian posterior of its parameters."""

    features: list  # [(k, "a" | "b" | "c"), ...]
    coefficients: np.ndarray
    covariance: np.ndarray
    score: float
    chi2_reduced: float

    def _row(self, x, t, quantity):
        x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
        return np.stack([_rk_basis(x, k) * _temperature_factor(f, t, quantity) for k, f in self.features], axis=-1)

    def _evaluate(self, x, t, quantity):
        rows = self._row(x, t, quantity)
        value = rows @ self.coefficients
        error = np.sqrt(np.einsum("...i,ij,...j->...", rows, self.covariance, rows))
        return value, error

    def gibbs(self, x, t):
        """Excess Gibbs energy and its standard error, J/mol."""
        return self._evaluate(x, t, "G")

    def enthalpy(self, x, t):
        """Excess (mixing) enthalpy and its standard error, J/mol."""
        return self._evaluate(x, t, "H")

    def entropy(self, x, t):
        """Excess mixing entropy and its standard error, J/(K mol)."""
        return self._evaluate(x, t, "S")

    def interaction_parameters(self) -> dict:
        """{k: {"a": ..., "b": ..., "c": ...}} for the Redlich-Kister L_k."""
        out = {}
        for (k, f), value in zip(self.features, self.coefficients):
            out.setdefault(k, {})[f] = float(value)
        return out

    def tdb_parameters(self, phase: str, component_a: str, component_b: str) -> list:
        """PARAMETER lines for a TDB file (298.15 to 6000 K); components in alphabetical order."""
        _check_alphabetical((component_a, component_b))
        names = {"a": "", "b": "*T", "c": "*T*LN(T)"}
        lines = []
        for k, terms in sorted(self.interaction_parameters().items()):
            expr = "".join(f"{v:+.6g}{names[f]}" for f, v in terms.items())
            lines.append(f"PARAMETER G({phase},{component_a},{component_b};{k}) 298.15 {expr}; 6000 N !")
        return lines


def _check_alphabetical(components):
    # pycalphad sorts a parameter's constituents, so G(LIQ,ZR,CU;1) is read as L1 (x_CU - x_ZR); with the
    # components the other way round every odd term would change sign without notice
    if [c.upper() for c in components] != sorted(c.upper() for c in components):
        raise ValueError(f"give the components in alphabetical order, x being the second: {sorted(components)}")


def _design(data: MixingData, features: list):
    rows, target, sigma = [], [], []
    if data.enthalpy.size:
        rows.append(np.stack([_rk_basis(data.enthalpy_x, k) * _temperature_factor(f, data.enthalpy_t, "H")
                              for k, f in features], axis=1))
        target.append(data.enthalpy)
        sigma.append(data.enthalpy_error)
    if data.scc.size:
        x, t = data.scc_x, data.scc_t
        rt = GAS_CONSTANT * t
        rows.append(np.stack([_rk_basis(x, k, derivative=2) * _temperature_factor(f, t, "G")
                              for k, f in features], axis=1))
        target.append(rt * (1.0 / data.scc - 1.0 / (x * (1 - x))))
        sigma.append(rt * data.scc_error / data.scc**2)
    return np.concatenate(rows), np.concatenate(target), np.concatenate(sigma)


def modified_aicc(rss: float, n: int, k: int, penalty: float = 1.0) -> float:
    """ESPEI's modified AICc of a least-squares fit with residual sum of squares ``rss``."""
    rss = max(rss, 1e-16)
    pk = penalty * k
    aic = n * np.log(rss / n) + 2.0 * pk
    if pk >= n - 1.0:
        return aic + (2.0 * penalty**2 * k**2 + 2.0 * pk) * (-n + pk + 3.0)
    return aic + (2.0 * penalty**2 * k**2 + 2.0 * pk) / (n - pk - 1.0)


def candidate_feature_sets(max_order: int = 3, temperature_features=("a", "b"), candidate_limit: int = 1000) -> list:
    """
    ESPEI's candidates (``espei.parameter_selection.model_building.build_candidate_models``): for
    successive RK orders L0, L0..L1, ..., every order takes its own successive temperature set, so
    L0 = a + bT with L1 = a is a candidate. Past ``candidate_limit`` ESPEI falls back to one shared
    temperature set for all orders, and so does this.
    """
    temp_sets = [list(temperature_features[: i + 1]) for i in range(len(temperature_features))]
    n, m = len(temp_sets), max_order
    count = m if n == 1 else n * (1 - n**m) // (1 - n)
    if n > 1 and count > candidate_limit:
        return [[(k, f) for k in range(order) for f in temps]
                for temps, order in itertools.product(temp_sets, range(1, max_order + 1))]
    return [[(k, f) for k, temps in enumerate(per_order) for f in temps]
            for order in range(1, max_order + 1)
            for per_order in itertools.product(temp_sets, repeat=order)]


def fit_excess_model(data: MixingData, features: list) -> ExcessModel:
    """Weighted least-squares fit of one candidate model."""
    a, y, sigma = _design(data, features)
    w = 1.0 / sigma
    aw, yw = a * w[:, None], y * w
    coefficients, *_ = np.linalg.lstsq(aw, yw, rcond=None)
    rss = float(np.sum((aw @ coefficients - yw) ** 2))
    n, k = len(y), len(features)
    chi2_reduced = rss / (n - k) if n > k else float("inf")
    if np.linalg.matrix_rank(aw) < k:
        # pinv would report zero variance along the direction the data cannot see
        warnings.warn(f"the data do not determine all of {features}; their uncertainties are reported as NaN")
        covariance = np.full((k, k), np.nan)
    else:
        covariance = np.linalg.inv(aw.T @ aw) * max(1.0, chi2_reduced)
    return ExcessModel(features, coefficients, covariance, modified_aicc(rss, n, k), chi2_reduced)


def select_excess_model(data: MixingData, max_order: int = 3, temperature_features=("a", "b"),
                        penalty: float = 1.0) -> tuple:
    """Fit every candidate and return ``(best, all_fits)``, best by modified AICc."""
    fits = []
    for features in candidate_feature_sets(max_order, temperature_features):
        model = fit_excess_model(data, features)
        if penalty != 1.0:
            a, y, sigma = _design(data, features)
            # a candidate with as many parameters as data has rss 0, not inf * 0
            rss = model.chi2_reduced * (len(y) - len(features)) if len(y) > len(features) else 0.0
            model.score = modified_aicc(rss, len(y), len(features), penalty)
        fits.append(model)
    return min(fits, key=lambda m: m.score), fits


def espei_datasets(x, t, values, output: str, components=("A", "B"), phase: str = "LIQUID",
                   reference: str = "Hermes", comment: str = "") -> list:
    """
    ESPEI non-equilibrium datasets, one per temperature, for ``output`` "HM_MIX" or "SM_MIX".

    Pass excess values, the MD mixing enthalpy as it is and the excess entropy (not the total mixing
    entropy). ESPEI by default subtracts the whole fixed model from _MIX data, ideal mixing included,
    so every dataset carries ``"excluded_model_contributions": ["idmix"]``. Without it an excess
    SM_MIX is fitted as if it were the total and the model comes out short by the ideal term. The
    ideal term adds nothing to HM_MIX, so there the flag changes no number; it is kept so every
    dataset states the same thing about its values. Components are written in the given order; x is
    the fraction of the second.
    """
    x, t, values = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (x, t, values))
    _check_alphabetical(components)
    pair = [c.upper() for c in components]
    out = []
    for temperature in np.unique(t):
        sel = t == temperature
        out.append({
            "components": pair,
            "phases": [phase],
            "solver": {
                "mode": "manual",
                "sublattice_site_ratios": [1],
                "sublattice_configurations": [[pair]] * int(sel.sum()),
                "sublattice_occupancies": [[[float(1 - xi), float(xi)]] for xi in x[sel]],
            },
            "conditions": {"P": 101325, "T": float(temperature)},
            "output": output,
            "values": [[[float(v) for v in values[sel]]]],
            "excluded_model_contributions": ["idmix"],
            "reference": reference,
            "comment": comment,
        })
    return out
