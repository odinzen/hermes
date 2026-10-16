"""Entropy at high temperature from an anchor at a moderate one and the heat capacity in between.

The trajectory routes of :mod:`hermes.compression_entropy` treat the shared part of the atoms' motion as Gaussian. On
EAM metals that holds to within about 0.3 J/(K mol) up to about half the melting point and then fails: the correlation
among three or more atoms that anharmonicity brings is not Gaussian, and near melting the entropy reads 0.5 to 1.4 too
high (checked against thermodynamic integration on 36 crystals). Thermodynamics carries an accurate value upward
instead. At constant pressure

    S(T) = S(T0) + int_T0^T C_P / T' dT' = S(T0) + H(T) / T - H(T0) / T0 + int_T0^T H / T'^2 dT',

the second form, by parts, needing only the enthalpy H = <U> + <K> + <P> V of NPT runs at a ladder of temperatures and
no derivative of noisy data. H(T) is smoothed by a least-squares polynomial in T, cubic by default, and the integral
is taken exactly on it. The result is as good as the anchor and the enthalpies, provided nothing melts or changes
phase between T0 and T, which this module cannot see.
"""

# Third-Party Library Imports
import numpy as np


def entropy_from_heat_capacity(
    temperatures, enthalpies, anchor_temperature: float, anchor_entropy: float, at=None, degree: int = 3
) -> dict:
    """
    Entropy at higher temperatures from an anchor value and enthalpies along the way.

    Args:
        temperatures: Temperatures of the NPT runs in K, the anchor temperature among them or below them.
        enthalpies: Mean enthalpy of each run, J/mol of atoms (any constant offset cancels).
        anchor_temperature: T0 in K, where ``anchor_entropy`` is known.
        anchor_entropy: S(T0) in J/(K mol of atoms), for example from the compression route.
        at: Temperatures at which to give S, in K; the run temperatures if omitted.
        degree: Degree of the polynomial in T fitted to H(T).

    Returns:
        Dictionary with ``temperature`` and ``entropy`` (J/(K mol of atoms)), the ``heat_capacity`` C_P = dH/dT of the
        fit there (J/(K mol)), the ``coefficients`` of H(T) (lowest power first) and the rms ``residual`` of the fit.

    Raises:
        ValueError: If there are fewer runs than polynomial terms plus one, or a requested temperature lies outside
            the runs (the polynomial would be extrapolated).
    """
    t = np.asarray(temperatures, dtype=float)
    h = np.asarray(enthalpies, dtype=float)
    if len(t) < degree + 2:
        raise ValueError(f"{len(t)} runs cannot fit and check a degree-{degree} enthalpy curve; give at least {degree + 2}")
    at = t if at is None else np.atleast_1d(np.asarray(at, dtype=float))
    lo, hi = t.min(), t.max()
    if np.any(at < lo - 1e-9) or np.any(at > hi + 1e-9) or not lo - 1e-9 <= anchor_temperature <= hi + 1e-9:
        raise ValueError(f"temperatures must lie within the runs, {lo:g} to {hi:g} K")
    coefficients = np.polynomial.polynomial.polyfit(t, h, degree)
    residual = float(np.sqrt(np.mean((np.polynomial.polynomial.polyval(t, coefficients) - h) ** 2)))

    def integral_of_c_over_t(t1):
        # C = sum_k k a_k T^(k-1), so int C / T dT = a_1 ln T + sum_{k>=2} k / (k - 1) a_k T^(k-1)
        out = coefficients[1] * np.log(t1) if degree >= 1 else 0.0
        for k in range(2, degree + 1):
            out = out + k / (k - 1) * coefficients[k] * t1 ** (k - 1)
        return out

    entropy = anchor_entropy + integral_of_c_over_t(at) - integral_of_c_over_t(anchor_temperature)
    heat_capacity = np.polynomial.polynomial.polyval(at, np.polynomial.polynomial.polyder(coefficients))
    return {"temperature": at, "entropy": entropy, "heat_capacity": heat_capacity, "coefficients": coefficients,
            "residual": residual}
