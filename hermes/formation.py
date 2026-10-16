"""Formation enthalpy, entropy and heat capacity of a phase, written as ESPEI datasets.

ESPEI fits a phase's Gibbs energy to HM_FORM, SM_FORM and CPM_FORM data: per mole of atoms, against the pure elements
in their reference states (ESPEI sets the GHSER functions to zero for the fit), with pycalphad then using the fitted
database. Hermes's entropies and enthalpies are absolute and belong to one interatomic potential and one route. A
formation value is therefore taken against the elements computed the same way, the same potential, route and
temperatures, as an atom-balanced difference; taken against laboratory (SGTE) references instead, the potential's own
error in each element would land in the formation value.

The phase is an endmember: one species on each sublattice, vacancies allowed. For a pure element in another lattice
the "formation" values are its lattice stability.
"""

# Standard Library Imports
from dataclasses import dataclass

# Third-Party Library Imports
import numpy as np


@dataclass
class PhaseProperties:
    """Enthalpy, entropy and heat capacity of one phase at a set of temperatures, per mole of atoms."""

    temperatures: np.ndarray  # K
    enthalpy: np.ndarray = None  # J/mol
    entropy: np.ndarray = None  # J/(K mol)
    heat_capacity: np.ndarray = None  # J/(K mol)

    def __post_init__(self):
        self.temperatures = np.atleast_1d(np.asarray(self.temperatures, dtype=float))
        for name in ("enthalpy", "entropy", "heat_capacity"):
            value = getattr(self, name)
            if value is not None:
                value = np.atleast_1d(np.asarray(value, dtype=float))
                if value.shape != self.temperatures.shape:
                    raise ValueError(f"{name} has shape {value.shape}, the temperatures {self.temperatures.shape}")
                setattr(self, name, value)


QUANTITIES = {"HM": "enthalpy", "SM": "entropy", "CPM": "heat_capacity"}


def atom_fractions(configuration, site_ratios) -> dict:
    """Mole fraction of each element in an endmember, e.g. (["AL", "NI"], [0.75, 0.25]) -> {"AL": 0.75, "NI": 0.25}."""
    if len(configuration) != len(site_ratios):
        raise ValueError("one species per sublattice, and one site ratio per sublattice")
    counts = {}
    for species, ratio in zip(configuration, site_ratios):
        if not isinstance(species, str):
            raise ValueError(f"sublattice holds {species}; formation data are for endmembers, one species per sublattice")
        if species.upper() != "VA":
            counts[species.upper()] = counts.get(species.upper(), 0.0) + float(ratio)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("the configuration holds no atoms")
    return {element: n / total for element, n in counts.items()}


def formation_properties(phase: PhaseProperties, references: dict, configuration, site_ratios) -> PhaseProperties:
    """
    Phase properties less the atom-weighted properties of its elements in their reference phases.

    Args:
        phase: The phase, from Hermes.
        references: {element: PhaseProperties} for every element of the phase, in its reference phase, from the same
            potential and route and at the same temperatures.
        configuration: Species on each sublattice, e.g. ``["CU", "ZR"]``.
        site_ratios: Sites of each sublattice.

    Returns:
        PhaseProperties of formation, with the quantities given for the phase and every reference.

    Raises:
        ValueError: If a reference is missing, is at other temperatures, or lacks a quantity the phase has.
    """
    fractions = atom_fractions(configuration, site_ratios)
    out = {}
    for name in QUANTITIES.values():
        value = getattr(phase, name)
        if value is None:
            continue
        formation = value.copy()
        for element, x in fractions.items():
            if element not in {k.upper() for k in references}:
                raise ValueError(f"no reference properties for {element}")
            ref = {k.upper(): v for k, v in references.items()}[element]
            if ref.temperatures.shape != phase.temperatures.shape or not np.allclose(ref.temperatures, phase.temperatures):
                raise ValueError(f"the {element} reference is at other temperatures than the phase")
            if getattr(ref, name) is None:
                raise ValueError(f"the {element} reference has no {name}")
            formation = formation - x * getattr(ref, name)
        out[name] = formation
    return PhaseProperties(phase.temperatures, **out)


def espei_formation_datasets(
    phase_name: str, configuration, site_ratios, phase: PhaseProperties, references: dict,
    reference: str = "Hermes", comment: str = "",
) -> list:
    """
    ESPEI non-equilibrium datasets of HM_FORM, SM_FORM and CPM_FORM for an endmember of a phase.

    One dataset per quantity the phase and its references carry, all temperatures in it, at 101325 Pa. Load them with
    ``espei.datasets.load_datasets`` after writing each as JSON, or check them with ``espei.datasets.check_dataset``.
    Fit with ``generate_parameters(..., ridge_alpha=1e-100)``: with ESPEI's default (None) scikit-learn's
    LinearRegression (version 1.9) loses the badly scaled 1/T term of the heat capacity, and the low-temperature
    heat capacity of the fit comes out wrong without any warning.

    Args:
        phase_name: Phase name as in the ESPEI phase models, e.g. ``"BCC_A2"``.
        configuration: Species on each sublattice of the endmember, e.g. ``["CU", "ZR"]``.
        site_ratios: Sites of each sublattice, as in the phase models.
        phase: The phase's properties from Hermes.
        references: {element: PhaseProperties} of the elements in their reference phases, computed the same way.
        reference: Data reference written into each dataset.
        comment: Comment written into each dataset.

    Returns:
        List of dataset dicts.
    """
    formation = formation_properties(phase, references, configuration, site_ratios)
    species = sorted({s.upper() for s in configuration})
    datasets = []
    for output, name in QUANTITIES.items():
        values = getattr(formation, name)
        if values is None:
            continue
        datasets.append({
            "components": species,
            "phases": [phase_name],
            "solver": {
                "mode": "manual",
                "sublattice_site_ratios": [float(r) for r in site_ratios],
                "sublattice_configurations": [[s.upper() for s in configuration]],
            },
            "conditions": {"P": 101325, "T": [float(t) for t in formation.temperatures]},
            "output": f"{output}_FORM",
            "values": [[[float(v)] for v in values]],
            "reference": reference,
            "comment": comment or "formation against the elements computed with the same potential and route",
        })
    return datasets
