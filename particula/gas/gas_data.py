"""Provide a batched gas data container for multi-box CFD simulations.

GasData isolates gas species arrays from behavior (vapor pressure strategies)
while embedding the batch dimension required for CFD experiments spanning
multiple boxes.

Example:
    Single-box simulation (n_boxes=1)::

        from particula.gas.gas_data import GasData
        import numpy as np

        data = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e15, 1e12, 1e10]]),  # (1, 3)
            partitioning=np.array([True, True, True]),
        )

    Multi-box CFD simulation (100 boxes)::

        cfd_data = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.zeros((100, 3)),  # (100, 3)
            partitioning=np.array([True, True, True]),
        )
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import VaporPressureStrategy
from particula.util.constants import AVOGADRO_NUMBER


@dataclass
class GasData:
    """Batched gas species data container for multi-box simulations.

    Simple data container with batch dimension built-in. Concentration
    arrays have shape (n_boxes, n_species) to support multi-box CFD.
    Single-box simulations use n_boxes=1.

    This is NOT a frozen dataclass - concentrations can be updated in place
    for performance. Use copy() if immutability is needed.

    Attributes:
        name: Species names. List of strings, length n_species.
        molar_mass: Molar masses in kg/mol. Shape: (n_species,)
        concentration: Number concentrations in molecules/m^3.
            Shape: (n_boxes, n_species)
        partitioning: Whether each species can partition to particles.
            Shape: (n_species,) - shared across boxes

    Raises:
        ValueError: If array shapes are inconsistent or species list is empty.
    """

    name: list[str]
    molar_mass: NDArray[np.float64]
    concentration: NDArray[np.float64]
    partitioning: NDArray[np.bool_]

    def __post_init__(self) -> None:
        """Validate array shapes are consistent and enforce boolean mask."""
        # Reject empty species set to avoid ambiguous shapes
        if len(self.name) == 0:
            raise ValueError("name must contain at least one species")

        # Normalize arrays to expected dtypes
        self.molar_mass = np.asarray(self.molar_mass, dtype=np.float64)
        self.concentration = np.asarray(self.concentration, dtype=np.float64)

        # Ensure partitioning is boolean or raise if conversion fails
        try:
            self.partitioning = np.asarray(self.partitioning, dtype=np.bool_)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "partitioning must be boolean convertible"
            ) from exc

        n_species = len(self.name)

        # concentration must be 2D (n_boxes, n_species)
        if self.concentration.ndim != 2:
            raise ValueError(
                "concentration must be 2D (n_boxes, n_species), "
                f"got ndim={self.concentration.ndim}"
            )

        # concentration width must match number of species
        if self.concentration.shape[1] != n_species:
            raise ValueError(
                "concentration n_species dimension does not match name list: "
                f"got {self.concentration.shape[1]}, expected {n_species}"
            )

        # molar_mass shape must align with n_species
        if self.molar_mass.shape != (n_species,):
            raise ValueError(
                "molar_mass shape does not match n_species: "
                f"got {self.molar_mass.shape}, expected ({n_species},)"
            )

        # partitioning must be 1D boolean with n_species elements
        if self.partitioning.shape != (n_species,):
            raise ValueError(
                "partitioning shape does not match n_species: "
                f"got {self.partitioning.shape}, expected ({n_species},)"
            )

    @property
    def n_boxes(self) -> int:
        """Number of simulation boxes (batch dimension)."""
        return int(self.concentration.shape[0])

    @property
    def n_species(self) -> int:
        """Number of gas species."""
        return len(self.name)

    def copy(self) -> "GasData":
        """Create a deep copy of this GasData."""
        return GasData(
            name=list(self.name),
            molar_mass=np.copy(self.molar_mass),
            concentration=np.copy(self.concentration),
            partitioning=np.copy(self.partitioning),
        )


def from_species(species: GasSpecies, n_boxes: int = 1) -> GasData:
    """Create a GasData instance from a GasSpecies object.

    Concentration is converted from kg/m^3 (GasSpecies) to molecules/m^3
    (GasData) using Avogadro's number and molar mass. Supports scalar or
    array inputs for names, molar masses, concentrations, and partitioning.

    Args:
        species: Source GasSpecies instance (single or multi-species).
        n_boxes: Number of boxes to replicate concentration into.

    Returns:
        GasData with concentration shaped (n_boxes, n_species).

    Raises:
        ValueError: If n_boxes < 1.
    """
    if n_boxes < 1:
        raise ValueError("n_boxes must be at least 1")

    names = species.get_name()
    names_list = [str(names)] if isinstance(names, str) else list(names)

    molar_mass = np.asarray(species.get_molar_mass(), dtype=np.float64)
    molar_mass = np.atleast_1d(molar_mass)

    mass_conc = np.asarray(species.get_concentration(), dtype=np.float64)
    mass_conc = np.atleast_1d(mass_conc)

    number_conc = (mass_conc / molar_mass) * AVOGADRO_NUMBER
    concentration = np.tile(number_conc, (n_boxes, 1))

    partitioning_raw = np.asarray(species.get_partitioning(), dtype=np.bool_)
    partitioning = np.atleast_1d(partitioning_raw)
    if partitioning.size == 1 and molar_mass.size > 1:
        partitioning = np.full(
            molar_mass.shape, partitioning.item(), dtype=bool
        )
    elif partitioning.shape != molar_mass.shape:
        partitioning = np.asarray(partitioning, dtype=np.bool_).reshape(
            molar_mass.shape
        )

    return GasData(
        name=names_list,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


def to_species(
    data: GasData,
    vapor_pressure_strategies: Sequence[VaporPressureStrategy],
    box_index: int = 0,
) -> GasSpecies:
    """Create a GasSpecies instance from a GasData object.

    Converts concentration from molecules/m^3 (GasData) to kg/m^3
    (GasSpecies) using molar mass and Avogadro's number. Requires a
    vapor pressure strategy for each species because GasData is
    behavior-free.

    Args:
        data: Source GasData instance.
        vapor_pressure_strategies: Strategy per species.
        box_index: Row index to extract concentrations from.

    Returns:
        GasSpecies built from the selected box.

    Raises:
        ValueError: If strategies length != n_species or partitioning differs
            across species.
        IndexError: If box_index is out of range.
    """
    if len(vapor_pressure_strategies) != data.n_species:
        raise ValueError(
            "vapor_pressure_strategies length must match data.n_species"
        )

    if box_index < 0 or box_index >= data.n_boxes:
        raise IndexError("box_index out of range")

    if not np.all(data.partitioning == data.partitioning[0]):
        raise ValueError("GasData partitioning must be uniform to convert")

    partitioning_flag = bool(data.partitioning[0])

    concentration_row = data.concentration[box_index]
    concentration_mass = (concentration_row * data.molar_mass) / AVOGADRO_NUMBER

    names = [str(name) for name in data.name]
    partitioning_flag = bool(data.partitioning[0])

    base_species = GasSpecies(
        name=names[0],
        molar_mass=data.molar_mass[0],
        vapor_pressure_strategy=vapor_pressure_strategies[0],
        partitioning=partitioning_flag,
        concentration=concentration_mass[0],
    )

    for idx in range(1, data.n_species):
        next_species = GasSpecies(
            name=names[idx],
            molar_mass=data.molar_mass[idx],
            vapor_pressure_strategy=vapor_pressure_strategies[idx],
            partitioning=partitioning_flag,
            concentration=concentration_mass[idx],
        )
        base_species += next_species

    return base_species
