"""Builder for creating validated :class:`GasData` instances.

Provides a fluent interface to set gas species fields with optional unit
conversion and automatic batch dimension handling.

Examples:
    Single-box with unit conversion::

        import numpy as np
        from particula.gas import GasDataBuilder

        gas = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia", "H2SO4"])
            .set_molar_mass([18.015, 17.031, 98.079], units="g/mol")
            .set_concentration([1e9, 1e6, 1e4], units="1/cm^3")
            .set_partitioning([True, True, True])
            .build()
        )

        print(gas.molar_mass)
        # [0.018015 0.017031 0.098079]
        print(gas.concentration.shape)
        # (1, 3)

    Multi-box with broadcasting::

        gas = (
            GasDataBuilder()
            .set_n_boxes(100)
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017], units="kg/mol")
            .set_concentration([1e15, 1e12], units="1/m^3")
            .set_partitioning([True, True])
            .build()
        )

        print(gas.concentration.shape)
        # (100, 2)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from particula.gas.gas_data import GasData
from particula.util.convert_units import get_unit_conversion


class GasDataBuilder:
    """Fluent builder that prepares arrays for :class:`GasData`.

    Each setter performs unit conversion and ensures correct dtypes. Batch
    dimensions are inserted automatically when 1D concentration is provided.
    Broadcasting to ``n_boxes`` is supported when set via ``set_n_boxes``.

    Supported units:
        - molar_mass: ``kg/mol`` (default), ``g/mol``
        - concentration: ``1/m^3`` (default), ``1/cm^3``

    Validation:
        - names must be non-empty
        - molar_mass must be 1D and positive
        - concentration must be 1D or 2D and non-negative
        - partitioning must be 1D and boolean-convertible
        - shapes must align with ``len(names)``
    """

    def __init__(self) -> None:
        """Initialize empty builder state."""
        self._names: Optional[list[str]] = None
        self._molar_mass: Optional[NDArray[np.float64]] = None
        self._concentration: Optional[NDArray[np.float64]] = None
        self._partitioning: Optional[NDArray[np.bool_]] = None
        self._n_boxes: Optional[int] = None

    def set_names(self, names: list[str]) -> "GasDataBuilder":
        """Set species names.

        Args:
            names: List of species names. Must be non-empty.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: If names list is empty.
        """
        if len(names) == 0:
            raise ValueError("names must contain at least one species")
        self._names = list(names)
        return self

    def set_molar_mass(
        self,
        molar_mass: Union[list[float], NDArray[np.float64]],
        units: str = "kg/mol",
    ) -> "GasDataBuilder":
        """Set molar masses with optional unit conversion.

        Args:
            molar_mass: Values shaped (n_species,).
            units: Units of provided values. Supported: ``kg/mol`` (default),
                ``g/mol``.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: If dimension is not 1D or any value is non-positive.
        """
        molar_mass_array = np.asarray(molar_mass, dtype=np.float64)
        if molar_mass_array.ndim != 1:
            raise ValueError("molar_mass must be 1D")

        if units != "kg/mol":
            molar_mass_array = molar_mass_array * get_unit_conversion(
                units, "kg/mol"
            )

        if np.any(molar_mass_array <= 0):
            raise ValueError("molar_mass must be positive")

        self._molar_mass = molar_mass_array
        return self

    def set_concentration(
        self,
        concentration: Union[list[float], NDArray[np.float64]],
        units: str = "1/m^3",
    ) -> "GasDataBuilder":
        """Set concentrations with unit conversion and auto batch dimension.

        Args:
            concentration: Concentration values. If 1D (n_species,), batch
                dimension is added. If 2D (n_boxes, n_species), used as-is.
            units: Units of provided concentration. Supported: ``1/m^3``
                (default), ``1/cm^3``.

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: If dimension is not 1D or 2D, or any value is negative.
        """
        concentration_array = np.asarray(concentration, dtype=np.float64)

        if concentration_array.ndim == 1:
            concentration_array = np.expand_dims(concentration_array, axis=0)
        elif concentration_array.ndim != 2:
            raise ValueError("concentration must be 1D or 2D")

        if units != "1/m^3":
            concentration_array = concentration_array * get_unit_conversion(
                units, "1/m^3"
            )

        if np.any(concentration_array < 0):
            raise ValueError("concentration must be non-negative")

        self._concentration = concentration_array
        return self

    def set_partitioning(
        self, partitioning: Union[list[bool], NDArray[np.bool_]]
    ) -> "GasDataBuilder":
        """Set species partitioning mask.

        Args:
            partitioning: Boolean-convertible values shaped (n_species,).

        Returns:
            Self for fluent chaining.

        Raises:
            ValueError: If array is not 1D or not boolean-convertible.
        """
        partitioning_array = np.asarray(partitioning, dtype=np.bool_)
        if partitioning_array.ndim != 1:
            raise ValueError("partitioning must be 1D")
        self._partitioning = partitioning_array
        return self

    def set_n_boxes(self, n_boxes: int) -> "GasDataBuilder":
        """Set number of boxes for optional broadcasting.

        Args:
            n_boxes: Target number of boxes when broadcasting 1D concentration.

        Returns:
            Self for fluent chaining.
        """
        self._n_boxes = int(n_boxes)
        return self

    def _broadcast_concentration(self) -> NDArray[np.float64]:
        """Broadcast concentration to requested n_boxes when applicable."""
        if self._concentration is None:
            raise ValueError("concentration must be set before broadcasting")
        if self._n_boxes is None:
            return self._concentration
        if self._concentration.shape[0] == 1:
            return np.broadcast_to(
                self._concentration,
                (self._n_boxes, self._concentration.shape[1]),
            ).copy()
        if self._concentration.shape[0] != self._n_boxes:
            raise ValueError(
                "concentration row count does not match requested n_boxes"
            )
        return self._concentration

    def build(self) -> GasData:
        """Construct a :class:`GasData` instance with validation.

        Returns:
            A validated ``GasData`` object.

        Raises:
            ValueError: If required fields are missing or shapes mismatch.
        """
        if self._names is None:
            raise ValueError("names is required")
        if self._molar_mass is None:
            raise ValueError("molar_mass is required")
        if self._concentration is None:
            raise ValueError("concentration is required")
        if self._partitioning is None:
            raise ValueError("partitioning is required")

        n_species = len(self._names)
        if self._molar_mass.shape != (n_species,):
            raise ValueError("molar_mass length must match names")
        if self._partitioning.shape != (n_species,):
            raise ValueError("partitioning length must match names")
        if self._concentration.shape[1] != n_species:
            raise ValueError("concentration width must match names")

        concentration_broadcasted = self._broadcast_concentration()

        return GasData(
            name=self._names,
            molar_mass=self._molar_mass,
            concentration=concentration_broadcasted,
            partitioning=self._partitioning,
        )
