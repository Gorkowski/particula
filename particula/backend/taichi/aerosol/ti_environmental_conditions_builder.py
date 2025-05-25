"""
Dedicated builder for the frozen dataclass
"""

from dataclasses import dataclass
from typing import Dict, Any


# ---------------------------------------------------------------------
# 1 · Original frozen dataclass
# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EnvironmentalConditions:
    temperature: float = 298.15
    pressure: float = 101_325.0
    mass_accommodation: float = 0.5
    dynamic_viscosity: float = 1.8e-5
    diffusion_coefficient: float = 2.0e-5
    time_step: float = 10.0
    simulation_volume: float = 1.0e-6


# ---------------------------------------------------------------------
# 2 · Dedicated builder
# ---------------------------------------------------------------------
class EnvironmentalConditionsBuilder:
    """
    Fluent builder for :class:`EnvironmentalConditions`.

    Example
    -------
    >>> env = (
    ...     EnvironmentalConditionsBuilder()
    ...     .temperature(310.0)
    ...     .pressure(90_000.0)
    ...     .dynamic_viscosity(2.1e-5)
    ...     .build()
    ... )
    >>> env
    EnvironmentalConditions(temperature=310.0, pressure=90000.0, ...)
    """

    def __init__(self) -> None:
        # Store only values the user overrides; defaults come from the dataclass
        self._overrides: Dict[str, Any] = {}

    # --------- fluent setters ----------------------------------------
    def temperature(self, value: float) -> "EnvironmentalConditionsBuilder":
        self._overrides["temperature"] = value
        return self

    def pressure(self, value: float) -> "EnvironmentalConditionsBuilder":
        self._overrides["pressure"] = value
        return self

    def mass_accommodation(
        self, value: float
    ) -> "EnvironmentalConditionsBuilder":
        self._overrides["mass_accommodation"] = value
        return self

    def dynamic_viscosity(
        self, value: float
    ) -> "EnvironmentalConditionsBuilder":
        self._overrides["dynamic_viscosity"] = value
        return self

    def diffusion_coefficient(
        self, value: float
    ) -> "EnvironmentalConditionsBuilder":
        self._overrides["diffusion_coefficient"] = value
        return self

    def time_step(self, value: float) -> "EnvironmentalConditionsBuilder":
        self._overrides["time_step"] = value
        return self

    def simulation_volume(
        self, value: float
    ) -> "EnvironmentalConditionsBuilder":
        self._overrides["simulation_volume"] = value
        return self

    # --------- finalise ---------------------------------------------
    def build(self) -> EnvironmentalConditions:
        """
        Create a frozen :class:`EnvironmentalConditions` instance,
        using defaults for any fields you did not override.
        """
        return EnvironmentalConditions(**self._overrides)
