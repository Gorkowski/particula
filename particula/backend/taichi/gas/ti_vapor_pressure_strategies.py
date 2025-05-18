"""
Taichi vapor pressure strategies for particula: fast, differentiable kernels.

This module provides Taichi-accelerated strategies for computing vapor
pressure and related quantities, used in gas-phase and multiphase
aerosol modeling. It includes implementations for constant, Antoine,
Clausius-Clapeyron, and Buck (water) vapor pressure laws.

Constants:
    - GAS_CONSTANT : Universal gas constant (J/(mol·K)).

Classes:
    - _VaporPressureMixin : Internal base for vapor pressure strategies.
    - ConstantVaporPressureStrategy : Constant vapor pressure.
    - AntoineVaporPressureStrategy : Antoine equation vapor pressure.
    - ClausiusClapeyronStrategy : Clausius-Clapeyron vapor pressure.
    - WaterBuckStrategy : Buck equation for water vapor pressure.

Examples:
    ```py title="Minimal Usage"
    from particula.backend.dispatch_register import get
    strat = get("ConstantVaporPressureStrategy", backend="taichi")(42.0)
    vp = strat.pure_vapor_pressure(300.0)
    print(vp)  # Output: 42.0
    ```

References:
    - Buck, A. L. "New equations for computing vapor pressure and enhancement
      factor." J. Appl. Meteorol. 20, 1527–1532 (1981).
    - "Vapor pressure", Wikipedia. https://en.wikipedia.org/wiki/Vapor_pressure
"""

import taichi as ti

# Add missing import for register
from particula.backend.taichi.gas.properties import (
    fget_partial_pressure,
    fget_concentration_from_pressure,
    fget_antoine_vapor_pressure,
    fget_buck_vapor_pressure,
    fget_clausius_clapeyron_vapor_pressure,
)
from particula.backend.dispatch_register import register


GAS_CONSTANT = 8.31446261815324  # J/(mol·K)


@ti.data_oriented
class _VaporPressureMixin:
    """
    (Internal) Shared Taichi helpers for vapor pressure strategies.

    Provides common kernels and Python wrappers for vapor pressure,
    partial pressure, and concentration calculations. Not intended for
    direct use; inherit for custom vapor pressure strategies.

    Attributes:
        (None; this is a mixin.)

    Inherited Public Methods:
        - pure_vapor_pressure: Compute pure vapor pressure.
        - partial_pressure: Compute partial pressure.
        - concentration: Compute concentration from pressure.
        - saturation_ratio: Compute saturation ratio.
        - saturation_concentration: Compute saturation concentration.

    Examples:
        ```py title="Mixin Usage"
        class MyStrategy(_VaporPressureMixin): ...
        ```

    References:
        - See module docstring.
    """

    # ── pure-vapor-pressure ────────────────────────────────────────────────
    @ti.kernel
    def kget_pure_vapor_pressure(self, temperature: ti.f64) -> ti.f64:
        """
        Kernel: compute pure vapor pressure at given temperature.

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - Pure vapor pressure [Pa].

        Examples:
            strat.kget_pure_vapor_pressure(300.0)
        """
        return self.fget_pure_vapor_pressure(temperature)  # subclass provides it

    # ── partial pressure ---------------------------------------------------
    @ti.func
    def fget_partial_pressure_internal(
        self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        """
        Compute partial pressure from concentration, molar mass, temperature.

        Arguments:
            - concentration : Concentration [kg/m³].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Partial pressure [Pa].

        Examples:
            self.fget_partial_pressure_internal(1.0, 0.018, 300.0)
        """
        return fget_partial_pressure(concentration, molar_mass, temperature)

    @ti.kernel
    def kget_partial_pressure(
        self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        """
        Kernel: compute partial pressure from concentration, molar mass, T.

        Arguments:
            - concentration : Concentration [kg/m³].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Partial pressure [Pa].

        Examples:
            strat.kget_partial_pressure(1.0, 0.018, 300.0)
        """
        return self.fget_partial_pressure_internal(
            concentration, molar_mass, temperature
        )

    # ── concentration from pressure ---------------------------------------
    @ti.func
    def fget_concentration_from_pressure_internal(
        self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        """
        Compute concentration from partial pressure, molar mass, temperature.

        Arguments:
            - partial_pressure : Partial pressure [Pa].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Concentration [kg/m³].

        Examples:
            self.fget_concentration_from_pressure_internal(100.0, 0.018, 300.0)
        """
        return fget_concentration_from_pressure(
            partial_pressure, molar_mass, temperature
        )

    @ti.kernel
    def kget_concentration_from_pressure(
        self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        """
        Kernel: compute concentration from partial pressure, molar mass, T.

        Arguments:
            - partial_pressure : Partial pressure [Pa].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Concentration [kg/m³].

        Examples:
            strat.kget_concentration_from_pressure(100.0, 0.018, 300.0)
        """
        return self.fget_concentration_from_pressure_internal(
            partial_pressure, molar_mass, temperature
        )

    # ── public python-side wrappers ---------------------------------------
    def pure_vapor_pressure(self, temperature: float):
        """
        Compute pure vapor pressure at given temperature.

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - Pure vapor pressure [Pa].

        Examples:
            strat.pure_vapor_pressure(300.0)
        """
        return self.kget_pure_vapor_pressure(temperature)

    def partial_pressure(
        self, concentration: float, molar_mass: float, temperature: float
    ):
        """
        Compute partial pressure from concentration, molar mass, temperature.

        Arguments:
            - concentration : Concentration [kg/m³].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Partial pressure [Pa].

        Examples:
            strat.partial_pressure(1.0, 0.018, 300.0)
        """
        return self.kget_partial_pressure(concentration, molar_mass, temperature)

    def concentration(
        self, partial_pressure: float, molar_mass: float, temperature: float
    ):
        """
        Compute concentration from partial pressure, molar mass, temperature.

        Arguments:
            - partial_pressure : Partial pressure [Pa].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Concentration [kg/m³].

        Examples:
            strat.concentration(100.0, 0.018, 300.0)
        """
        return self.kget_concentration_from_pressure(
            partial_pressure, molar_mass, temperature
        )

    def saturation_ratio(
        self, concentration: float, molar_mass: float, temperature: float
    ):
        """
        Compute saturation ratio (partial/pure vapor pressure).

        Arguments:
            - concentration : Concentration [kg/m³].
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Saturation ratio [unitless].

        Examples:
            strat.saturation_ratio(1.0, 0.018, 300.0)
        """
        partial = self.partial_pressure(concentration, molar_mass, temperature)
        pure = self.pure_vapor_pressure(temperature)
        return partial / pure

    def saturation_concentration(self, molar_mass: float, temperature: float):
        """
        Compute saturation concentration at given molar mass and temperature.

        Arguments:
            - molar_mass : Molar mass [kg/mol].
            - temperature : Temperature [K].

        Returns:
            - Saturation concentration [kg/m³].

        Examples:
            strat.saturation_concentration(0.018, 300.0)
        """
        vp = self.pure_vapor_pressure(temperature)
        return self.concentration(vp, molar_mass, temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class ConstantVaporPressureStrategy(_VaporPressureMixin):
    """
    Constant pure-vapor-pressure Taichi strategy.

    Returns a fixed vapor pressure, independent of temperature.

    Attributes:
        - vapor_pressure : ti.field, constant vapor pressure [Pa].

    Inherited Public Methods:
        - pure_vapor_pressure
        - partial_pressure
        - concentration
        - saturation_ratio
        - saturation_concentration

    Examples:
        ```py title="ConstantVaporPressureStrategy"
        strat = ConstantVaporPressureStrategy(42.0)
        vp = strat.pure_vapor_pressure(300.0)
        # vp == 42.0
        ```

    References:
        - See module docstring.
    """
    def __init__(self, vapor_pressure: float):
        self.vapor_pressure = ti.field(dtype=ti.f64, shape=())
        self.vapor_pressure[None] = vapor_pressure

    @ti.func
    def fget_pure_vapor_pressure(self, temperature: ti.f64) -> ti.f64:
        """
        Return constant pure vapor pressure (ignores temperature).

        Arguments:
            - temperature : Temperature [K] (ignored).

        Returns:
            - Constant vapor pressure [Pa].

        Examples:
            self.fget_pure_vapor_pressure(300.0)
        """
        return self.vapor_pressure[None]

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class AntoineVaporPressureStrategy(_VaporPressureMixin):
    """
    Antoine pure-vapor-pressure Taichi strategy.

    Computes vapor pressure using the Antoine equation:
        log₁₀(P) = A − B ⁄ (C + T)
    where P is vapor pressure [Pa], T is temperature [°C], and
    A, B, C are substance-specific coefficients.

    Attributes:
        - coefficient_a : ti.field, Antoine A parameter.
        - coefficient_b : ti.field, Antoine B parameter.
        - coefficient_c : ti.field, Antoine C parameter.

    Inherited Public Methods:
        - pure_vapor_pressure
        - partial_pressure
        - concentration
        - saturation_ratio
        - saturation_concentration

    Examples:
        ```py title="AntoineVaporPressureStrategy"
        strat = AntoineVaporPressureStrategy(8.07131, 1730.63, 233.426)
        vp = strat.pure_vapor_pressure(373.15)
        ```

    References:
        - "Antoine equation", Wikipedia.
          https://en.wikipedia.org/wiki/Antoine_equation
    """
    def __init__(
        self, coefficient_a: float = 0.0, coefficient_b: float = 0.0,
        coefficient_c: float = 0.0
    ):
        self.coefficient_a = ti.field(dtype=ti.f64, shape=())
        self.coefficient_b = ti.field(dtype=ti.f64, shape=())
        self.coefficient_c = ti.field(dtype=ti.f64, shape=())
        self.coefficient_a[None] = coefficient_a
        self.coefficient_b[None] = coefficient_b
        self.coefficient_c[None] = coefficient_c

    @ti.func
    def fget_pure_vapor_pressure(self, temperature: ti.f64) -> ti.f64:
        """
        Compute pure vapor pressure using Antoine equation.

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - Pure vapor pressure [Pa].

        Examples:
            self.fget_pure_vapor_pressure(373.15)
        """
        return fget_antoine_vapor_pressure(
            self.coefficient_a[None],
            self.coefficient_b[None],
            self.coefficient_c[None],
            temperature
        )

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class ClausiusClapeyronStrategy(_VaporPressureMixin):
    """
    Clausius-Clapeyron pure-vapor-pressure Taichi strategy.

    Computes vapor pressure using the Clausius-Clapeyron equation:
        ln P = ln P₀ − L ⁄ R × (1 ⁄ T − 1 ⁄ T₀)
    where P is vapor pressure [Pa], T is temperature [K], L is latent heat
    [J/mol], R is the gas constant, and P₀, T₀ are reference values.

    Attributes:
        - latent_heat : ti.field, latent heat of vaporization [J/mol].
        - temperature_initial : ti.field, reference temperature [K].
        - pressure_initial : ti.field, reference pressure [Pa].

    Inherited Public Methods:
        - pure_vapor_pressure
        - partial_pressure
        - concentration
        - saturation_ratio
        - saturation_concentration

    Examples:
        ```py title="ClausiusClapeyronStrategy"
        strat = ClausiusClapeyronStrategy(40000, 373.15, 101325)
        vp = strat.pure_vapor_pressure(350.0)
        ```

    References:
        - "Clausius–Clapeyron relation", Wikipedia.
          https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """
    def __init__(
        self, latent_heat: float, temperature_initial: float, pressure_initial: float
    ):
        self.latent_heat = ti.field(dtype=ti.f64, shape=())
        self.temperature_initial = ti.field(dtype=ti.f64, shape=())
        self.pressure_initial = ti.field(dtype=ti.f64, shape=())
        self.latent_heat[None] = latent_heat
        self.temperature_initial[None] = temperature_initial
        self.pressure_initial[None] = pressure_initial

    @ti.func
    def fget_pure_vapor_pressure(self, temperature: ti.f64) -> ti.f64:
        """
        Compute pure vapor pressure using Clausius-Clapeyron equation.

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - Pure vapor pressure [Pa].

        Examples:
            self.fget_pure_vapor_pressure(350.0)
        """
        return fget_clausius_clapeyron_vapor_pressure(
            self.latent_heat[None],
            self.temperature_initial[None],
            self.pressure_initial[None],
            temperature,
            gas_constant=GAS_CONSTANT,
        )

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class WaterBuckStrategy(_VaporPressureMixin):
    """
    Buck pure-vapor-pressure Taichi strategy for water.

    Computes water vapor pressure using the Buck (1981) equation.

    Attributes:
        (None; no parameters.)

    Inherited Public Methods:
        - pure_vapor_pressure
        - partial_pressure
        - concentration
        - saturation_ratio
        - saturation_concentration

    Examples:
        ```py title="WaterBuckStrategy"
        strat = WaterBuckStrategy()
        vp = strat.pure_vapor_pressure(300.0)
        ```

    References:
        - Buck, A. L. "New equations for computing vapor pressure and
          enhancement factor." J. Appl. Meteorol. 20, 1527–1532 (1981).
    """
    def __init__(self):
        pass

    @ti.func
    def fget_pure_vapor_pressure(self, temperature: ti.f64) -> ti.f64:
        """
        Compute pure vapor pressure for water using Buck equation.

        Arguments:
            - temperature : Temperature [K].

        Returns:
            - Pure vapor pressure [Pa].

        Examples:
            self.fget_pure_vapor_pressure(300.0)
        """
        return fget_buck_vapor_pressure(temperature)

