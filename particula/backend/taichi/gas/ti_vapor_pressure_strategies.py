"""Taichi implementation of particula.gas.vapor_pressure_strategies."""
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

ti.init(default_fp=ti.f64)

GAS_CONSTANT = 8.31446261815324  # J/(mol·K)


@ti.data_oriented
class _VaporPressureMixin:
    """Shared Taichi helpers for vapor-pressure strategies."""

    # ── pure-vapor-pressure ────────────────────────────────────────────────
    @ti.kernel
    def _pure_vp_kernel(self, temperature: ti.f64) -> ti.f64:
        return self._pure_vp_func(temperature)   # subclass provides _pure_vp_func

    # ── partial pressure ---------------------------------------------------
    @ti.func
    def _partial_pressure_func(
        self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        return fget_partial_pressure(concentration, molar_mass, temperature)

    @ti.kernel
    def _partial_pressure_kernel(
        self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        return self._partial_pressure_func(concentration, molar_mass, temperature)

    # ── concentration from pressure ---------------------------------------
    @ti.func
    def _concentration_func(
        self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        return fget_concentration_from_pressure(partial_pressure, molar_mass, temperature)

    @ti.kernel
    def _concentration_kernel(
        self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64
    ) -> ti.f64:
        return self._concentration_func(partial_pressure, molar_mass, temperature)

    # ── public python-side wrappers ---------------------------------------
    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp_kernel(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_pressure_kernel(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration_kernel(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vp = self.pure_vapor_pressure(temperature)
        return self.concentration(vp, molar_mass, temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class ConstantVaporPressureStrategy(_VaporPressureMixin):
    """Constant pure-vapor-pressure Taichi strategy."""
    def __init__(self, vapor_pressure: float):
        self.vapor_pressure = ti.field(dtype=ti.f64, shape=())
        self.vapor_pressure[None] = vapor_pressure

    @ti.func
    def _pure_vp_func(self, temperature: ti.f64) -> ti.f64:
        """Element-wise pure-vapor-pressure."""
        return self.vapor_pressure[None]

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class AntoineVaporPressureStrategy(_VaporPressureMixin):
    """Antoine pure-vapor-pressure Taichi strategy."""
    def __init__(self, coefficient_a: float = 0.0, coefficient_b: float = 0.0, coefficient_c: float = 0.0):
        self.coefficient_a = ti.field(dtype=ti.f64, shape=())
        self.coefficient_b = ti.field(dtype=ti.f64, shape=())
        self.coefficient_c = ti.field(dtype=ti.f64, shape=())
        self.coefficient_a[None] = coefficient_a
        self.coefficient_b[None] = coefficient_b
        self.coefficient_c[None] = coefficient_c

    @ti.func
    def _pure_vp_func(self, temperature: ti.f64) -> ti.f64:
        """Element-wise Antoine pure vapor pressure."""
        return fget_antoine_vapor_pressure(self.coefficient_a[None], self.coefficient_b[None], self.coefficient_c[None], temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class ClausiusClapeyronStrategy(_VaporPressureMixin):
    """Clausius-Clapeyron pure-vapor-pressure Taichi strategy."""
    def __init__(self, latent_heat: float, temperature_initial: float, pressure_initial: float):
        self.latent_heat = ti.field(dtype=ti.f64, shape=())
        self.temperature_initial = ti.field(dtype=ti.f64, shape=())
        self.pressure_initial = ti.field(dtype=ti.f64, shape=())
        self.latent_heat[None] = latent_heat
        self.temperature_initial[None] = temperature_initial
        self.pressure_initial[None] = pressure_initial

    @ti.func
    def _pure_vp_func(self, temperature: ti.f64) -> ti.f64:
        """Element-wise Clausius-Clapeyron pure vapor pressure."""
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
    """Buck pure-vapor-pressure Taichi strategy for water."""
    def __init__(self):
        pass

    @ti.func
    def _pure_vp_func(self, temperature: ti.f64) -> ti.f64:
        """Element-wise Buck pure vapor pressure for water."""
        return fget_buck_vapor_pressure(temperature)

# ─────────────────────────────────────────────────────────────────────────────
@register("ConstantVaporPressureStrategy", backend="taichi")
def _build_constant(*args, **kw):
    return ConstantVaporPressureStrategy(*args, **kw)

@register("AntoineVaporPressureStrategy", backend="taichi")
def _build_antoine(*args, **kw):
    return AntoineVaporPressureStrategy(*args, **kw)

@register("ClausiusClapeyronStrategy", backend="taichi")
def _build_cc(*args, **kw):
    return ClausiusClapeyronStrategy(*args, **kw)

@register("WaterBuckStrategy", backend="taichi")
def _build_buck(*args, **kw):
    return WaterBuckStrategy(*args, **kw)
