"""Taichi implementation of particula.gas.vapor_pressure_strategies."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register
ti.init(default_fp=ti.f64)

# --- thermodynamic helpers --------------------------------------------------
@ti.func
def fget_partial_pressure(conc: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
    R = 8.314462618
    return conc * R * T / M                     # ideal-gas p = ρRT/M

@ti.func
def fget_concentration_from_pressure(p: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
    R = 8.314462618
    return p * M / (R * T)                      # ρ = pM / RT

# --- pure-vapor-pressure formulas -------------------------------------------
@ti.func
def fget_antoine_vp(a: ti.f64, b: ti.f64, c: ti.f64, T: ti.f64) -> ti.f64:
    log10_p = a - b / (T - c)
    return ti.pow(10.0, log10_p) * 133.32238741499998   # mmHg → Pa

@ti.func
def fget_clausius_clapeyron_vp(L: ti.f64,
                               T0: ti.f64,
                               P0: ti.f64,
                               T: ti.f64) -> ti.f64:
    R = 8.314462618
    return P0 * ti.exp((L / R) * (1.0 / T0 - 1.0 / T))

@ti.func
def fget_buck_vp(T: ti.f64) -> ti.f64:
    t_c = T - 273.15
    below = 6.1115 * ti.exp((23.036 - t_c / 333.7) * t_c / (279.82 + t_c)) * 100.0
    above = 6.1121 * ti.exp((18.678 - t_c / 234.5) * t_c / (257.14 + t_c)) * 100.0
    return ti.select(t_c < 0.0, below, above)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class ConstantVaporPressureStrategy:
    """Constant pure-vapor-pressure Taichi strategy."""
    def __init__(self, vapor_pressure: float):
        self.vp = ti.field(dtype=ti.f64, shape=())
        self.vp[None] = vapor_pressure

    @ti.kernel
    def _pure_vp(self, T: ti.f64) -> ti.f64:
        return self.vp[None]

    @ti.kernel
    def _partial_p(self, conc: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_partial_pressure(conc, M, T)

    @ti.kernel
    def _concentration(self, p: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_concentration_from_pressure(p, M, T)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_p(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vp = self.pure_vapor_pressure(temperature)
        return self.concentration(vp, molar_mass, temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class AntoineVaporPressureStrategy:
    """Antoine pure-vapor-pressure Taichi strategy."""
    def __init__(self, a: float = 0.0, b: float = 0.0, c: float = 0.0):
        self.a = ti.field(dtype=ti.f64, shape=())
        self.b = ti.field(dtype=ti.f64, shape=())
        self.c = ti.field(dtype=ti.f64, shape=())
        self.a[None] = a
        self.b[None] = b
        self.c[None] = c

    @ti.kernel
    def _pure_vp(self, T: ti.f64) -> ti.f64:
        return fget_antoine_vp(self.a[None], self.b[None], self.c[None], T)

    @ti.kernel
    def _partial_p(self, conc: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_partial_pressure(conc, M, T)

    @ti.kernel
    def _concentration(self, p: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_concentration_from_pressure(p, M, T)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_p(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vp = self.pure_vapor_pressure(temperature)
        return self.concentration(vp, molar_mass, temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class ClausiusClapeyronStrategy:
    """Clausius-Clapeyron pure-vapor-pressure Taichi strategy."""
    def __init__(self, latent_heat: float, temperature_initial: float, pressure_initial: float):
        self.latent_heat = ti.field(dtype=ti.f64, shape=())
        self.temperature_initial = ti.field(dtype=ti.f64, shape=())
        self.pressure_initial = ti.field(dtype=ti.f64, shape=())
        self.latent_heat[None] = latent_heat
        self.temperature_initial[None] = temperature_initial
        self.pressure_initial[None] = pressure_initial

    @ti.kernel
    def _pure_vp(self, T: ti.f64) -> ti.f64:
        return fget_clausius_clapeyron_vp(
            self.latent_heat[None],
            self.temperature_initial[None],
            self.pressure_initial[None],
            T
        )

    @ti.kernel
    def _partial_p(self, conc: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_partial_pressure(conc, M, T)

    @ti.kernel
    def _concentration(self, p: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_concentration_from_pressure(p, M, T)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_p(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vp = self.pure_vapor_pressure(temperature)
        return self.concentration(vp, molar_mass, temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class WaterBuckStrategy:
    """Buck pure-vapor-pressure Taichi strategy for water."""
    def __init__(self):
        pass

    @ti.kernel
    def _pure_vp(self, T: ti.f64) -> ti.f64:
        return fget_buck_vp(T)

    @ti.kernel
    def _partial_p(self, conc: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_partial_pressure(conc, M, T)

    @ti.kernel
    def _concentration(self, p: ti.f64, M: ti.f64, T: ti.f64) -> ti.f64:
        return fget_concentration_from_pressure(p, M, T)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_p(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vp = self.pure_vapor_pressure(temperature)
        return self.concentration(vp, molar_mass, temperature)

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
