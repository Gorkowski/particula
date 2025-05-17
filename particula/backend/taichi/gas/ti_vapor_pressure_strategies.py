"""Taichi implementation of particula.gas.vapor_pressure_strategies."""
import taichi as ti
from particula.backend.dispatch_register import register
ti.init(default_fp=ti.f64)

# --- thermodynamic helpers --------------------------------------------------
@ti.func
def fget_partial_pressure(
    concentration: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise ideal-gas partial pressure."""
    gas_constant = 8.314462618
    return concentration * gas_constant * temperature / molar_mass  # ideal-gas p = ρRT/M

@ti.func
def fget_concentration_from_pressure(
    partial_pressure: ti.f64,
    molar_mass: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise concentration from pressure."""
    gas_constant = 8.314462618
    return partial_pressure * molar_mass / (gas_constant * temperature)  # ρ = pM / RT

# --- pure-vapor-pressure formulas -------------------------------------------
@ti.func
def fget_antoine_vp(
    coefficient_a: ti.f64,
    coefficient_b: ti.f64,
    coefficient_c: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise Antoine vapor pressure."""
    log10_pressure = coefficient_a - coefficient_b / (temperature - coefficient_c)
    return ti.pow(10.0, log10_pressure) * 133.32238741499998   # mmHg → Pa

@ti.func
def fget_clausius_clapeyron_vp(
    latent_heat: ti.f64,
    temperature_initial: ti.f64,
    pressure_initial: ti.f64,
    temperature: ti.f64,
) -> ti.f64:
    """Element-wise Clausius-Clapeyron vapor pressure."""
    gas_constant = 8.314462618
    return pressure_initial * ti.exp((latent_heat / gas_constant) * (1.0 / temperature_initial - 1.0 / temperature))

@ti.func
def fget_buck_vp(temperature: ti.f64) -> ti.f64:
    """Element-wise Buck vapor pressure for water."""
    temperature_celsius = temperature - 273.15
    below = 6.1115 * ti.exp((23.036 - temperature_celsius / 333.7) * temperature_celsius / (279.82 + temperature_celsius)) * 100.0
    above = 6.1121 * ti.exp((18.678 - temperature_celsius / 234.5) * temperature_celsius / (257.14 + temperature_celsius)) * 100.0
    return ti.select(temperature_celsius < 0.0, below, above)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class ConstantVaporPressureStrategy:
    """Constant pure-vapor-pressure Taichi strategy."""
    def __init__(self, vapor_pressure: float):
        self.vapor_pressure = ti.field(dtype=ti.f64, shape=())
        self.vapor_pressure[None] = vapor_pressure

    @ti.func
    def _pure_vp_func(self, temperature: ti.f64) -> ti.f64:
        """Element-wise pure-vapor-pressure."""
        return self.vapor_pressure[None]

    @ti.kernel
    def _pure_vp_kernel(self, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for pure vapor pressure."""
        return self._pure_vp_func(temperature)

    @ti.func
    def _partial_pressure_func(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise partial pressure."""
        return fget_partial_pressure(concentration, molar_mass, temperature)

    @ti.kernel
    def _partial_pressure_kernel(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for partial pressure."""
        return self._partial_pressure_func(concentration, molar_mass, temperature)

    @ti.func
    def _concentration_func(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise concentration from pressure."""
        return fget_concentration_from_pressure(partial_pressure, molar_mass, temperature)

    @ti.kernel
    def _concentration_kernel(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for concentration from pressure."""
        return self._concentration_func(partial_pressure, molar_mass, temperature)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp_kernel(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_pressure_kernel(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration_kernel(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vapor_pressure = self.pure_vapor_pressure(temperature)
        return self.concentration(vapor_pressure, molar_mass, temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class AntoineVaporPressureStrategy:
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
        return fget_antoine_vp(self.coefficient_a[None], self.coefficient_b[None], self.coefficient_c[None], temperature)

    @ti.kernel
    def _pure_vp_kernel(self, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for Antoine pure vapor pressure."""
        return self._pure_vp_func(temperature)

    @ti.func
    def _partial_pressure_func(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise partial pressure."""
        return fget_partial_pressure(concentration, molar_mass, temperature)

    @ti.kernel
    def _partial_pressure_kernel(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for partial pressure."""
        return self._partial_pressure_func(concentration, molar_mass, temperature)

    @ti.func
    def _concentration_func(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise concentration from pressure."""
        return fget_concentration_from_pressure(partial_pressure, molar_mass, temperature)

    @ti.kernel
    def _concentration_kernel(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for concentration from pressure."""
        return self._concentration_func(partial_pressure, molar_mass, temperature)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp_kernel(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_pressure_kernel(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration_kernel(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vapor_pressure = self.pure_vapor_pressure(temperature)
        return self.concentration(vapor_pressure, molar_mass, temperature)

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

    @ti.func
    def _pure_vp_func(self, temperature: ti.f64) -> ti.f64:
        """Element-wise Clausius-Clapeyron pure vapor pressure."""
        return fget_clausius_clapeyron_vp(
            self.latent_heat[None],
            self.temperature_initial[None],
            self.pressure_initial[None],
            temperature
        )

    @ti.kernel
    def _pure_vp_kernel(self, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for Clausius-Clapeyron pure vapor pressure."""
        return self._pure_vp_func(temperature)

    @ti.func
    def _partial_pressure_func(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise partial pressure."""
        return fget_partial_pressure(concentration, molar_mass, temperature)

    @ti.kernel
    def _partial_pressure_kernel(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for partial pressure."""
        return self._partial_pressure_func(concentration, molar_mass, temperature)

    @ti.func
    def _concentration_func(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise concentration from pressure."""
        return fget_concentration_from_pressure(partial_pressure, molar_mass, temperature)

    @ti.kernel
    def _concentration_kernel(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for concentration from pressure."""
        return self._concentration_func(partial_pressure, molar_mass, temperature)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp_kernel(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_pressure_kernel(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration_kernel(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vapor_pressure = self.pure_vapor_pressure(temperature)
        return self.concentration(vapor_pressure, molar_mass, temperature)

# ─────────────────────────────────────────────────────────────────────────────
@ti.data_oriented
class WaterBuckStrategy:
    """Buck pure-vapor-pressure Taichi strategy for water."""
    def __init__(self):
        pass

    @ti.func
    def _pure_vp_func(self, temperature: ti.f64) -> ti.f64:
        """Element-wise Buck pure vapor pressure for water."""
        return fget_buck_vp(temperature)

    @ti.kernel
    def _pure_vp_kernel(self, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for Buck pure vapor pressure."""
        return self._pure_vp_func(temperature)

    @ti.func
    def _partial_pressure_func(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise partial pressure."""
        return fget_partial_pressure(concentration, molar_mass, temperature)

    @ti.kernel
    def _partial_pressure_kernel(self, concentration: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for partial pressure."""
        return self._partial_pressure_func(concentration, molar_mass, temperature)

    @ti.func
    def _concentration_func(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Element-wise concentration from pressure."""
        return fget_concentration_from_pressure(partial_pressure, molar_mass, temperature)

    @ti.kernel
    def _concentration_kernel(self, partial_pressure: ti.f64, molar_mass: ti.f64, temperature: ti.f64) -> ti.f64:
        """Kernel wrapper for concentration from pressure."""
        return self._concentration_func(partial_pressure, molar_mass, temperature)

    def pure_vapor_pressure(self, temperature: float):
        return self._pure_vp_kernel(temperature)

    def partial_pressure(self, concentration: float, molar_mass: float, temperature: float):
        return self._partial_pressure_kernel(concentration, molar_mass, temperature)

    def concentration(self, partial_pressure: float, molar_mass: float, temperature: float):
        return self._concentration_kernel(partial_pressure, molar_mass, temperature)

    def saturation_ratio(self, concentration: float, molar_mass: float, temperature: float):
        return self.partial_pressure(concentration, molar_mass, temperature) / self.pure_vapor_pressure(temperature)

    def saturation_concentration(self, molar_mass: float, temperature: float):
        vapor_pressure = self.pure_vapor_pressure(temperature)
        return self.concentration(vapor_pressure, molar_mass, temperature)

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
