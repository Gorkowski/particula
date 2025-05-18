import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

# ─── Implementations under test ────────────────────────────────────────────
from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies_v2 \
    import CondensationIsothermal as TiCondensationIsothermal
from particula.dynamics.condensation.condensation_strategies \
    import CondensationIsothermal as PyCondensationIsothermal


# ─── Helper to build a *working* v2 object (class ctor is mis-typed) ───────
def _build_taichi_condensation_isothermal(
    molar_mass: float = 0.018,
    diffusion_coefficient: float = 2.0e-5,
    accommodation_coefficient: float = 1.0,
):
    """Return a TiCondensationIsothermal instance with properly
    initialised Taichi fields so its kernels can be called directly."""
    taichi_condensation = TiCondensationIsothermal(
        molar_mass=np.array([molar_mass], dtype=np.float64),
        diffusion_coefficient=np.array(diffusion_coefficient, dtype=np.float64),
        accommodation_coefficient=np.array([accommodation_coefficient], dtype=np.float64),
        update_gases=np.array([0], dtype=np.int16),
    )

    # Manually patch the fields that the constructor currently overwrites
    taichi_condensation.molar_mass = ti.field(dtype=ti.f64, shape=(1,))
    taichi_condensation.molar_mass.from_numpy(np.array([molar_mass], dtype=np.float64))

    taichi_condensation.diffusion_coefficient = ti.field(dtype=ti.f64, shape=())
    taichi_condensation.diffusion_coefficient[None] = diffusion_coefficient

    taichi_condensation.accommodation_coefficient = ti.field(dtype=ti.f64, shape=(1,))
    taichi_condensation.accommodation_coefficient.from_numpy(
        np.array([accommodation_coefficient], dtype=np.float64)
    )
    return taichi_condensation


# ─── Tests ─────────────────────────────────────────────────────────────────
def test_first_order_mass_transport_kernel_parity():
    """Kernel result must match the reference NumPy implementation."""
    particle_radius = np.array([1e-7, 2e-7, 5e-6, 3e-5], dtype=np.float64)
    temperature, pressure = 298.15, 101_325.0
    dynamic_viscosity = 1.85e-5

    taichi_condensation = _build_taichi_condensation_isothermal()
    result_taichi = np.empty((particle_radius.size, 1), dtype=np.float64)
    taichi_condensation._kget_first_order_mass_transport(
        particle_radius, temperature, pressure, dynamic_viscosity, result_taichi
    )

    python_condensation = PyCondensationIsothermal(molar_mass=0.018)
    result_python = python_condensation.first_order_mass_transport(
        particle_radius, temperature, pressure, dynamic_viscosity
    )[:, None]  # add species axis

    np.testing.assert_allclose(result_taichi, result_python, rtol=1e-7)


def test_kernel_runs_v2():
    """Smoke test: v2 kernel compiles & returns finite outputs."""
    particle_radius = np.array([1e-7, 1e-7], dtype=np.float64)
    taichi_condensation = _build_taichi_condensation_isothermal()
    dynamic_viscosity = 1.85e-5
    mass_transport_result = np.empty((particle_radius.size, 1), dtype=np.float64)

    taichi_condensation._kget_first_order_mass_transport(
        particle_radius, 300.0, 101_325.0, dynamic_viscosity, mass_transport_result
    )

    assert (
        np.all(np.isfinite(mass_transport_result))
        and mass_transport_result.shape == (particle_radius.size, 1)
    )
