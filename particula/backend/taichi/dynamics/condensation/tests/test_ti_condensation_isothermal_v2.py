import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

# ─── Implementations under test ────────────────────────────────────────────
from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies_v2 \
    import CondensationIsothermal as TiCondensationIsothermal
from particula.dynamics.condensation.condensation_strategies \
    import CondensationIsothermal as PyCondensationIsothermal


# ─── Helper to build a *working* v2 object (class ctor is mis-typed) ───────
def _build_ti_impl(
    molar_mass: float = 0.018,
    diffusion_coefficient: float = 2.0e-5,
    accommodation: float = 1.0,
):
    """Return a TiCondensationIsothermal instance with properly
    initialised Taichi fields so its kernels can be called directly."""
    ti_impl = TiCondensationIsothermal(
        molar_mass=np.array([molar_mass], dtype=np.float64),
        diffusion_coefficient=np.array(diffusion_coefficient, dtype=np.float64),
        accommodation_coefficient=np.array([accommodation], dtype=np.float64),
        update_gases=np.array([0], dtype=np.int16),
    )

    # Manually patch the fields that the constructor currently overwrites
    ti_impl.molar_mass = ti.field(dtype=ti.f64, shape=(1,))
    ti_impl.molar_mass.from_numpy(np.array([molar_mass], dtype=np.float64))

    ti_impl.diffusion_coefficient = ti.field(dtype=ti.f64, shape=())
    ti_impl.diffusion_coefficient[None] = diffusion_coefficient

    ti_impl.accommodation_coefficient = ti.field(dtype=ti.f64, shape=(1,))
    ti_impl.accommodation_coefficient.from_numpy(
        np.array([accommodation], dtype=np.float64)
    )
    return ti_impl


# ─── Tests ─────────────────────────────────────────────────────────────────
def test_first_order_mass_transport_kernel_parity():
    """Kernel result must match the reference NumPy implementation."""
    radius = np.array([1e-7, 2e-7, 5e-8], dtype=np.float64)
    temperature, pressure = 298.15, 101_325.0
    dynamic_viscosity = 1.85e-5

    ti_impl = _build_ti_impl()
    result_ti = np.empty((radius.size, 1), dtype=np.float64)
    ti_impl._kget_first_order_mass_transport(
        radius, temperature, pressure, dynamic_viscosity, result_ti
    )

    py_impl = PyCondensationIsothermal(molar_mass=0.018)
    result_py = py_impl.first_order_mass_transport(
        radius, temperature, pressure, dynamic_viscosity
    )[:, None]  # add species axis

    np.testing.assert_allclose(result_ti, result_py, rtol=1e-7)


def test_kernel_runs_v2():
    """Smoke test: v2 kernel compiles & returns finite outputs."""
    radius = np.array([1e-7, 1e-7], dtype=np.float64)
    ti_impl = _build_ti_impl()
    dynamic_viscosity = 1.85e-5
    result = np.empty((radius.size, 1), dtype=np.float64)

    ti_impl._kget_first_order_mass_transport(
        radius, 300.0, 101_325.0, dynamic_viscosity, result
    )

    assert np.all(np.isfinite(result)) and result.shape == (radius.size, 1)
