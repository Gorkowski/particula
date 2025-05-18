import numpy as np
import taichi as ti
import particula as par

from particula.backend.taichi.particles.ti_surface_strategies import (
    TiSurfaceStrategyMolar   as TiMolar,
)

ti.init(arch=ti.cpu, default_fp=ti.f64)      # only needs to be done once


# ─────────────────────────── helpers ─────────────────────────────────────────
def _check(py_strat, ti_strat,
           molar_mass, mass_conc,
           radius, temperature):

    np.testing.assert_allclose(
        py_strat.effective_surface_tension(mass_conc),
        ti_strat.effective_surface_tension(mass_conc),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        py_strat.effective_density(mass_conc),
        ti_strat.effective_density(mass_conc),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        py_strat.kelvin_radius(molar_mass, mass_conc, temperature),
        ti_strat.kelvin_radius(molar_mass, mass_conc, temperature),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        py_strat.kelvin_term(radius, molar_mass, mass_conc, temperature),
        ti_strat.kelvin_term(radius, molar_mass, mass_conc, temperature),
        rtol=1e-7,
    )


# ───────────────────────── strategy-specific tests ───────────────────────────
def test_surface_strategy_molar():
    σ   = np.array([0.072, 0.058], dtype=np.float64)   # surface tension [N m⁻¹]
    ρ   = np.array([998.0, 1100.0], dtype=np.float64)  # density [kg m⁻³]
    M   = np.array([0.018, 0.046], dtype=np.float64)   # molar mass [kg mol⁻¹]
    c   = np.array([1.3, 2.6], dtype=np.float64)       # mass concentration [kg m⁻³]
    r   = 5e-7                                         # particle radius [m]
    T   = 298.15                                       # temperature [K]

    _check(
        par.particles.SurfaceStrategyMolar(σ, ρ, M),
        TiMolar(σ, ρ, M),
        M, c, r, T,
    )


def test_surface_strategy_mass():
    σ = np.array([0.065, 0.045], dtype=np.float64)
    ρ = np.array([1050.0, 900.0], dtype=np.float64)
    c = np.array([0.8, 3.1], dtype=np.float64)
    r = 4e-7
    T = 300.0

    _check(
        par.particles.SurfaceStrategyMass(σ, ρ),
        TiMass(σ, ρ),
        None, c, r, T,        # molar_mass not used by mass-weighted recipe
    )


def test_surface_strategy_volume():
    σ = np.array([0.069], dtype=np.float64)
    ρ = np.array([970.0], dtype=np.float64)
    c = np.array([1.0], dtype=np.float64)
    r = 1e-6
    T = 310.0

    _check(
        par.particles.SurfaceStrategyVolume(σ, ρ),
        TiVol(σ, ρ),
        None, c, r, T,        # molar_mass not used for volume weighting
    )

