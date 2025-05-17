import numpy as np, particula as par
from particula.backend.taichi.particles.ti_activity_strategies import (
    ActivityIdealMolar as TiMolar,
    ActivityIdealMass  as TiMass,
    ActivityIdealVolume as TiVol,
    ActivityKappaParameter as TiKap,
)

import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)


def _check(strategy_py, strategy_ti, conc, p0):
    np.testing.assert_allclose(
        strategy_py.activity(conc),
        strategy_ti.activity(conc),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        strategy_py.partial_pressure(p0, conc),
        strategy_ti.partial_pressure(p0, conc),
        rtol=1e-12,
    )


# ───────────────────────── strategy-specific tests ──────────────────────────
def test_activity_ideal_molar():
    conc_s, conc_v = 1.23, np.array([0.5, 1.0, 2.0])
    p0_s,  p0_v    = 950.0, np.array([500.0, 800.0, 1000.0])

    _check(
        par.particles.ActivityIdealMolar(molar_mass=0.018),
        TiMolar(molar_mass=0.018),
        conc_s, p0_s,
    )
    _check(
        par.particles.ActivityIdealMolar(molar_mass=0.018),
        TiMolar(molar_mass=0.018),
        conc_v, p0_v,
    )


def test_activity_ideal_mass():
    conc_s, conc_v = 1.23, np.array([0.5, 1.0, 2.0])
    p0_s,  p0_v    = 950.0, np.array([500.0, 800.0, 1000.0])

    _check(par.particles.ActivityIdealMass(), TiMass(), conc_s, p0_s)
    _check(par.particles.ActivityIdealMass(), TiMass(), conc_v, p0_v)


def test_activity_ideal_volume():
    conc_s, conc_v = 1.23, np.array([0.5, 1.0, 2.0])
    p0_s,  p0_v    = 950.0, np.array([500.0, 800.0, 1000.0])

    density = 1000.0
    _check(
        par.particles.ActivityIdealVolume(density=density),
        TiVol(density=density),
        conc_s, p0_s,
    )
    _check(
        par.particles.ActivityIdealVolume(density=density),
        TiVol(density=density),
        conc_v, p0_v,
    )


def test_activity_kappa_parameter():
    kappa      = np.array([0.1, 0.0])
    density    = np.array([1000.0, 1200.0])
    molar_mass = np.array([0.018, 0.058])
    conc_arr   = np.array([1.0, 2.0])
    p0_arr     = np.array([600.0, 700.0])

    _check(
        par.particles.ActivityKappaParameter(kappa, density, molar_mass),
        TiKap(kappa, density, molar_mass),
        conc_arr,
        p0_arr,
    )
