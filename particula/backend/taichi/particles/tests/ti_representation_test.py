import taichi as ti; ti.init(arch=ti.cpu, default_fp=ti.f64)
import numpy as np, numpy.testing as npt

from particula.particles.distribution_strategies import MassBasedMovingBin
from particula.backend.taichi.particles.ti_distribution_strategies import TiMassBasedMovingBin
from particula.particles.activity_strategies import ActivityIdealMass
from particula.backend.taichi.particles.ti_activity_strategies import ActivityIdealMass as TiActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyMass
from particula.backend.taichi.particles.ti_surface_strategies import SurfaceStrategyMass as TiSurfaceStrategyMass
from particula.particles.representation import ParticleRepresentation as PyRep
from particula.backend.taichi.particles.ti_representation import TiParticleRepresentation as TiRep

def test_mass_concentration_parity():
    distribution   = np.array([1e-18, 3e-18])
    density        = np.array([1000.0])
    concentration  = np.array([1e6, 2e6])
    charge         = np.array([0.0, 0.0])

    py_obj = PyRep(
        MassBasedMovingBin(), ActivityIdealMass(), SurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )
    ti_obj = TiRep(
        TiMassBasedMovingBin(), TiActivityIdealMass(), TiSurfaceStrategyMass(),
        distribution, density, concentration, charge,
    )
    npt.assert_allclose(py_obj.get_mass_concentration(),
                        ti_obj.get_mass_concentration(),
                        rtol=1e-12)
