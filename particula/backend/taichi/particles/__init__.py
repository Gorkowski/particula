"""
Particles module for Taichi backend.
"""

from particula.backend.taichi.particles.properties.ti_knudsen_number_module import (
    ti_get_knudsen_number,
    kget_knudsen_number,
    fget_knudsen_number,
)

from .ti_activity_strategies import (          # noqa: F401
    ActivityIdealMolar as TiActivityIdealMolar,
    ActivityIdealMass  as TiActivityIdealMass,
    ActivityIdealVolume as TiActivityIdealVolume,
    ActivityKappaParameter as TiActivityKappaParameter,
)

# ─── Distribution strategies ───────────────────────────────────────────
from .ti_distribution_strategies import (
    MassBasedMovingBin            as TiMassBasedMovingBin,
    RadiiBasedMovingBin           as TiRadiiBasedMovingBin,
    SpeciatedMassMovingBin        as TiSpeciatedMassMovingBin,
    ParticleResolvedSpeciatedMass as TiParticleResolvedSpeciatedMass,
)

# ─── Surface strategies ───────────────────────────────────────────
from .ti_surface_strategies import (
    SurfaceStrategyMolar   as TiSurfaceStrategyMolar,
    SurfaceStrategyMass    as TiSurfaceStrategyMass,
    SurfaceStrategyVolume  as TiSurfaceStrategyVolume,
)

from .ti_representation import TiParticleRepresentation          # NEW
__all__ = [
    "TiActivityIdealMolar",
    "TiActivityIdealMass",
    "TiActivityIdealVolume",
    "TiActivityKappaParameter",
    "TiMassBasedMovingBin",
    "TiRadiiBasedMovingBin",
    "TiSpeciatedMassMovingBin",
    "TiParticleResolvedSpeciatedMass",
]
__all__.extend([
    "TiSurfaceStrategyMolar",
    "TiSurfaceStrategyMass",
    "TiSurfaceStrategyVolume",
])
__all__.append("TiParticleRepresentation")                       # NEW
