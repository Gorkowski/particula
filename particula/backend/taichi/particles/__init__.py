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
