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

__all__.extend([
    "TiActivityIdealMolar",
    "TiActivityIdealMass",
    "TiActivityIdealVolume",
    "TiActivityKappaParameter",
])
