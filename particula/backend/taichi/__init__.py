"""
taichi backend for Particula

import/initalize taichi
"""

import taichi as ti

from . import dynamics       # noqa: F401
from . import particles      # noqa: F401
from .particles import properties as _particle_properties   # noqa: F401
from . import gas            # noqa: F401
from .gas import properties as _gas_properties              # noqa: F401
from . import util           # noqa: F401

__all__ = ["dynamics", "particles", "gas", "util"]
