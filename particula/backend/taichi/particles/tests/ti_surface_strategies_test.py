import taichi as ti
import numpy as np
import unittest
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.backend.taichi.particles.ti_surface_strategies import (
    SurfaceStrategyMolar, SurfaceStrategyMass, SurfaceStrategyVolume
)

