"""
Integrated Aerosol Dynamics and Particle-Resolved Simulation Representation
"""

import taichi as ti
from particula.backend.taichi.util import FieldIO


ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)


