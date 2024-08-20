"""Module for aerodynamic mobility of a particle in a fluid.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


AERODYNAMIC_SHAPE_FACTOR_DICT = {
    "sphere": 1.0,
    "cube": 1.08,
    "cylinder_avg_aspect_2": 1.10,
    "cylinder_avg_aspect_5": 1.35,
    "cylinder_avg_aspect_10": 1.68,
    "spheres_cluster_3": 1.15,
    "spheres_cluster_4": 1.17,
    "bituminous_coal": 1.08,
    "quartz": 1.36,
    "sand": 1.57,
    "talc": 1.88,
}


def particle_aerodynamic_mobility(
    radius: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the aerodynamic mobility of a particle, defined as the ratio
    of the slip correction factor to the product of the dynamic viscosity of
    the fluid, the particle radius, and a slip correction constant derived.

    This mobility quantifies the ease with which a particle can move through
    a fluid.

    Arguments:
        radius : The radius of the particle (m).
        slip_correction_factor : The slip correction factor for the particle
            in the fluid (dimensionless).
        dynamic_viscosity : The dynamic viscosity of the fluid (Pa.s).

    Returns:
        The particle aerodynamic mobility (m^2/s).
    """
    return slip_correction_factor / (6 * np.pi * dynamic_viscosity * radius)
