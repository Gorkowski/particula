"""Particle settling velocity in a fluid."""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.constants import STANDARD_GRAVITY


def particle_settling_velocity(
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    gravitational_acceleration: float = STANDARD_GRAVITY.m,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the settling velocity of a particle in a fluid.

    Arguments:
        particle_radius: The radius of the particle [m].
        particle_density: The density of the particle [kg/m³].
        slip_correction_factor: The slip correction factor to
            account for non-continuum effects [dimensionless].
        gravitational_acceleration: The gravitational acceleration.
            Defaults to standard gravity [9.80665 m/s²].
        dynamic_viscosity: The dynamic viscosity of the fluid [Pa*s].

    Returns:
        The settling velocity of the particle in the fluid [m/s].

    """

    # Calculate the settling velocity using the given formula
    settling_velocity = (
        (2 * particle_radius) ** 2
        * particle_density
        * slip_correction_factor
        * gravitational_acceleration
        / (18 * dynamic_viscosity)
    )

    return settling_velocity
