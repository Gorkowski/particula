"""Particle diffusion coefficient calculation."""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.constants import BOLTZMANN_CONSTANT


def particle_diffusion_coefficient(
    temperature: Union[float, NDArray[np.float64]],
    particle_aerodynamic_mobility: Union[float, NDArray[np.float64]],
    boltzmann_constant: float = BOLTZMANN_CONSTANT.m,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the diffusion coefficient of a particle.

    Arguments:
        temperature: The temperature at which the particle is
            diffusing, in Kelvin. Defaults to 298.15 K.
        boltzmann_constant: The Boltzmann constant. Defaults to the
            standard value of 1.380649 x 10^-23 J/K.
        particle_aerodynamic_mobility: The aerodynamic mobility of
            the particle [m^2/s].

    Returns:
        The diffusion coefficient of the particle [m^2/s].
    """
    return (
        boltzmann_constant * temperature * particle_aerodynamic_mobility
    )
