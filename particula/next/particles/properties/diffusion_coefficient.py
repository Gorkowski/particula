"""Particle diffusion coefficient calculation."""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from particula.constants import BOLTZMANN_CONSTANT
from particula.next.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity
)
from particula.next.gas.properties.mean_free_path import (
    molecule_mean_free_path
)
from particula.next.particles.properties.aerodynamic_mobility_module import (
    particle_aerodynamic_mobility
)
from particula.next.particles.properties.slip_correction_module import (
    cunningham_slip_correction
)
from particula.next.particles.properties.knudsen_number_module import (
    calculate_knudsen_number
)


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


def particle_diffusion_coefficient_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the diffusion coefficient of a particle.

    Arguments:
        temperature: The temperature of the system in Kelvin (K).
        particle_radius: The radius of the particle in meters (m).
        pressure: The pressure of the system in Pascals (Pa).

    Returns:
        The diffusion coefficient of the particle in square meters per
        second (m²/s).
    """

    # Step 1: Calculate the dynamic viscosity of the gas
    dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)

    # Step 2: Calculate the mean free path of the gas molecules
    mean_free_path = molecule_mean_free_path(
        temperature=temperature,
        pressure=pressure,
        dynamic_viscosity=dynamic_viscosity,
    )

    # Step 3: Calculate the Knudsen number (characterizes flow regime)
    knudsen_number = calculate_knudsen_number(
        mean_free_path=mean_free_path, particle_radius=particle_radius
    )

    # Step 4: Calculate the slip correction factor (Cunningham correction)
    slip_correction_factor = cunningham_slip_correction(
        knudsen_number=knudsen_number,
    )

    # Step 5: Calculate the particle aerodynamic mobility
    aerodynamic_mobility = particle_aerodynamic_mobility(
        radius=particle_radius,
        slip_correction_factor=slip_correction_factor,
        dynamic_viscosity=dynamic_viscosity,
    )

    # Step 6: Calculate the particle diffusion coefficient
    return particle_diffusion_coefficient(
        temperature=temperature,
        particle_aerodynamic_mobility=aerodynamic_mobility,
    )
