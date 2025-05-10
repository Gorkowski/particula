import numpy as np
from numpy.typing import NDArray
from typing import Union
from particula.backend.dispatch_register import register
from particula.util.validate_inputs import validate_inputs


@validate_inputs(
    {
        "particle_radius": "nonnegative",
        "dynamic_viscosity": "positive",
        "slip_correction": "positive",
    }
)
def get_friction_factor(
    particle_radius: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
    slip_correction: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Reference NumPy implementation: f = 6πμr / C."""
    return 6.0 * np.pi * dynamic_viscosity * particle_radius / slip_correction


# expose to the dispatch system
register("get_friction_factor", backend="python")(get_friction_factor)
