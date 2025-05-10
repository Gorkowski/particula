"""
Knudsen number calculation with backend dispatch.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs

def _knudsen_number_numpy(
    particle_radius: Union[float, NDArray[np.float64]],
    mean_free_path: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the Knudsen number using NumPy.
    """
    return mean_free_path / particle_radius

try:
    import taichi as ti

    @ti.func
    def _knudsen_number_taichi(
        particle_radius: float,
        mean_free_path: float,
    ) -> float:
        return mean_free_path / particle_radius

    _HAS_TAICHI = True
except ImportError:
    _HAS_TAICHI = False

@validate_inputs(
    {
        "particle_radius": "positive",
        "mean_free_path": "positive",
        "backend": "optional_str",
    }
)
def get_knudsen_number(
    particle_radius: Union[float, NDArray[np.float64]],
    mean_free_path: Union[float, NDArray[np.float64]],
    backend: str = "numpy",
) -> Union[float, NDArray[np.float64]]:
    """
    Compute the Knudsen number (Kn) with backend dispatch.

    Arguments:
        - particle_radius : Particle radius in meters (m).
        - mean_free_path : Mean free path in meters (m).
        - backend : "numpy" (default) or "taichi".

    Returns:
        - The Knudsen number (dimensionless).
    """
    if backend == "taichi":
        if not _HAS_TAICHI:
            raise ImportError("Taichi is not installed.")
        return _knudsen_number_taichi(particle_radius, mean_free_path)
    # Default to numpy
    return _knudsen_number_numpy(particle_radius, mean_free_path)
