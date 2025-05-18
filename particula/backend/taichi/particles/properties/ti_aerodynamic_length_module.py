"""Taichi-accelerated aerodynamic length calculation module."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_aerodynamic_length(
    physical_length: ti.f64,
    physical_slip: ti.f64,
    aerodynamic_slip: ti.f64,
    density: ti.f64,
    reference_density: ti.f64,
    aerodynamic_shape_factor: ti.f64
) -> ti.f64:
    """Element-wise Taichi function for aerodynamic length.

    Arguments:
        - physical_length : Physical length of the particle.
        - physical_slip : Physical slip correction factor.
        - aerodynamic_slip : Aerodynamic slip correction factor.
        - density : Particle density.
        - reference_density : Reference density.
        - aerodynamic_shape_factor : Aerodynamic shape factor.

    Returns:
        - Aerodynamic length.
    """
    return physical_length * ti.sqrt(
        (physical_slip / aerodynamic_slip) *
        (density / (reference_density * aerodynamic_shape_factor))
    )

@ti.kernel
def kget_aerodynamic_length(
    physical_length_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    physical_slip_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    aerodynamic_slip_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    density_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    reference_density: ti.f64,
    aerodynamic_shape_factor: ti.f64,
    aerodynamic_length_array: ti.types.ndarray(dtype=ti.f64, ndim=1)
):
    """Vectorized Taichi kernel for aerodynamic length.

    Arguments:
        - physical_length_array : Array of physical lengths.
        - physical_slip_array : Array of physical slip correction factors.
        - aerodynamic_slip_array : Array of aerodynamic slip correction factors.
        - density_array : Array of particle densities.
        - reference_density : Reference density.
        - aerodynamic_shape_factor : Aerodynamic shape factor.
        - aerodynamic_length_array : Output array for aerodynamic lengths.

    Returns:
        - None
    """
    for i in range(aerodynamic_length_array.shape[0]):
        aerodynamic_length_array[i] = fget_aerodynamic_length(
            physical_length_array[i],
            physical_slip_array[i],
            aerodynamic_slip_array[i],
            density_array[i],
            reference_density,
            aerodynamic_shape_factor
        )

@register("get_aerodynamic_length", backend="taichi")
def ti_get_aerodynamic_length(
    physical_length,
    physical_slip,
    aerodynamic_slip,
    density,
    reference_density: float = 1000.0,
    aerodynamic_shape_factor: float = 1.0
):
    """Taichi wrapper for aerodynamic length calculation.

    Arguments:
        - physical_length : Physical length(s) of the particle(s).
        - physical_slip : Physical slip correction factor(s).
        - aerodynamic_slip : Aerodynamic slip correction factor(s).
        - density : Particle density(ies).
        - reference_density : Reference density.
        - aerodynamic_shape_factor : Aerodynamic shape factor.

    Returns:
        - Aerodynamic length(s) as a NumPy array or scalar.
    """
    # 5 a – type guard
    if not (
        isinstance(physical_length, np.ndarray)
        and isinstance(physical_slip, np.ndarray)
        and isinstance(aerodynamic_slip, np.ndarray)
        and isinstance(density, np.ndarray)
    ):
        raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

    # 5 b – ensure 1-D float64 NumPy arrays
    physical_length_array = np.atleast_1d(physical_length).astype(np.float64)
    physical_slip_array = np.atleast_1d(physical_slip).astype(np.float64)
    aerodynamic_slip_array = np.atleast_1d(aerodynamic_slip).astype(np.float64)
    density_array = np.atleast_1d(density).astype(np.float64)
    n = physical_length_array.size

    # 5 c – allocate Taichi NDArray buffers
    physical_length_ti = ti.ndarray(dtype=ti.f64, shape=n)
    physical_slip_ti = ti.ndarray(dtype=ti.f64, shape=n)
    aerodynamic_slip_ti = ti.ndarray(dtype=ti.f64, shape=n)
    density_ti = ti.ndarray(dtype=ti.f64, shape=n)
    aerodynamic_length_ti = ti.ndarray(dtype=ti.f64, shape=n)
    physical_length_ti.from_numpy(physical_length_array)
    physical_slip_ti.from_numpy(physical_slip_array)
    aerodynamic_slip_ti.from_numpy(aerodynamic_slip_array)
    density_ti.from_numpy(density_array)

    # 5 d – launch the kernel
    kget_aerodynamic_length(
        physical_length_ti,
        physical_slip_ti,
        aerodynamic_slip_ti,
        density_ti,
        float(reference_density),
        float(aerodynamic_shape_factor),
        aerodynamic_length_ti
    )

    # 5 e – convert result back to NumPy and unwrap if it is a single value
    aerodynamic_length_np = aerodynamic_length_ti.to_numpy()
    return aerodynamic_length_np.item() if aerodynamic_length_np.size == 1 else aerodynamic_length_np
