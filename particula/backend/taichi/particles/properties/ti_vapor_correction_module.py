# required imports
import taichi as ti
import numpy as np
from numbers import Number  # new
from particula.backend.dispatch_register import register


# ─── 3. element-wise Taichi function ────────────────────────────────────────
@ti.func
def fget_vapor_transition_correction(
    knudsen_number: ti.f64, mass_accommodation: ti.f64
) -> ti.f64:
    """
    Taichi function to compute the vapor transition correction.
    """
    return (0.75 * mass_accommodation * (1.0 + knudsen_number)) / (
        knudsen_number * knudsen_number
        + knudsen_number
        + 0.283 * mass_accommodation * knudsen_number
        + 0.75 * mass_accommodation
    )


# ─── 4. vectorised kernel ───────────────────────────────────────────────────
@ti.kernel
def kget_vapor_transition_correction(
    knudsen_number: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mass_accommodation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Taichi kernel to compute the vapor transition correction.
    """
    for i in range(result.shape[0]):
        result[i] = fget_vapor_transition_correction(
            knudsen_number[i], mass_accommodation[i]
        )


# ─── 5. public wrapper (backend registration) ───────────────────────────────
@register("get_vapor_transition_correction", backend="taichi")
def ti_get_vapor_transition_correction(knudsen_number, mass_accommodation):
    """
    Taichi backend for get_vapor_transition_correction.

    Accepts scalar or array-like inputs, broadcasts them to the same
    shape, calls the Taichi kernel and returns a NumPy array with the
    broadcast shape (or a scalar if both inputs were scalars).
    """
    # 5 a – coerce to NumPy arrays (scalars become 0-d arrays)
    knudsen_number_np = (
        np.asarray(knudsen_number, dtype=np.float64)
        if not isinstance(knudsen_number, Number)
        else np.array(knudsen_number, dtype=np.float64)
    )
    mass_accommodation_np = (
        np.asarray(mass_accommodation, dtype=np.float64)
        if not isinstance(mass_accommodation, Number)
        else np.array(mass_accommodation, dtype=np.float64)
    )

    # 5 b – broadcast to a common shape and flatten
    knudsen_number_b, mass_accommodation_b = np.broadcast_arrays(
        knudsen_number_np, mass_accommodation_np
    )
    flat_knudsen_number = knudsen_number_b.ravel()
    flat_mass_accommodation = mass_accommodation_b.ravel()
    n = flat_knudsen_number.size

    # 5 c – allocate Taichi buffers
    knudsen_number_ti = ti.ndarray(dtype=ti.f64, shape=n)
    mass_accommodation_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    knudsen_number_ti.from_numpy(flat_knudsen_number)
    mass_accommodation_ti.from_numpy(flat_mass_accommodation)

    # 5 d – launch the kernel
    kget_vapor_transition_correction(
        knudsen_number_ti, mass_accommodation_ti, result_ti
    )

    # 5 e – reshape back and restore scalar return if needed
    result_array = result_ti.to_numpy().reshape(knudsen_number_b.shape)
    return result_array.item() if result_array.size == 1 else result_array
