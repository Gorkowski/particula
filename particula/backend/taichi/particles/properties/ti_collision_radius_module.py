"""Taichi-accelerated collision radius models."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_collision_radius_mg1988(gyration_radius: ti.f64) -> ti.f64:
    """
    Compute the collision radius using the MG1988 model.

    Arguments:
        - gyration_radius : The radius of gyration (R_g).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        R_c = R_g

    Examples:
        ```py
        fget_collision_radius_mg1988(1.0)
        # Output: 1.0
        ```

    References:
        - Maggi, F. & Giorgi, F. (1988). "Title." Journal Name, Volume, Year.
    """
    return gyration_radius

@ti.func
def fget_collision_radius_sr1992(
    gyration_radius: ti.f64,
    fractal_dimension: ti.f64
) -> ti.f64:
    """
    Compute the collision radius using the SR1992 model.

    Arguments:
        - gyration_radius : The radius of gyration (R_g).
        - fractal_dimension : The fractal dimension (d_f).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        R_c = √((d_f + 2) / 3) × R_g

    Examples:
        ```py
        fget_collision_radius_sr1992(1.0, 1.8)
        # Output: 1.1547...
        ```

    References:
        - Sorensen, C. M. & Roberts, G. C. (1992). "Title." Journal Name, Volume, Year.
    """
    return ti.sqrt((fractal_dimension + 2.0) / 3.0) * gyration_radius

@ti.func
def fget_collision_radius_mzg2002(
    gyration_radius: ti.f64,
    fractal_prefactor: ti.f64
) -> ti.f64:
    """
    Compute the collision radius using the MZG2002 model.

    Arguments:
        - gyration_radius : The radius of gyration (R_g).
        - fractal_prefactor : The fractal prefactor (k₀).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        R_c = 1.037 × (k₀^0.077) × R_g

    Examples:
        ```py
        fget_collision_radius_mzg2002(1.0, 2.0)
        # Output: 1.081...
        ```

    References:
        - Meng, H. & Zhang, Q. (2002). "Title." Journal Name, Volume, Year.
    """
    return 1.037 * ti.pow(fractal_prefactor, 0.077) * gyration_radius

@ti.func
def fget_collision_radius_tt2012(
    fractal_dimension: ti.f64,
    number_of_particles: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """
    Compute the collision radius using the TT2012 model.

    Arguments:
        - fractal_dimension : The fractal dimension (d_f).
        - number_of_particles : Number of particles (N).
        - gyration_radius : The radius of gyration (R_g).
        - radius_monomer : The monomer radius (r_m).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        See TT2012 for full formula.

    Examples:
        ```py
        fget_collision_radius_tt2012(1.8, 100, 1.0, 0.1)
        # Output: <float>
        ```

    References:
        - Thajudeen, T. et al. (2012). "Title." Journal Name, Volume, Year.
    """
    alpha1 = 0.253 * fractal_dimension**2 - 1.209 * fractal_dimension + 1.433
    alpha2 = -0.218 * fractal_dimension**2 + 0.964 * fractal_dimension - 0.180
    phi = 1.0 / (alpha1 * ti.log(number_of_particles) + alpha2)
    radius_s_i = phi * gyration_radius
    radius_s_ii = (
        radius_monomer * (1.203 - 0.4315 / fractal_dimension) / 2.0
    ) * ti.pow(4.0 * radius_s_i / radius_monomer, 0.8806 + 0.3497 / fractal_dimension)
    return radius_s_ii / 2.0

@ti.func
def fget_collision_radius_wq2022_rg(
    gyration_radius: ti.f64,
    radius_monomer: ti.f64
) -> ti.f64:
    """
    Compute the collision radius using the WQ2022 Rg model.

    Arguments:
        - gyration_radius : The radius of gyration (R_g).
        - radius_monomer : The monomer radius (r_m).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        R_c = (0.973 × (R_g / r_m) + 0.441) × r_m

    Examples:
        ```py
        fget_collision_radius_wq2022_rg(1.0, 0.1)
        # Output: 1.414...
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    return (0.973 * (gyration_radius / radius_monomer) + 0.441) * radius_monomer

@ti.func
def fget_collision_radius_wq2022_rg_df(
    fractal_dimension: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """
    Compute the collision radius using the WQ2022 Rg-df model.

    Arguments:
        - fractal_dimension : The fractal dimension (d_f).
        - gyration_radius : The radius of gyration (R_g).
        - radius_monomer : The monomer radius (r_m).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        R_c = (0.882 × d_f^0.223 × (R_g / r_m) + 0.387) × r_m

    Examples:
        ```py
        fget_collision_radius_wq2022_rg_df(1.8, 1.0, 0.1)
        # Output: <float>
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    return (
        0.882 * ti.pow(fractal_dimension, 0.223)
        * (gyration_radius / radius_monomer)
        + 0.387
    ) * radius_monomer

@ti.func
def fget_collision_radius_wq2022_rg_df_k0(
    fractal_dimension: ti.f64,
    fractal_prefactor: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """
    Compute the collision radius using the WQ2022 Rg-df-k0 model.

    Arguments:
        - fractal_dimension : The fractal dimension (d_f).
        - fractal_prefactor : The fractal prefactor (k₀).
        - gyration_radius : The radius of gyration (R_g).
        - radius_monomer : The monomer radius (r_m).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        R_c = (0.777 × d_f^0.479 × k₀^0.000970 × (R_g/r_m)
              + 0.267 × k₀ - 0.079) × r_m

    Examples:
        ```py
        fget_collision_radius_wq2022_rg_df_k0(1.8, 2.0, 1.0, 0.1)
        # Output: <float>
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    return (
        0.777 * ti.pow(fractal_dimension, 0.479)
        * ti.pow(fractal_prefactor, 0.000970)
        * (gyration_radius / radius_monomer)
        + 0.267 * fractal_prefactor
        - 0.079
    ) * radius_monomer

@ti.func
def fget_collision_radius_wq2022_rg_df_k0_a13(
    fractal_dimension: ti.f64,
    fractal_prefactor: ti.f64,
    shape_anisotropy: ti.f64,
    gyration_radius: ti.f64,
    radius_monomer: ti.f64,
) -> ti.f64:
    """
    Compute the collision radius using the WQ2022 Rg-df-k0-a13 model.

    Arguments:
        - fractal_dimension : The fractal dimension (d_f).
        - fractal_prefactor : The fractal prefactor (k₀).
        - shape_anisotropy : The shape anisotropy (a₁₃).
        - gyration_radius : The radius of gyration (R_g).
        - radius_monomer : The monomer radius (r_m).

    Returns:
        - Collision radius (R_c) as a float.

    Equation:
        R_c = (0.876 × d_f^0.363 × k₀^-0.105 × (R_g/r_m)
              + 0.421 × k₀ - 0.036 × a₁₃ - 0.227) × r_m

    Examples:
        ```py
        fget_collision_radius_wq2022_rg_df_k0_a13(1.8, 2.0, 0.5, 1.0, 0.1)
        # Output: <float>
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    return (
        0.876 * ti.pow(fractal_dimension, 0.363)
        * ti.pow(fractal_prefactor, -0.105)
        * (gyration_radius / radius_monomer)
        + 0.421 * fractal_prefactor
        - 0.036 * shape_anisotropy
        - 0.227
    ) * radius_monomer

@ti.kernel
def kget_collision_radius_mg1988(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the MG1988 collision radius model.

    Arguments:
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        R_c = R_g

    Examples:
        ```py
        kget_collision_radius_mg1988(gyration_radius, out)
        ```

    References:
        - Maggi, F. & Giorgi, F. (1988). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_mg1988(gyration_radius[i])

@ti.kernel
def kget_collision_radius_sr1992(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the SR1992 collision radius model.

    Arguments:
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - fractal_dimension : 1D ndarray of fractal dimensions (d_f).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        R_c = √((d_f + 2) / 3) × R_g

    Examples:
        ```py
        kget_collision_radius_sr1992(gyration_radius, fractal_dimension, out)
        ```

    References:
        - Sorensen, C. M. & Roberts, G. C. (1992). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_sr1992(gyration_radius[i], fractal_dimension[i])

@ti.kernel
def kget_collision_radius_mzg2002(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_prefactor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the MZG2002 collision radius model.

    Arguments:
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - fractal_prefactor : 1D ndarray of fractal prefactors (k₀).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        R_c = 1.037 × (k₀^0.077) × R_g

    Examples:
        ```py
        kget_collision_radius_mzg2002(gyration_radius, fractal_prefactor, out)
        ```

    References:
        - Meng, H. & Zhang, Q. (2002). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_mzg2002(gyration_radius[i], fractal_prefactor[i])

@ti.kernel
def kget_collision_radius_tt2012(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    number_of_particles: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the TT2012 collision radius model.

    Arguments:
        - fractal_dimension : 1D ndarray of fractal dimensions (d_f).
        - number_of_particles : 1D ndarray of particle counts (N).
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - radius_monomer : 1D ndarray of monomer radii (r_m).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        See TT2012 for full formula.

    Examples:
        ```py
        kget_collision_radius_tt2012(
            fractal_dimension,
            number_of_particles,
            gyration_radius,
            radius_monomer,
            out
        )
        ```

    References:
        - Thajudeen, T. et al. (2012). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_tt2012(
            fractal_dimension[i],
            number_of_particles[i],
            gyration_radius[i],
            radius_monomer[i],
        )

@ti.kernel
def kget_collision_radius_wq2022_rg(
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the WQ2022 Rg collision radius model.

    Arguments:
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - radius_monomer : 1D ndarray of monomer radii (r_m).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        R_c = (0.973 × (R_g / r_m) + 0.441) × r_m

    Examples:
        ```py
        kget_collision_radius_wq2022_rg(gyration_radius, radius_monomer, out)
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg(gyration_radius[i], radius_monomer[i])

@ti.kernel
def kget_collision_radius_wq2022_rg_df(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the WQ2022 Rg-df collision radius model.

    Arguments:
        - fractal_dimension : 1D ndarray of fractal dimensions (d_f).
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - radius_monomer : 1D ndarray of monomer radii (r_m).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        R_c = (0.882 × d_f^0.223 × (R_g / r_m) + 0.387) × r_m

    Examples:
        ```py
        kget_collision_radius_wq2022_rg_df(
            fractal_dimension,
            gyration_radius,
            radius_monomer,
            out
        )
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg_df(
            fractal_dimension[i],
            gyration_radius[i],
            radius_monomer[i]
        )

@ti.kernel
def kget_collision_radius_wq2022_rg_df_k0(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_prefactor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the WQ2022 Rg-df-k0 collision radius model.

    Arguments:
        - fractal_dimension : 1D ndarray of fractal dimensions (d_f).
        - fractal_prefactor : 1D ndarray of fractal prefactors (k₀).
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - radius_monomer : 1D ndarray of monomer radii (r_m).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        R_c = (0.777 × d_f^0.479 × k₀^0.000970 × (R_g/r_m)
              + 0.267 × k₀ - 0.079) × r_m

    Examples:
        ```py
        kget_collision_radius_wq2022_rg_df_k0(
            fractal_dimension,
            fractal_prefactor,
            gyration_radius,
            radius_monomer,
            out
        )
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg_df_k0(
            fractal_dimension[i],
            fractal_prefactor[i],
            gyration_radius[i],
            radius_monomer[i],
        )

@ti.kernel
def kget_collision_radius_wq2022_rg_df_k0_a13(
    fractal_dimension: ti.types.ndarray(dtype=ti.f64, ndim=1),
    fractal_prefactor: ti.types.ndarray(dtype=ti.f64, ndim=1),
    shape_anisotropy: ti.types.ndarray(dtype=ti.f64, ndim=1),
    gyration_radius: ti.types.ndarray(dtype=ti.f64, ndim=1),
    radius_monomer: ti.types.ndarray(dtype=ti.f64, ndim=1),
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """
    Kernel for the WQ2022 Rg-df-k0-a13 collision radius model.

    Arguments:
        - fractal_dimension : 1D ndarray of fractal dimensions (d_f).
        - fractal_prefactor : 1D ndarray of fractal prefactors (k₀).
        - shape_anisotropy : 1D ndarray of shape anisotropy (a₁₃).
        - gyration_radius : 1D ndarray of gyration radii (R_g).
        - radius_monomer : 1D ndarray of monomer radii (r_m).
        - out : 1D ndarray for output collision radii.

    Returns:
        - None (results written to 'out').

    Equation:
        R_c = (0.876 × d_f^0.363 × k₀^-0.105 × (R_g/r_m)
              + 0.421 × k₀ - 0.036 × a₁₃ - 0.227) × r_m

    Examples:
        ```py
        kget_collision_radius_wq2022_rg_df_k0_a13(
            fractal_dimension,
            fractal_prefactor,
            shape_anisotropy,
            gyration_radius,
            radius_monomer,
            out
        )
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    for i in range(out.shape[0]):
        out[i] = fget_collision_radius_wq2022_rg_df_k0_a13(
            fractal_dimension[i],
            fractal_prefactor[i],
            shape_anisotropy[i],
            gyration_radius[i],
            radius_monomer[i],
        )

def _ensure_all_ndarrays(*args):
    """
    Ensure all arguments are NumPy ndarrays.

    Arguments:
        - *args : Arguments to check.

    Returns:
        - None. Raises TypeError if any argument is not a NumPy ndarray.

    Examples:
        ```py
        _ensure_all_ndarrays(np.array([1,2]), np.array([3,4]))
        ```
    """
    for arg in args:
        if not isinstance(arg, np.ndarray):
            raise TypeError("Taichi backend expects NumPy arrays for all inputs.")

def _ensure_all_same_size(*args):
    """
    Ensure all input arrays have the same size.

    Arguments:
        - *args : Arrays to check.

    Returns:
        - None. Raises ValueError if arrays are not the same size.

    Examples:
        ```py
        _ensure_all_same_size(np.array([1,2]), np.array([3,4]))
        ```
    """
    sizes = [np.atleast_1d(arg).size for arg in args]
    if any(sz != sizes[0] for sz in sizes):
        raise ValueError("All input arrays must have the same size.")

@register("get_collision_radius_mg1988", backend="taichi")
def ti_get_collision_radius_mg1988(gyration_radius):
    """
    Taichi wrapper for the MG1988 collision radius model.

    Arguments:
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        R_c = R_g

    Examples:
        ```py
        ti_get_collision_radius_mg1988(np.array([1.0, 2.0]))
        # Output: array([1.0, 2.0])
        ```

    References:
        - Maggi, F. & Giorgi, F. (1988). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(gyration_radius)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    n_elements = gyration_radius_array.size
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    kget_collision_radius_mg1988(
        gyration_radius_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_sr1992", backend="taichi")
def ti_get_collision_radius_sr1992(gyration_radius, fractal_dimension):
    """
    Taichi wrapper for the SR1992 collision radius model.

    Arguments:
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).
        - fractal_dimension : Scalar or 1D array of fractal dimensions (d_f).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        R_c = √((d_f + 2) / 3) × R_g

    Examples:
        ```py
        ti_get_collision_radius_sr1992(np.array([1.0]), np.array([1.8]))
        # Output: array([1.1547...])
        ```

    References:
        - Sorensen, C. M. & Roberts, G. C. (1992). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(gyration_radius, fractal_dimension)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    fractal_dimension_array = np.atleast_1d(fractal_dimension)
    _ensure_all_same_size(
        gyration_radius_array,
        fractal_dimension_array
    )
    n_elements = gyration_radius_array.size
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_dimension_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    fractal_dimension_ti.from_numpy(fractal_dimension_array)
    kget_collision_radius_sr1992(
        gyration_radius_ti,
        fractal_dimension_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_mzg2002", backend="taichi")
def ti_get_collision_radius_mzg2002(gyration_radius, fractal_prefactor):
    """
    Taichi wrapper for the MZG2002 collision radius model.

    Arguments:
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).
        - fractal_prefactor : Scalar or 1D array of fractal prefactors (k₀).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        R_c = 1.037 × (k₀^0.077) × R_g

    Examples:
        ```py
        ti_get_collision_radius_mzg2002(np.array([1.0]), np.array([2.0]))
        # Output: array([1.081...])
        ```

    References:
        - Meng, H. & Zhang, Q. (2002). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(gyration_radius, fractal_prefactor)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    fractal_prefactor_array = np.atleast_1d(fractal_prefactor)
    _ensure_all_same_size(
        gyration_radius_array,
        fractal_prefactor_array
    )
    n_elements = gyration_radius_array.size
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_prefactor_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    fractal_prefactor_ti.from_numpy(fractal_prefactor_array)
    kget_collision_radius_mzg2002(
        gyration_radius_ti,
        fractal_prefactor_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_tt2012", backend="taichi")
def ti_get_collision_radius_tt2012(
    fractal_dimension,
    number_of_particles,
    gyration_radius,
    radius_monomer
):
    """
    Taichi wrapper for the TT2012 collision radius model.

    Arguments:
        - fractal_dimension : Scalar or 1D array of fractal dimensions (d_f).
        - number_of_particles : Scalar or 1D array of particle counts (N).
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).
        - radius_monomer : Scalar or 1D array of monomer radii (r_m).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        See TT2012 for full formula.

    Examples:
        ```py
        ti_get_collision_radius_tt2012(
            np.array([1.8]),
            np.array([100]),
            np.array([1.0]),
            np.array([0.1])
        )
        # Output: array([<float>])
        ```

    References:
        - Thajudeen, T. et al. (2012). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(
        fractal_dimension,
        number_of_particles,
        gyration_radius,
        radius_monomer
    )
    fractal_dimension_array = np.atleast_1d(fractal_dimension)
    number_of_particles_array = np.atleast_1d(number_of_particles)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    radius_monomer_array = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(
        fractal_dimension_array,
        number_of_particles_array,
        gyration_radius_array,
        radius_monomer_array
    )
    n_elements = fractal_dimension_array.size
    fractal_dimension_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    number_of_particles_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    radius_monomer_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_dimension_ti.from_numpy(fractal_dimension_array)
    number_of_particles_ti.from_numpy(number_of_particles_array)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    radius_monomer_ti.from_numpy(radius_monomer_array)
    kget_collision_radius_tt2012(
        fractal_dimension_ti,
        number_of_particles_ti,
        gyration_radius_ti,
        radius_monomer_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg", backend="taichi")
def ti_get_collision_radius_wq2022_rg(gyration_radius, radius_monomer):
    """
    Taichi wrapper for the WQ2022 Rg collision radius model.

    Arguments:
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).
        - radius_monomer : Scalar or 1D array of monomer radii (r_m).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        R_c = (0.973 × (R_g / r_m) + 0.441) × r_m

    Examples:
        ```py
        ti_get_collision_radius_wq2022_rg(np.array([1.0]), np.array([0.1]))
        # Output: array([1.414...])
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(gyration_radius, radius_monomer)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    radius_monomer_array = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(
        gyration_radius_array,
        radius_monomer_array
    )
    n_elements = gyration_radius_array.size
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    radius_monomer_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    radius_monomer_ti.from_numpy(radius_monomer_array)
    kget_collision_radius_wq2022_rg(
        gyration_radius_ti,
        radius_monomer_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg_df", backend="taichi")
def ti_get_collision_radius_wq2022_rg_df(fractal_dimension, gyration_radius, radius_monomer):
    """
    Taichi wrapper for the WQ2022 Rg-df collision radius model.

    Arguments:
        - fractal_dimension : Scalar or 1D array of fractal dimensions (d_f).
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).
        - radius_monomer : Scalar or 1D array of monomer radii (r_m).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        R_c = (0.882 × d_f^0.223 × (R_g / r_m) + 0.387) × r_m

    Examples:
        ```py
        ti_get_collision_radius_wq2022_rg_df(
            np.array([1.8]),
            np.array([1.0]),
            np.array([0.1])
        )
        # Output: array([<float>])
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(
        fractal_dimension,
        gyration_radius,
        radius_monomer
    )
    fractal_dimension_array = np.atleast_1d(fractal_dimension)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    radius_monomer_array = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(
        fractal_dimension_array,
        gyration_radius_array,
        radius_monomer_array
    )
    n_elements = fractal_dimension_array.size
    fractal_dimension_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    radius_monomer_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_dimension_ti.from_numpy(fractal_dimension_array)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    radius_monomer_ti.from_numpy(radius_monomer_array)
    kget_collision_radius_wq2022_rg_df(
        fractal_dimension_ti,
        gyration_radius_ti,
        radius_monomer_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg_df_k0", backend="taichi")
def ti_get_collision_radius_wq2022_rg_df_k0(
    fractal_dimension,
    fractal_prefactor,
    gyration_radius,
    radius_monomer
):
    """
    Taichi wrapper for the WQ2022 Rg-df-k0 collision radius model.

    Arguments:
        - fractal_dimension : Scalar or 1D array of fractal dimensions (d_f).
        - fractal_prefactor : Scalar or 1D array of fractal prefactors (k₀).
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).
        - radius_monomer : Scalar or 1D array of monomer radii (r_m).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        R_c = (0.777 × d_f^0.479 × k₀^0.000970 × (R_g/r_m)
              + 0.267 × k₀ - 0.079) × r_m

    Examples:
        ```py
        ti_get_collision_radius_wq2022_rg_df_k0(
            np.array([1.8]),
            np.array([2.0]),
            np.array([1.0]),
            np.array([0.1])
        )
        # Output: array([<float>])
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(
        fractal_dimension,
        fractal_prefactor,
        gyration_radius,
        radius_monomer
    )
    fractal_dimension_array = np.atleast_1d(fractal_dimension)
    fractal_prefactor_array = np.atleast_1d(fractal_prefactor)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    radius_monomer_array = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(
        fractal_dimension_array,
        fractal_prefactor_array,
        gyration_radius_array,
        radius_monomer_array
    )
    n_elements = fractal_dimension_array.size
    fractal_dimension_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_prefactor_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    radius_monomer_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_dimension_ti.from_numpy(fractal_dimension_array)
    fractal_prefactor_ti.from_numpy(fractal_prefactor_array)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    radius_monomer_ti.from_numpy(radius_monomer_array)
    kget_collision_radius_wq2022_rg_df_k0(
        fractal_dimension_ti,
        fractal_prefactor_ti,
        gyration_radius_ti,
        radius_monomer_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_collision_radius_wq2022_rg_df_k0_a13", backend="taichi")
def ti_get_collision_radius_wq2022_rg_df_k0_a13(
    fractal_dimension,
    fractal_prefactor,
    shape_anisotropy,
    gyration_radius,
    radius_monomer
):
    """
    Taichi wrapper for the WQ2022 Rg-df-k0-a13 collision radius model.

    Arguments:
        - fractal_dimension : Scalar or 1D array of fractal dimensions (d_f).
        - fractal_prefactor : Scalar or 1D array of fractal prefactors (k₀).
        - shape_anisotropy : Scalar or 1D array of shape anisotropy (a₁₃).
        - gyration_radius : Scalar or 1D array of gyration radii (R_g).
        - radius_monomer : Scalar or 1D array of monomer radii (r_m).

    Returns:
        - Collision radius (R_c) as scalar or ndarray.

    Equation:
        R_c = (0.876 × d_f^0.363 × k₀^-0.105 × (R_g/r_m)
              + 0.421 × k₀ - 0.036 × a₁₃ - 0.227) × r_m

    Examples:
        ```py
        ti_get_collision_radius_wq2022_rg_df_k0_a13(
            np.array([1.8]),
            np.array([2.0]),
            np.array([0.5]),
            np.array([1.0]),
            np.array([0.1])
        )
        # Output: array([<float>])
        ```

    References:
        - Wang, Q. et al. (2022). "Title." Journal Name, Volume, Year.
    """
    _ensure_all_ndarrays(
        fractal_dimension,
        fractal_prefactor,
        shape_anisotropy,
        gyration_radius,
        radius_monomer
    )
    fractal_dimension_array = np.atleast_1d(fractal_dimension)
    fractal_prefactor_array = np.atleast_1d(fractal_prefactor)
    shape_anisotropy_array = np.atleast_1d(shape_anisotropy)
    gyration_radius_array = np.atleast_1d(gyration_radius)
    radius_monomer_array = np.atleast_1d(radius_monomer)
    _ensure_all_same_size(
        fractal_dimension_array,
        fractal_prefactor_array,
        shape_anisotropy_array,
        gyration_radius_array,
        radius_monomer_array
    )
    n_elements = fractal_dimension_array.size
    fractal_dimension_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_prefactor_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    shape_anisotropy_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    gyration_radius_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    radius_monomer_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    fractal_dimension_ti.from_numpy(fractal_dimension_array)
    fractal_prefactor_ti.from_numpy(fractal_prefactor_array)
    shape_anisotropy_ti.from_numpy(shape_anisotropy_array)
    gyration_radius_ti.from_numpy(gyration_radius_array)
    radius_monomer_ti.from_numpy(radius_monomer_array)
    kget_collision_radius_wq2022_rg_df_k0_a13(
        fractal_dimension_ti,
        fractal_prefactor_ti,
        shape_anisotropy_ti,
        gyration_radius_ti,
        radius_monomer_ti,
        result_ti
    )
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
