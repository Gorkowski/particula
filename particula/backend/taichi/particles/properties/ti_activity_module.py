import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_surface_partial_pressure(
    pure_vapor_pressure: ti.f64, activity: ti.f64
) -> ti.f64:
    return pure_vapor_pressure * activity

@ti.func
def fget_ideal_activity_mass(mass_single: ti.f64, total_mass: ti.f64) -> ti.f64:
    return 0.0 if total_mass == 0.0 else mass_single / total_mass

@ti.func
def fget_ideal_activity_volume(
    mass_single: ti.f64,
    density_single: ti.f64,
    total_volume: ti.f64
) -> ti.f64:
    return (
        0.0
        if total_volume == 0.0
        else (mass_single / density_single) / total_volume
    )

@ti.func
def fget_ideal_activity_molar(
    mass_single: ti.f64,
    molar_mass_single: ti.f64,
    total_moles: ti.f64
) -> ti.f64:
    return (
        0.0
        if total_moles == 0.0
        else (mass_single / molar_mass_single) / total_moles
    )

@ti.func
def fget_water_activity_from_kappa_row(
    mass_concentration: ti.template(),  # 2-D mass-concentration field
    kappa: ti.template(),               # 1-D κ array
    density: ti.template(),             # 1-D density array
    molar_mass: ti.template(),          # 1-D molar-mass array
    water_index: ti.i32,                # column that corresponds to water
    row_index: ti.i32,                  # current particle / mixture index
    n_species: ti.i32,                  # number of species (= mass_concentration.shape[1])
) -> ti.f64:
    """
    Compute the water activity (a_w) for a single mixture (row) using the
    κ–Köhler mixing rule.

    Arguments:
        - mass_concentration : 2-D field of species mass concentrations
          [kg m⁻³] (rows = mixtures, cols = species).
        - kappa : 1-D array of κ hygroscopicity coefficients
          (dimensionless).
        - density : 1-D array of species densities [kg m⁻³].
        - molar_mass : 1-D array of species molar masses [kg mol⁻¹].
        - water_index : Column index that corresponds to water.
        - row_index : Row being evaluated (mixture index).
        - n_species : Total number of species
          (= mass_concentration.shape[1]).

    Returns:
        - Water activity a_w for the selected row (dimensionless).

    References:
        - Petters & Kreidenweis, “A single parameter representation of
          hygroscopic growth and cloud condensation nucleus activity,”
          Atmos. Chem. Phys., 7 (2007).
    """
    # ---- volume fractions ----------------------------------------
    volume_sum = 0.0
    for s in range(n_species):
        volume_sum += mass_concentration[row_index, s] / density[s]

    water_volume_fraction = 0.0
    if volume_sum > 0.0:
        water_volume_fraction = (
            mass_concentration[row_index, water_index] / density[water_index]
        ) / volume_sum

    solute_volume_fraction = 1.0 - water_volume_fraction

    # ---- bulk κ of the (non-water) solute phase ------------------
    bulk_kappa = 0.0
    if solute_volume_fraction > 0.0:
        if n_species == 2:  # single-solute shortcut
            solute_index = 1 if water_index == 0 else 0
            bulk_kappa = kappa[solute_index]
        else:
            for s in range(n_species):
                if s != water_index:
                    volume_fraction_species = (
                        (mass_concentration[row_index, s] / density[s]) / volume_sum
                    )
                    bulk_kappa += (
                        (volume_fraction_species / solute_volume_fraction) * kappa[s]
                    )

    # ---- convert κ + φ_w to activity -----------------------------
    volume_term = 0.0
    if water_volume_fraction > 0.0:
        volume_term = (
            bulk_kappa * solute_volume_fraction / water_volume_fraction
        )

    return (
        0.0
        if water_volume_fraction == 0.0
        else 1.0 / (1.0 + volume_term)
    )

# 1-D ---------------------------------------------------------------
@ti.kernel
def kget_surface_partial_pressure(
    pure_vapor_pressure_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    activity_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(result_array.shape[0]):
        result_array[i] = fget_surface_partial_pressure(
            pure_vapor_pressure_array[i], activity_array[i]
        )

# 2-D rows = particles / mixtures ; cols = species  -----------------
@ti.kernel
def kget_ideal_activity_mass(
    mass_concentration_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    for i in range(mass_concentration_array.shape[0]):
        row_sum = 0.0
        for s in range(mass_concentration_array.shape[1]):
            row_sum += mass_concentration_array[i, s]
        for s in range(mass_concentration_array.shape[1]):
            result_array[i, s] = fget_ideal_activity_mass(
                mass_concentration_array[i, s], row_sum
            )

@ti.kernel
def kget_ideal_activity_volume(
    mass_concentration_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
    density_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    n_species = mass_concentration_array.shape[1]
    for i in range(mass_concentration_array.shape[0]):
        volume_sum = 0.0
        for s in range(n_species):
            volume_sum += mass_concentration_array[i, s] / density_array[s]
        for s in range(n_species):
            result_array[i, s] = fget_ideal_activity_volume(
                mass_concentration_array[i, s], density_array[s], volume_sum
            )

@ti.kernel
def kget_ideal_activity_molar(
    mass_concentration_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
    molar_mass_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    n_species = mass_concentration_array.shape[1]
    for i in range(mass_concentration_array.shape[0]):
        mole_sum = 0.0
        for s in range(n_species):
            mole_sum += mass_concentration_array[i, s] / molar_mass_array[s]
        for s in range(n_species):
            result_array[i, s] = fget_ideal_activity_molar(
                mass_concentration_array[i, s], molar_mass_array[s], mole_sum
            )

@ti.kernel
def kget_kappa_activity(
    mass_concentration_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
    kappa_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    density_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    molar_mass_array: ti.types.ndarray(dtype=ti.f64, ndim=1),
    water_index: ti.i32,
    result_array: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    n_species = mass_concentration_array.shape[1]
    for i in range(mass_concentration_array.shape[0]):

        # mole-fraction part (all species first)
        mole_sum = 0.0
        for s in range(n_species):
            mole_sum += mass_concentration_array[i, s] / molar_mass_array[s]
        for s in range(n_species):
            moles_of_species = mass_concentration_array[i, s] / molar_mass_array[s]
            result_array[i, s] = (
                0.0 if mole_sum == 0.0 else moles_of_species / mole_sum
            )

        # water activity via κ–Köhler mix
        water_activity = fget_water_activity_from_kappa_row(
            mass_concentration_array,
            kappa_array,
            density_array,
            molar_mass_array,
            water_index,
            i,
            n_species,
        )
        result_array[i, water_index] = water_activity

@register("get_surface_partial_pressure", backend="taichi")
def ti_get_surface_partial_pressure(pure_vapor_pressure, activity):
    """
    Vectorised Taichi backend for
    get_surface_partial_pressure(…).

    Calculates Pₛ = P₀ × a element-wise.

    Arguments:
        - pure_vapor_pressure : Scalar or array-like of pure vapor
          pressure [Pa].
        - activity : Scalar or array-like activity (dimensionless).

    Returns:
        - Surface partial pressure with the same shape as the inputs
          [Pa].

    Examples:
        ```py
        p_surface = ti_get_surface_partial_pressure(
            pure_vapor_pressure=[100.0, 200.0],
            activity=[0.8, 0.6]
        )
        ```
    """
    if isinstance(pure_vapor_pressure, float):
        return pure_vapor_pressure * activity  # scalar shortcut

    pure_vapor_pressure_np = np.atleast_1d(pure_vapor_pressure)
    activity_np = np.atleast_1d(activity)
    n_elements = pure_vapor_pressure_np.size
    pure_vapor_pressure_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    activity_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n_elements)
    pure_vapor_pressure_ti.from_numpy(pure_vapor_pressure_np)
    activity_ti.from_numpy(activity_np)
    kget_surface_partial_pressure(
        pure_vapor_pressure_ti, activity_ti, result_ti
    )
    output = result_ti.to_numpy()
    return output.item() if output.size == 1 else output

@register("get_ideal_activity_mass", backend="taichi")
def ti_get_ideal_activity_mass(mass_concentration):
    """
    Vectorised Taichi backend for get_ideal_activity_mass(…).

    Computes the ideal activity for each species in a mixture based on
    mass fraction.

    Arguments:
        - mass_concentration : Scalar, 1-D, or 2-D array of species
          mass concentrations [kg m⁻³].

    Returns:
        - Array of ideal activities (dimensionless), same shape as input.

    Examples:
        ```py
        a_mass = ti_get_ideal_activity_mass(
            mass_concentration=[[1.0, 2.0], [3.0, 4.0]]
        )
        ```
    """
    if isinstance(mass_concentration, float):
        return 1.0
    mass_concentration_np = np.atleast_2d(mass_concentration)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mass_concentration_ti = ti.ndarray(
        dtype=ti.f64, shape=mass_concentration_np.shape
    )
    result_ti = ti.ndarray(dtype=ti.f64, shape=mass_concentration_np.shape)
    mass_concentration_ti.from_numpy(mass_concentration_np)
    kget_ideal_activity_mass(mass_concentration_ti, result_ti)
    output = result_ti.to_numpy()
    if single_row:
        return output[0]
    return output

@register("get_ideal_activity_volume", backend="taichi")
def ti_get_ideal_activity_volume(mass_concentration, density):
    """
    Vectorised Taichi backend for get_ideal_activity_volume(…).

    Computes the ideal activity for each species in a mixture based on
    volume fraction.

    Arguments:
        - mass_concentration : Scalar, 1-D, or 2-D array of species
          mass concentrations [kg m⁻³].
        - density : 1-D array of species densities [kg m⁻³].

    Returns:
        - Array of ideal activities (dimensionless), same shape as input.

    Examples:
        ```py
        a_vol = ti_get_ideal_activity_volume(
            mass_concentration=[[1.0, 2.0], [3.0, 4.0]],
            density=[1000.0, 1200.0]
        )
        ```
    """
    if isinstance(mass_concentration, float):
        return 1.0
    mass_concentration_np = np.atleast_2d(mass_concentration)
    density_np = np.atleast_1d(density)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mass_concentration_ti = ti.ndarray(
        dtype=ti.f64, shape=mass_concentration_np.shape
    )
    density_ti = ti.ndarray(dtype=ti.f64, shape=density_np.shape)
    result_ti = ti.ndarray(dtype=ti.f64, shape=mass_concentration_np.shape)
    mass_concentration_ti.from_numpy(mass_concentration_np)
    density_ti.from_numpy(density_np)
    kget_ideal_activity_volume(
        mass_concentration_ti, density_ti, result_ti
    )
    output = result_ti.to_numpy()
    if single_row:
        return output[0]
    return output

@register("get_ideal_activity_molar", backend="taichi")
def ti_get_ideal_activity_molar(mass_concentration, molar_mass):
    """
    Vectorised Taichi backend for get_ideal_activity_molar(…).

    Computes the ideal activity for each species in a mixture based on
    molar fraction.

    Arguments:
        - mass_concentration : Scalar, 1-D, or 2-D array of species
          mass concentrations [kg m⁻³].
        - molar_mass : 1-D array of species molar masses [kg mol⁻¹].

    Returns:
        - Array of ideal activities (dimensionless), same shape as input.

    Examples:
        ```py
        a_mol = ti_get_ideal_activity_molar(
            mass_concentration=[[1.0, 2.0], [3.0, 4.0]],
            molar_mass=[0.018, 0.044]
        )
        ```
    """
    if isinstance(mass_concentration, float):
        return 1.0
    mass_concentration_np = np.atleast_2d(mass_concentration)
    molar_mass_np = np.atleast_1d(molar_mass)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mass_concentration_ti = ti.ndarray(
        dtype=ti.f64, shape=mass_concentration_np.shape
    )
    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=molar_mass_np.shape)
    result_ti = ti.ndarray(dtype=ti.f64, shape=mass_concentration_np.shape)
    mass_concentration_ti.from_numpy(mass_concentration_np)
    molar_mass_ti.from_numpy(molar_mass_np)
    kget_ideal_activity_molar(
        mass_concentration_ti, molar_mass_ti, result_ti
    )
    output = result_ti.to_numpy()
    if single_row:
        return output[0]
    return output

@register("get_kappa_activity", backend="taichi")
def ti_get_kappa_activity(
    mass_concentration, kappa, density, molar_mass, water_index
):
    """
    Vectorised Taichi backend for get_kappa_activity(…).

    Computes the kappa-Köhler water activity and mole fractions for
    each mixture row.

    Arguments:
        - mass_concentration : Scalar, 1-D, or 2-D array of species
          mass concentrations [kg m⁻³].
        - kappa : 1-D array of κ hygroscopicity coefficients
          (dimensionless).
        - density : 1-D array of species densities [kg m⁻³].
        - molar_mass : 1-D array of species molar masses [kg mol⁻¹].
        - water_index : Index of the water species column.

    Returns:
        - Array of activities (dimensionless), same shape as input.

    Examples:
        ```py
        a_kappa = ti_get_kappa_activity(
            mass_concentration=[[1.0, 2.0], [3.0, 4.0]],
            kappa=[0.3, 0.0],
            density=[1000.0, 1200.0],
            molar_mass=[0.018, 0.044],
            water_index=0
        )
        ```
    """
    if isinstance(mass_concentration, float):
        return 1.0
    mass_concentration_np = np.atleast_2d(mass_concentration)
    kappa_np = np.atleast_1d(kappa)
    density_np = np.atleast_1d(density)
    molar_mass_np = np.atleast_1d(molar_mass)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mass_concentration_ti = ti.ndarray(
        dtype=ti.f64, shape=mass_concentration_np.shape
    )
    kappa_ti = ti.ndarray(dtype=ti.f64, shape=kappa_np.shape)
    density_ti = ti.ndarray(dtype=ti.f64, shape=density_np.shape)
    molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=molar_mass_np.shape)
    result_ti = ti.ndarray(dtype=ti.f64, shape=mass_concentration_np.shape)
    mass_concentration_ti.from_numpy(mass_concentration_np)
    kappa_ti.from_numpy(kappa_np)
    density_ti.from_numpy(density_np)
    molar_mass_ti.from_numpy(molar_mass_np)
    kget_kappa_activity(
        mass_concentration_ti,
        kappa_ti,
        density_ti,
        molar_mass_ti,
        int(water_index),
        result_ti,
    )
    output = result_ti.to_numpy()
    if single_row:
        return output[0]
    return output
