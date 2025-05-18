"""Taichi implementation of get_mass_fractions_from_moles."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register


@ti.func
def fget_mass_fraction(partial_mass: ti.f64, total_mass: ti.f64) -> ti.f64:
    """
    Compute the mass fraction wᵢ of a single component.

    wᵢ = (xᵢ · Mᵢ) ∕ Σ(x · M); returns 0 when Σ(x · M)=0.

    Arguments:
        - partial_mass : xᵢ · Mᵢ [kg mol⁻¹].
        - total_mass   : Σ(x · M)  [kg mol⁻¹].

    Returns:
        - wᵢ : Mass fraction of the component (0 ≤ wᵢ ≤ 1).

    References:
        - "Mass fraction", Wikipedia.
    """
    return 0.0 if total_mass == 0.0 else partial_mass / total_mass


@ti.kernel
def kget_mass_fractions_1d(
    mole_fraction: ti.types.ndarray(dtype=ti.f64, ndim=1),     # xᵢ
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),        # Mᵢ
    mass_fraction: ti.types.ndarray(dtype=ti.f64, ndim=1),     # wᵢ
):
    """
    Compute mass fractions from mole fractions and molar masses (1-D).

    Arguments:
        - mole_fraction : 1-D ndarray of mole fractions xᵢ.
        - molar_mass   : 1-D ndarray of molar masses Mᵢ [kg mol⁻¹].
        - mass_fraction : 1-D ndarray for output wᵢ (in-place).

    Returns:
        - None (in-place update).

    Examples:
        ```py
        # x = [0.2, 0.8], M = [0.018, 0.044]
        # w ≈ [0.093, 0.907]
        ```
    """
    # first pass – Σ(x·M)
    total_weighted_mass = ti.f64(0.0)
    for i in range(mass_fraction.shape[0]):
        total_weighted_mass += mole_fraction[i] * molar_mass[i]
    # second pass – wᵢ
    for i in range(mass_fraction.shape[0]):
        mass_fraction[i] = fget_mass_fraction(
            mole_fraction[i] * molar_mass[i], total_weighted_mass
        )


@ti.kernel
def kget_mass_fractions_2d(
    mole_fraction: ti.types.ndarray(dtype=ti.f64, ndim=2),     # xᵢⱼ
    molar_mass: ti.types.ndarray(dtype=ti.f64, ndim=1),        # Mⱼ
    mass_fraction: ti.types.ndarray(dtype=ti.f64, ndim=2),     # wᵢⱼ
):
    """
    Compute mass fractions from mole fractions and molar masses (2-D, row-wise).

    Arguments:
        - mole_fraction : 2-D ndarray of mole fractions xᵢⱼ.
        - molar_mass   : 1-D ndarray of molar masses Mⱼ [kg mol⁻¹].
        - mass_fraction : 2-D ndarray for output wᵢⱼ (in-place).

    Returns:
        - None (in-place update).

    Examples:
        ```py
        # x = [[0.2, 0.8], [0.5, 0.5]], M = [0.018, 0.044]
        # w ≈ [[0.093, 0.907], [0.290, 0.710]]
        ```
    """
    rows = mass_fraction.shape[0]
    cols = mass_fraction.shape[1]
    for r in range(rows):
        total_weighted_mass = ti.f64(0.0)
        for c in range(cols):
            total_weighted_mass += mole_fraction[r, c] * molar_mass[c]
        for c in range(cols):
            mass_fraction[r, c] = fget_mass_fraction(
                mole_fraction[r, c] * molar_mass[c], total_weighted_mass
            )


@register("get_mass_fractions_from_moles", backend="taichi")
def ti_get_mass_fractions_from_moles(mole_fractions, molar_masses):
    """
    Convert mole fractions to mass fractions (Taichi accelerated).

    The routine dispatches to 1-D or 2-D Taichi kernels, returning a
    NumPy array of the same shape as `mole_fractions`.

    Arguments:
        - mole_fractions : 1-D/2-D ndarray of mole fractions xᵢⱼ.
        - molar_masses   : 1-D ndarray of molar mass Mⱼ [kg mol⁻¹].

    Returns:
        - mass_fractions : ndarray of wᵢⱼ with same shape as input.

    Examples:
        ```py
        w = ti_get_mass_fractions_from_moles(np.array([0.2, 0.8]),
                                             np.array([0.018, 0.044]))
        # w ≈ array([0.093, 0.907])
        ```

    References:
        - P. Atkins & J. de Paula, *Physical Chemistry*, 11 ed., 2018.
    """
    # 5 a – type guard
    if not (isinstance(mole_fractions, np.ndarray)
            and isinstance(molar_masses, np.ndarray)):
        raise TypeError(
            "Taichi backend expects NumPy arrays for both inputs: "
            "mole fractions and molar masses."
        )

    mole_fraction_array = np.asarray(mole_fractions, dtype=np.float64)
    molar_mass_array = np.asarray(molar_masses, dtype=np.float64)

    if mole_fraction_array.shape[-1] != molar_mass_array.shape[-1]:
        raise ValueError("Last dimension of inputs must match.")

    # 1-D case
    if mole_fraction_array.ndim == 1:
        n = mole_fraction_array.size
        mole_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n)
        molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mass_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mole_fraction_ti.from_numpy(mole_fraction_array)
        molar_mass_ti.from_numpy(molar_mass_array)
        kget_mass_fractions_1d(mole_fraction_ti, molar_mass_ti, mass_fraction_ti)
        result_array = mass_fraction_ti.to_numpy()
        return result_array.item() if result_array.size == 1 else result_array

    # 2-D case
    if mole_fraction_array.ndim == 2:
        rows, cols = mole_fraction_array.shape
        mole_fraction_ti = ti.ndarray(dtype=ti.f64, shape=(rows, cols))
        molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=cols)
        mass_fraction_ti = ti.ndarray(dtype=ti.f64, shape=(rows, cols))
        mole_fraction_ti.from_numpy(mole_fraction_array)
        molar_mass_ti.from_numpy(molar_mass_array)
        kget_mass_fractions_2d(mole_fraction_ti, molar_mass_ti, mass_fraction_ti)
        result_array = mass_fraction_ti.to_numpy()
        return result_array.item() if result_array.size == 1 else result_array

    raise ValueError("mole_fractions must be 1-D or 2-D.")
