"""Taichi implementation of get_mass_fractions_from_moles."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register


@ti.func
def fget_mass_fraction(partial_mass: ti.f64, total_mass: ti.f64) -> ti.f64:
    """Return wᵢ = (xᵢ·Mᵢ)/Σ(x·M); 0 when Σ(x·M)=0."""
    return 0.0 if total_mass == 0.0 else partial_mass / total_mass


@ti.kernel
def kget_mass_fractions_1d(                       # 1-D wrapper kernel
    mole_fraction: ti.types.ndarray(dtype=ti.f64, ndim=1),     # xᵢ
    molecular_weight: ti.types.ndarray(dtype=ti.f64, ndim=1),  # Mᵢ
    mass_fraction: ti.types.ndarray(dtype=ti.f64, ndim=1),     # wᵢ
):
    # first pass – Σ(x·M)
    total_weighted_mass = ti.f64(0.0)
    for i in range(mass_fraction.shape[0]):
        total_weighted_mass += mole_fraction[i] * molecular_weight[i]
    # second pass – wᵢ
    for i in range(mass_fraction.shape[0]):
        mass_fraction[i] = fget_mass_fraction(mole_fraction[i] * molecular_weight[i], total_weighted_mass)


@ti.kernel
def kget_mass_fractions_2d(                       # 2-D wrapper kernel (row-wise)
    mole_fraction: ti.types.ndarray(dtype=ti.f64, ndim=2),     # xᵢⱼ
    molecular_weight: ti.types.ndarray(dtype=ti.f64, ndim=1),  # Mⱼ
    mass_fraction: ti.types.ndarray(dtype=ti.f64, ndim=2),     # wᵢⱼ
):
    rows = mass_fraction.shape[0]
    cols = mass_fraction.shape[1]
    for r in range(rows):
        total_weighted_mass = ti.f64(0.0)
        for c in range(cols):
            total_weighted_mass += mole_fraction[r, c] * molecular_weight[c]
        for c in range(cols):
            mass_fraction[r, c] = fget_mass_fraction(mole_fraction[r, c] * molecular_weight[c], total_weighted_mass)


@register("get_mass_fractions_from_moles", backend="taichi")
def ti_get_mass_fractions_from_moles(mole_fractions, molecular_weights):
    """Taichi wrapper for get_mass_fractions_from_moles."""
    # 5 a – type guard
    if not (isinstance(mole_fractions, np.ndarray)
            and isinstance(molecular_weights, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    mole_fraction_array = np.asarray(mole_fractions, dtype=np.float64)
    molecular_weight_array = np.asarray(molecular_weights, dtype=np.float64)

    if mole_fraction_array.shape[-1] != molecular_weight_array.shape[-1]:
        raise ValueError("Last dimension of inputs must match.")

    # 1-D case
    if mole_fraction_array.ndim == 1:
        n = mole_fraction_array.size
        mole_fraction_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        molecular_weight_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mass_fraction_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mole_fraction_ti.from_numpy(mole_fraction_array)
        molecular_weight_ti.from_numpy(molecular_weight_array)
        kget_mass_fractions_1d(mole_fraction_ti, molecular_weight_ti, mass_fraction_ti)
        result_array = mass_fraction_ti.to_numpy()
        return result_array.item() if result_array.size == 1 else result_array

    # 2-D case
    if mole_fraction_array.ndim == 2:
        rows, cols = mole_fraction_array.shape
        mole_fraction_ti  = ti.ndarray(dtype=ti.f64, shape=(rows, cols))
        molecular_weight_ti = ti.ndarray(dtype=ti.f64, shape=cols)
        mass_fraction_ti = ti.ndarray(dtype=ti.f64, shape=(rows, cols))
        mole_fraction_ti.from_numpy(mole_fraction_array)
        molecular_weight_ti.from_numpy(molecular_weight_array)
        kget_mass_fractions_2d(mole_fraction_ti, molecular_weight_ti, mass_fraction_ti)
        result_array = mass_fraction_ti.to_numpy()
        return result_array.item() if result_array.size == 1 else result_array

    raise ValueError("mole_fractions must be 1-D or 2-D.")
