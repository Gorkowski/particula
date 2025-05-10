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
    mole: ti.types.ndarray(dtype=ti.f64, ndim=1),     # xᵢ
    mw: ti.types.ndarray(dtype=ti.f64, ndim=1),       # Mᵢ
    out: ti.types.ndarray(dtype=ti.f64, ndim=1),      # wᵢ
):
    # first pass – Σ(x·M)
    tot = ti.f64(0.0)
    for i in range(out.shape[0]):
        tot += mole[i] * mw[i]
    # second pass – wᵢ
    for i in range(out.shape[0]):
        out[i] = fget_mass_fraction(mole[i] * mw[i], tot)


@ti.kernel
def kget_mass_fractions_2d(                       # 2-D wrapper kernel (row-wise)
    mole: ti.types.ndarray(dtype=ti.f64, ndim=2),     # xᵢⱼ
    mw: ti.types.ndarray(dtype=ti.f64, ndim=1),       # Mⱼ
    out: ti.types.ndarray(dtype=ti.f64, ndim=2),      # wᵢⱼ
):
    rows, cols = out.shape
    for r in range(rows):
        tot = ti.f64(0.0)
        for c in range(cols):
            tot += mole[r, c] * mw[c]
        for c in range(cols):
            out[r, c] = fget_mass_fraction(mole[r, c] * mw[c], tot)


@register("get_mass_fractions_from_moles", backend="taichi")
def ti_get_mass_fractions_from_moles(mole_fractions, molecular_weights):
    """Taichi wrapper for get_mass_fractions_from_moles."""
    # 5 a – type guard
    if not (isinstance(mole_fractions, np.ndarray)
            and isinstance(molecular_weights, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    x = np.asarray(mole_fractions, dtype=np.float64)
    mw = np.asarray(molecular_weights, dtype=np.float64)

    if x.shape[-1] != mw.shape[-1]:
        raise ValueError("Last dimension of inputs must match.")

    # 1-D case
    if x.ndim == 1:
        n = x.size
        x_ti, mw_ti, out_ti = [ti.ndarray(ti.f64, shape=n) for _ in range(3)]
        x_ti.from_numpy(x)
        mw_ti.from_numpy(mw)
        kget_mass_fractions_1d(x_ti, mw_ti, out_ti)
        return out_ti.to_numpy()

    # 2-D case
    if x.ndim == 2:
        rows, cols = x.shape
        x_ti = ti.ndarray(ti.f64, shape=(rows, cols))
        mw_ti = ti.ndarray(ti.f64, shape=cols)
        out_ti = ti.ndarray(ti.f64, shape=(rows, cols))
        x_ti.from_numpy(x)
        mw_ti.from_numpy(mw)
        kget_mass_fractions_2d(x_ti, mw_ti, out_ti)
        return out_ti.to_numpy()

    raise ValueError("mole_fractions must be 1-D or 2-D.")
