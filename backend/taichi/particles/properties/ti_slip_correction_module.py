
# 2 – required imports
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# 3 – element-wise Taichi function
@ti.func
def fget_slip_correction(kn: ti.f64) -> ti.f64:
    """Cunningham slip-correction factor (Taichi scalar)."""
    return 1.0 + kn * (1.257 + 0.4 * ti.exp(-1.1 / kn))

# 4 – vectorised kernel
@ti.kernel
def kget_slip_correction(
    kn: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Cunningham slip-correction factor (Taichi kernel)."""
    for i in range(res.shape[0]):
        res[i] = fget_slip_correction(kn[i])

# 5 – public wrapper, registered for the dispatcher
@register("get_cunningham_slip_correction", backend="taichi")
def ti_get_cunningham_slip_correction(kn):
    """Cunningham slip-correction factor (Taichi wrapper)."""
    if not isinstance(kn, np.ndarray):
        raise TypeError("Taichi backend expects NumPy arrays for the input.")

    kn_np = np.atleast_1d(kn).astype(np.float64)
    n = kn_np.size

    kn_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kn_ti.from_numpy(kn_np)

    kget_slip_correction(kn_ti, res_ti)

    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out
