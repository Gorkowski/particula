from particula.backend.dispatch_register import register
import taichi as ti
import numpy as np

# ---------- 3. element-wise Taichi helpers ----------
@ti.func
def fget_distribution_in_dn(          # scalar version
    delta: ti.f64,
    diameter: ti.f64,
    dn_val: ti.f64,
    inverse_flag: ti.i32,             # 0 → dn/dlogdp→d_num, 1 → inverse
) -> ti.f64:
    lower = diameter - 0.5 * delta
    upper = diameter + 0.5 * delta
    log_factor = ti.log(upper / lower) / ti.log(10.0)
    return ti.select(inverse_flag == 1,
                     dn_val / log_factor,
                     dn_val * log_factor)

@ti.func
def fget_pdf_distribution_in_pmf(     # scalar version
    delta: ti.f64,
    y_val: ti.f64,
    to_pdf_flag: ti.i32,              # 1 → PMF→PDF , 0 → PDF→PMF
) -> ti.f64:
    return ti.select(to_pdf_flag == 1,
                     y_val / delta,
                     y_val * delta)

# ---------- 4. vectorised kernels ----------
@ti.kernel
def kget_distribution_in_dn(                     # noqa: N802
    diameter: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dn_arr: ti.types.ndarray(dtype=ti.f64, ndim=1),
    inverse_flag: ti.i32,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    n = result.shape[0]
    for i in range(n):

        delta = ti.select(
            i < n - 1,
            diameter[i + 1] - diameter[i],
            (diameter[i] - diameter[i - 1]) ** 2 / (diameter[i - 1] - diameter[i - 2]),
        )
        result[i] = fget_distribution_in_dn(delta, diameter[i], dn_arr[i], inverse_flag)


@ti.kernel
def kget_pdf_distribution_in_pmf(               # noqa: N802
    x_arr: ti.types.ndarray(dtype=ti.f64, ndim=1),
    y_arr: ti.types.ndarray(dtype=ti.f64, ndim=1),
    to_pdf_flag: ti.i32,
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    n = result.shape[0]
    for i in range(n):

        delta = ti.select(
            i < n - 1,
            x_arr[i + 1] - x_arr[i],
            (x_arr[i] - x_arr[i - 1]) ** 2 / (x_arr[i - 1] - x_arr[i - 2]),
        )
        result[i] = fget_pdf_distribution_in_pmf(delta, y_arr[i], to_pdf_flag)

# ---------- 5. public wrappers ----------
@register("get_distribution_in_dn", backend="taichi")
def ti_get_distribution_in_dn(diameter, dn_dlogdp, inverse=False):
    if not (isinstance(diameter, np.ndarray) and isinstance(dn_dlogdp, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    d, dn = np.atleast_1d(diameter), np.atleast_1d(dn_dlogdp)
    if d.shape != dn.shape:
        raise ValueError("diameter and dn_dlogdp must have identical shape.")
    n = d.size
    d_ti = ti.ndarray(dtype=ti.f64, shape=n); d_ti.from_numpy(d)
    dn_ti = ti.ndarray(dtype=ti.f64, shape=n); dn_ti.from_numpy(dn)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kget_distribution_in_dn(d_ti, dn_ti, int(bool(inverse)), res_ti)
    res = res_ti.to_numpy()
    return res.item() if res.size == 1 else res


@register("get_pdf_distribution_in_pmf", backend="taichi")
def ti_get_pdf_distribution_in_pmf(x_array, distribution, to_pdf=True):
    if not (isinstance(x_array, np.ndarray) and isinstance(distribution, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    x, y = np.atleast_1d(x_array), np.atleast_1d(distribution)
    if x.shape != y.shape:
        raise ValueError("x_array and distribution must have identical shape.")
    n = x.size
    x_ti = ti.ndarray(dtype=ti.f64, shape=n); x_ti.from_numpy(x)
    y_ti = ti.ndarray(dtype=ti.f64, shape=n); y_ti.from_numpy(y)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    kget_pdf_distribution_in_pmf(x_ti, y_ti, int(bool(to_pdf)), res_ti)
    res = res_ti.to_numpy()
    return res.item() if res.size == 1 else res

# ---------- 6. (nothing to add here – __init__ updated below)
# ---------- 7. tests added in separate file
# ---------- 8. uses ti.f64 throughout
# ---------- 9. naming & style follow guide
