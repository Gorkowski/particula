"""Benchmarks Python, Taichi wrapper, and raw kernel implementations of
get_diffusive_knudsen_number."""
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)
from particula.util.reduced_quantity import get_reduced_self_broadcast
from particula.particles.properties import coulomb_enhancement as ce

from particula.particles.properties.diffusive_knudsen_module import (
    get_diffusive_knudsen_number as py_func,
)
from particula.backend.taichi.particles.properties.ti_diffusive_knudsen_module import (
    ti_get_diffusive_knudsen_number as ti_func,
    kget_diffusive_knudsen_number   as ti_kernel,
)

RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(1, 4, 8, dtype=int)      # 10 … 10 000
ti.init(arch=ti.cpu)

def benchmark_diffusive_knudsen_number_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ---------------- random input data --------------------------
        r_arr      = rng.random(n, dtype=np.float64) * 3e-7 + 1e-8      # particle radius  (m)
        m_arr      = rng.random(n, dtype=np.float64) * 4e-17 + 1e-18    # particle mass    (kg)
        f_arr      = rng.random(n, dtype=np.float64) * 2.0   + 0.1      # friction factor  (-)
        phi_arr    = rng.random(n, dtype=np.float64) * 0.3              # coulomb φ/φ_T    (-)

        # prepare square 2-D buffers for kernel -----------------------
        # ‑ broadcast 1-D inputs into (n,n) reduced-pair matrices exactly
        #   as done inside the Taichi wrapper.
        sum_r_mat   = r_arr[:, None] + r_arr[None, :]          # rᵢ+rⱼ
        red_mass    = get_reduced_self_broadcast(m_arr)        # (n, n)
        red_fric    = get_reduced_self_broadcast(f_arr)        # (n, n)

        kin_1d  = ce.get_coulomb_kinetic_limit   (phi_arr)     # (n,)
        cont_1d = ce.get_coulomb_continuum_limit (phi_arr)     # (n,)
        kin_enh  = np.broadcast_to(np.atleast_2d(kin_1d ), (n, n)).astype(np.float64)
        cont_enh = np.broadcast_to(np.atleast_2d(cont_1d), (n, n)).astype(np.float64)
        temp_scalar = 298.15

        # ---- Taichi device buffers (re-create per size) -------------
        def make_buf(arr):
            buf = ti.ndarray(dtype=ti.f64, shape=arr.shape)
            buf.from_numpy(arr)
            return buf

        sum_r_ti    = make_buf(sum_r_mat)
        red_mass_ti = make_buf(red_mass)
        red_fric_ti = make_buf(red_fric)
        cont_enh_ti = make_buf(cont_enh)
        kin_enh_ti  = make_buf(kin_enh)
        res_ti      = ti.ndarray(dtype=ti.f64, shape=sum_r_mat.shape)

        # ---------------- timing -------------------------------------
        ops = n * n     # one value per matrix element
        stats_py     = get_function_benchmark(
            lambda: py_func(r_arr, m_arr, f_arr, phi_arr, temp_scalar), ops_per_call=ops
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(r_arr, m_arr, f_arr, phi_arr, temp_scalar), ops_per_call=ops
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(
                sum_r_ti, red_mass_ti, red_fric_ti,
                cont_enh_ti, kin_enh_ti, temp_scalar, res_ti
            ),
            ops_per_call=ops,
        )

        # collect CSV row ---------------------------------------------
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # --------------- header construction -----------------------------
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # --------------- output directory --------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --------------- CSV ---------------------------------------------
    csv_path = os.path.join(out_dir, "diffusive_knudsen_number_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # --------------- system info JSON --------------------------------
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # --------------- throughput plot ---------------------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        "Diffusive Knudsen-number throughput benchmark",
        os.path.join(out_dir, "diffusive_knudsen_number_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_diffusive_knudsen_number_csv()
