"""
Benchmarks the reference Python, Taichi wrapper and raw Taichi kernels for
get_kelvin_radius()  (1-D) and get_kelvin_term() (2-D).
"""
# -- standard & Particula imports ------------------------------------------
import os
import json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

# -- functions to benchmark ------------------------------------------------
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius as py_get_kelvin_radius,
    get_kelvin_term   as py_get_kelvin_term,
)
from particula.backend.taichi.particles.properties.ti_kelvin_effect_module import (
    ti_get_kelvin_radius,
    ti_get_kelvin_term,
    kget_kelvin_radius,
    kget_kelvin_term,
)

# -- benchmark configuration ----------------------------------------------
RNG_SEED          = 42
ARRAY_LENGTHS_1D  = np.logspace(2, 7, 6, dtype=int)   # 10² … 10⁷   (1-D)
ARRAY_LENGTHS_2D  = np.logspace(2, 5, 4, dtype=int)   # 10² … 10⁵   (2-D)
ti.init(arch=ti.cpu)                                  # CPU backend

# ------------------------------------------------------------------------- #
def benchmark_kelvin_radius_csv() -> None:
    """Benchmark get_kelvin_radius() over ARRAY_LENGTHS_1D and save CSV."""
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS_1D:
        # -------- random positive input data ------------------------------
        st = rng.random(n, dtype=np.float64)*0.02 + 0.05      # 0.05 … 0.07 N m⁻¹
        de = rng.random(n, dtype=np.float64)*500   + 500.0    # 500 … 1000 kg m⁻³
        mm = rng.random(n, dtype=np.float64)*0.01  + 0.018    # 0.018 … 0.028 kg mol⁻¹
        T  = 298.15                                          # constant (K)

        # -------- Taichi buffers ------------------------------------------
        st_ti = ti.ndarray(dtype=ti.f64, shape=n); st_ti.from_numpy(st)
        de_ti = ti.ndarray(dtype=ti.f64, shape=n); de_ti.from_numpy(de)
        mm_ti = ti.ndarray(dtype=ti.f64, shape=n); mm_ti.from_numpy(mm)
        rk_ti = ti.ndarray(dtype=ti.f64, shape=n)

        # -------- timing --------------------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_get_kelvin_radius(st, de, mm, T), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_kelvin_radius(st, de, mm, T), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_kelvin_radius(st_ti, de_ti, mm_ti, T, rk_ti),
            ops_per_call=n,
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ------------- header & file output -----------------------------------
    py_hdr   = ["python_"        + h for h in stats_py["array_headers"]]
    ti_hdr   = ["taichi_"        + h for h in stats_ti["array_headers"]]
    k_hdr    = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header   = ["array_length", *py_hdr, *ti_hdr, *k_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "kelvin_radius_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "Kelvin-radius throughput benchmark",
        os.path.join(out_dir, "kelvin_radius_benchmark.png"),
    )

# ------------------------------------------------------------------------- #
def benchmark_kelvin_term_csv() -> None:
    """Benchmark get_kelvin_term() over ARRAY_LENGTHS_2D and save CSV."""
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS_2D:
        # -------- random positive input data ------------------------------
        pr = rng.random(n, dtype=np.float64)*9e-8 + 1e-8      # 1e-8 … 1e-7 m
        kr = rng.random(n, dtype=np.float64)*9e-8 + 1e-8
        pr_mat = np.broadcast_to(pr[:, None], (n, n))
        kr_mat = np.broadcast_to(kr[None, :], (n, n))

        # -------- Taichi buffers ------------------------------------------
        pr_ti = ti.ndarray(dtype=ti.f64, shape=pr_mat.shape); pr_ti.from_numpy(pr_mat)
        kr_ti = ti.ndarray(dtype=ti.f64, shape=kr_mat.shape); kr_ti.from_numpy(kr_mat)
        res_ti = ti.ndarray(dtype=ti.f64, shape=pr_mat.shape)

        # -------- timing --------------------------------------------------
        ops = n * n  # one operation per matrix element
        stats_py     = get_function_benchmark(
            lambda: py_get_kelvin_term(pr, kr),               ops_per_call=ops
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_kelvin_term(pr, kr),               ops_per_call=ops
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_kelvin_term(pr_ti, kr_ti, res_ti),   ops_per_call=ops
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ------------- header & file output -----------------------------------
    py_hdr   = ["python_"        + h for h in stats_py["array_headers"]]
    ti_hdr   = ["taichi_"        + h for h in stats_ti["array_headers"]]
    k_hdr    = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header   = ["array_length", *py_hdr, *ti_hdr, *k_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "kelvin_term_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "Kelvin-term throughput benchmark",
        os.path.join(out_dir, "kelvin_term_benchmark.png"),
    )

# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_kelvin_radius_csv()
    benchmark_kelvin_term_csv()
