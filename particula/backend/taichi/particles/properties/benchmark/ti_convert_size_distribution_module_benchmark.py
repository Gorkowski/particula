"""Benchmarks convert-size-distribution routines (Python vs. Taichi)."""
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

# reference Python implementations
from particula.particles.properties.convert_size_distribution import (
    get_distribution_in_dn   as py_get_distribution_in_dn,
    get_pdf_distribution_in_pmf as py_get_pdf_distribution_in_pmf,
)

# Taichi wrapper functions + raw kernels
from particula.backend.taichi.particles.properties.ti_convert_size_distribution_module import (
    ti_get_distribution_in_dn,
    ti_get_pdf_distribution_in_pmf,
    kget_distribution_in_dn,
    kget_pdf_distribution_in_pmf,
)

# reproducibility
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_distribution_in_dn_csv() -> None:
    """Time pure-Python, Taichi wrapper, and raw kernel for get_distribution_in_dn."""
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ---- random inputs -------------------------------------------------
        diameter = rng.random(n, dtype=np.float64) * 9.9e-7 + 1e-8   # 1e-8 … 1e-6
        dn_dlog  = rng.random(n, dtype=np.float64) * 9.9e4 + 1e4     # 1e4 … 1e5

        # ---- Taichi buffers ------------------------------------------------
        d_ti  = ti.ndarray(dtype=ti.f64, shape=n); d_ti.from_numpy(diameter)
        dn_ti = ti.ndarray(dtype=ti.f64, shape=n); dn_ti.from_numpy(dn_dlog)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)

        # ---- timing --------------------------------------------------------
        stats_py = get_function_benchmark(
            lambda: py_get_distribution_in_dn(diameter, dn_dlog), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_get_distribution_in_dn(diameter, dn_dlog), ops_per_call=n
        )
        stats_ke = get_function_benchmark(
            lambda: kget_distribution_in_dn(d_ti, dn_ti, 0, res_ti), ops_per_call=n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_ke["array_stats"],
        ])

    # ---- header ------------------------------------------------------------
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_ke["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ---- output ------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_combined_csv(os.path.join(out_dir, "distribution_in_dn_benchmark.csv"),
                      header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "distribution_in_dn throughput benchmark",
        os.path.join(out_dir, "distribution_in_dn_benchmark.png"),
    )

def benchmark_pdf_from_pmf_csv() -> None:
    """Time pure-Python, Taichi wrapper, and raw kernel for get_pdf_distribution_in_pmf."""
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ---- random inputs -------------------------------------------------
        x_arr = rng.random(n, dtype=np.float64) * 5.0 + 1.0   # 1 … 6
        pmf    = rng.random(n, dtype=np.float64) * 8.0 + 2.0  # 2 … 10

        # ---- Taichi buffers ------------------------------------------------
        x_ti   = ti.ndarray(dtype=ti.f64, shape=n); x_ti.from_numpy(x_arr)
        pmf_ti = ti.ndarray(dtype=ti.f64, shape=n); pmf_ti.from_numpy(pmf)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)

        # ---- timing --------------------------------------------------------
        stats_py = get_function_benchmark(
            lambda: py_get_pdf_distribution_in_pmf(x_arr, pmf), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_get_pdf_distribution_in_pmf(x_arr, pmf), ops_per_call=n
        )
        stats_ke = get_function_benchmark(
            lambda: kget_pdf_distribution_in_pmf(x_ti, pmf_ti, 1, res_ti), ops_per_call=n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_ke["array_stats"],
        ])

    # ---- header ------------------------------------------------------------
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_ke["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ---- output ------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_combined_csv(os.path.join(out_dir, "pdf_from_pmf_benchmark.csv"),
                      header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "pdf_from_pmf throughput benchmark",
        os.path.join(out_dir, "pdf_from_pmf_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_distribution_in_dn_csv()
    benchmark_pdf_from_pmf_csv()
