"""Benchmarks lognormal size-distribution routines
(pure-Python, Taichi wrapper, Taichi kernel)."""
# ------------------------------------------------------------------------- #
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

# -------- reference Python ------------------------------------------------
from particula.particles.properties.lognormal_size_distribution import (
    get_lognormal_pdf_distribution as py_pdf,
    get_lognormal_pmf_distribution as py_pmf,
)

# -------- Taichi wrapper & kernel ----------------------------------------
from particula.backend.taichi.particles.properties.ti_lognormal_size_distribution_module import (
    ti_get_lognormal_pdf_distribution  as ti_pdf,
    ti_get_lognormal_pmf_distribution  as ti_pmf,
    kget_lognormal_pdf_distribution    as k_pdf,
    _compute_mode_weights,
)

# -------- reproducibility & array lengths --------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)


def _fixed_mode_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mode, GSD and #particles arrays reused in every benchmark call."""
    mode   = np.array([5e-8, 1e-7], dtype=np.float64)
    gsd    = np.array([1.5,   2.0], dtype=np.float64)
    n_part = np.array([1e9,  5e9],  dtype=np.float64)
    return mode, gsd, n_part


# ======================================================================= #
#  BENCHMARK 1 – PDF
# ======================================================================= #
def benchmark_lognormal_pdf_distribution_csv() -> None:
    """
    Time pure-Python, Taichi wrapper, and raw kernel for the PDF routine
    over ARRAY_LENGTHS, then save CSV/JSON/PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng                     = np.random.default_rng(seed=RNG_SEED)
    mode, gsd, n_part       = _fixed_mode_arrays()
    n_m                     = mode.size

    for n in ARRAY_LENGTHS:
        # ----- random x-axis ---------------------------------------------
        x_vals = rng.uniform(1e-9, 1e-6, n, dtype=np.float64)

        # ----- Taichi buffers -------------------------------------------
        x_ti   = ti.ndarray(dtype=ti.f64, shape=n)
        mode_ti= ti.ndarray(dtype=ti.f64, shape=n_m)
        gsd_ti = ti.ndarray(dtype=ti.f64, shape=n_m)
        w_ti   = ti.ndarray(dtype=ti.f64, shape=n_m)
        out_ti = ti.ndarray(dtype=ti.f64, shape=n)
        x_ti.from_numpy(x_vals)
        mode_ti.from_numpy(mode)
        gsd_ti.from_numpy(gsd)
        w_ti.from_numpy(_compute_mode_weights(x_vals, mode, gsd, n_part))

        # ----- timing ----------------------------------------------------
        stats_py     = get_function_benchmark(lambda: py_pdf(x_vals, mode, gsd, n_part), ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda: ti_pdf(x_vals, mode, gsd, n_part), ops_per_call=n)
        stats_kernel = get_function_benchmark(lambda: k_pdf(x_ti, mode_ti, gsd_ti, w_ti, out_ti), ops_per_call=n)

        rows.append([n,
                     *stats_py["array_stats"],
                     *stats_ti["array_stats"],
                     *stats_kernel["array_stats"]])

    # header --------------------------------------------------------------
    python_hdr = ["python_"+h         for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"+h         for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_"+h  for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # output --------------------------------------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_combined_csv(os.path.join(out_dir, "lognormal_pdf_distribution_benchmark.csv"),
                      header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(header, rows,
                                    "Lognormal PDF distribution benchmark",
                                    os.path.join(out_dir, "lognormal_pdf_distribution_benchmark.png"))


# ======================================================================= #
#  BENCHMARK 2 – PMF  (no raw kernel available)
# ======================================================================= #
def benchmark_lognormal_pmf_distribution_csv() -> None:
    """Benchmark the PMF wrapper vs. pure-Python implementation."""
    rows: list[list[float]] = []
    rng                     = np.random.default_rng(seed=RNG_SEED)
    mode, gsd, n_part       = _fixed_mode_arrays()

    for n in ARRAY_LENGTHS:
        x_vals  = rng.uniform(1e-9, 1e-6, n, dtype=np.float64)

        stats_py = get_function_benchmark(lambda: py_pmf(x_vals, mode, gsd, n_part), ops_per_call=n)
        stats_ti = get_function_benchmark(lambda: ti_pmf(x_vals, mode, gsd, n_part), ops_per_call=n)

        rows.append([n, *stats_py["array_stats"], *stats_ti["array_stats"]])

    python_hdr = ["python_"+h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"+h for h in stats_ti["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_combined_csv(os.path.join(out_dir, "lognormal_pmf_distribution_benchmark.csv"),
                      header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(header, rows,
                                    "Lognormal PMF distribution benchmark",
                                    os.path.join(out_dir, "lognormal_pmf_distribution_benchmark.png"))


# ======================================================================= #
if __name__ == "__main__":
    benchmark_lognormal_pdf_distribution_csv()
    benchmark_lognormal_pmf_distribution_csv()
