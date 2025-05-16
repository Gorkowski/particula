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

def _run_benchmark(
    name: str,
    make_np_args,
    py_func,
    ti_func,
    ti_kernel,
) -> None:
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # --------------------------- random inputs ----------------------
        np_args, kernel_flag = make_np_args(rng, n)

        # Taichi buffers
        ti_args = [ti.ndarray(dtype=ti.f64, shape=n) for _ in np_args]
        for buf, arr in zip(ti_args, np_args):
            buf.from_numpy(arr)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)

        # --------------------------- timings ----------------------------
        stats_py = get_function_benchmark(
            lambda: py_func(*np_args), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_func(*np_args), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(*ti_args, kernel_flag, res_ti),
            ops_per_call=n,
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # --------------------------- CSV header ----------------------------
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # --------------------------- output -------------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_name = f"{name}_benchmark.csv"
    csv_path = os.path.join(out_dir, csv_name)
    save_combined_csv(csv_path, header, rows)

    # --------------------------- system info ---------------------------
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # --------------------------- throughput plot -----------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        f"{name} throughput benchmark",
        os.path.join(out_dir, f"{name}_benchmark.png"),
    )

def benchmark_distribution_in_dn_csv() -> None:
    """
    Time get_distribution_in_dn (Python, wrapper, kernel) over ARRAY_LENGTHS.
    """

    def _make_args(rng, n):
        diameter = rng.random(n, dtype=np.float64) * 9.9e-7 + 1e-8  # 1e-8 … 1e-6
        dn_dlog  = rng.random(n, dtype=np.float64) * 9.9e4 + 1e4    # 1e4 … 1e5
        # kernel flag 0  → dn/dlogdp → d_num   (same direction as wrapper’s default)
        return (diameter, dn_dlog), 0

    _run_benchmark(
        "distribution_in_dn",
        _make_args,
        py_get_distribution_in_dn,
        ti_get_distribution_in_dn,
        kget_distribution_in_dn,
    )

def benchmark_pdf_from_pmf_csv() -> None:
    """
    Time get_pdf_distribution_in_pmf (Python, wrapper, kernel) over ARRAY_LENGTHS.
    """

    def _make_args(rng, n):
        x_arr  = rng.random(n, dtype=np.float64) * 5.0 + 1.0   # 1 … 6
        pmf    = rng.random(n, dtype=np.float64) * 8.0 + 2.0   # 2 … 10
        # kernel flag 1 → PMF → PDF   (to_pdf=True)
        return (x_arr, pmf), 1

    _run_benchmark(
        "pdf_from_pmf",
        _make_args,
        py_get_pdf_distribution_in_pmf,
        ti_get_pdf_distribution_in_pmf,
        kget_pdf_distribution_in_pmf,
    )

if __name__ == "__main__":
    benchmark_distribution_in_dn_csv()
    benchmark_pdf_from_pmf_csv()
