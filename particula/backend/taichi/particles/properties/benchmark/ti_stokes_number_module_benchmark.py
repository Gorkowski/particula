"""Benchmarks Python vs. Taichi (wrapper + kernel) for stokes-number."""
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)

# functions to benchmark
from particula.particles.properties.stokes_number import (
    get_stokes_number as py_func,
)
from particula.backend.taichi.particles.properties.ti_stokes_number_module import (
    ti_get_stokes_number as ti_func,
    kget_stokes_number   as ti_kernel,
)

RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)     # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_stokes_number_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then emit CSV/JSON/PNG into ./benchmark_outputs/.
    """
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        pit = rng.random(n, dtype=np.float64) + 1e-12   # >0
        kt  = rng.random(n, dtype=np.float64) + 1e-12

        pit_ti = ti.ndarray(dtype=ti.f64, shape=n)
        kt_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pit_ti.from_numpy(pit)
        kt_ti.from_numpy(kt)

        stats_py     = get_function_benchmark(
            lambda: py_func(pit, kt), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(pit, kt), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(pit_ti, kt_ti, res_ti), ops_per_call=n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ti_stokes_number_module_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Stokes-number throughput benchmark",
        os.path.join(out_dir, "ti_stokes_number_module_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_stokes_number_csv()
