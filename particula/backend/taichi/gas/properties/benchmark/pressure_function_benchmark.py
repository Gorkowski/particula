"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel."""
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

# ---------- functions to benchmark -----------------------------------------
from particula.gas.properties.pressure_function import (
    get_partial_pressure              as py_partial_pressure,
    get_saturation_ratio_from_pressure as py_satur_ratio,
)
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    ti_get_partial_pressure                 as ti_partial_pressure,
    kget_partial_pressure                   as k_partial_pressure,
    ti_get_saturation_ratio_from_pressure   as ti_satur_ratio,
    kget_saturation_ratio_from_pressure     as k_satur_ratio,
)

# ---------- benchmark configuration ----------------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)         # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_partial_pressure_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # random input data
        concentration = rng.random(n, dtype=np.float64) + 1e-9
        molar_mass    = rng.random(n, dtype=np.float64) + 1e-9
        temperature   = rng.random(n, dtype=np.float64) + 1e-9

        # Taichi buffers
        arr1 = ti.ndarray(dtype=ti.f64, shape=n)
        arr2 = ti.ndarray(dtype=ti.f64, shape=n)
        arr3 = ti.ndarray(dtype=ti.f64, shape=n)
        res  = ti.ndarray(dtype=ti.f64, shape=n)
        arr1.from_numpy(concentration)
        arr2.from_numpy(molar_mass)
        arr3.from_numpy(temperature)

        # timing
        stats_py     = get_function_benchmark(
            lambda: py_partial_pressure(concentration, molar_mass, temperature), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_partial_pressure(concentration, molar_mass, temperature), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: k_partial_pressure(arr1, arr2, arr3, res), ops_per_call=n
        )

        # collect one CSV row
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # header construction
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # output directory
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "partial_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # system info JSON
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # throughput plot
    plot_throughput_vs_array_length(
        header,
        rows,
        "Partial Pressure throughput benchmark",
        os.path.join(out_dir, "partial_pressure_benchmark.png"),
    )

def benchmark_saturation_ratio_from_pressure_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # random input data
        partial_pressure     = rng.random(n, dtype=np.float64) * 1000 + 1e-9
        pure_vapor_pressure = rng.random(n, dtype=np.float64) * 1000 + 1e-9

        # Taichi buffers
        arr1 = ti.ndarray(dtype=ti.f64, shape=n)
        arr2 = ti.ndarray(dtype=ti.f64, shape=n)
        res  = ti.ndarray(dtype=ti.f64, shape=n)
        arr1.from_numpy(partial_pressure)
        arr2.from_numpy(pure_vapor_pressure)

        # timing
        stats_py     = get_function_benchmark(
            lambda: py_satur_ratio(partial_pressure, pure_vapor_pressure), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_satur_ratio(partial_pressure, pure_vapor_pressure), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: k_satur_ratio(arr1, arr2, res), ops_per_call=n
        )

        # collect one CSV row
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # header construction
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # output directory
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "saturation_ratio_from_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # system info JSON
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # throughput plot
    plot_throughput_vs_array_length(
        header,
        rows,
        "Saturation Ratio from Pressure throughput benchmark",
        os.path.join(out_dir, "saturation_ratio_from_pressure_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_partial_pressure_csv()
    benchmark_saturation_ratio_from_pressure_csv()
