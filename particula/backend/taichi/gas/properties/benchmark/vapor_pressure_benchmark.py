"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel for vapor pressure routines."""
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

from particula.gas.properties.vapor_pressure_module import (
    get_antoine_vapor_pressure,
    get_clausius_clapeyron_vapor_pressure,
    get_buck_vapor_pressure,
)
from particula.backend.taichi.gas.properties.ti_vapor_pressure_module import (
    ti_get_antoine_vapor_pressure,
    ti_get_clausius_clapeyron_vapor_pressure,
    ti_get_buck_vapor_pressure,
    kget_antoine_vapor_pressure,
    kget_clausius_clapeyron_vapor_pressure,
    kget_buck_vapor_pressure,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_antoine_vapor_pressure_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel for Antoine vapor pressure
    over ARRAY_LENGTHS, then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # Antoine parameters: a, b, c, temperature
        a = rng.uniform(7.0, 9.0, n)
        b = rng.uniform(1000.0, 2000.0, n)
        c = rng.uniform(200.0, 250.0, n)
        T = rng.uniform(250.0, 400.0, n)

        # Taichi buffers
        a_ti = ti.ndarray(dtype=ti.f64, shape=n)
        b_ti = ti.ndarray(dtype=ti.f64, shape=n)
        c_ti = ti.ndarray(dtype=ti.f64, shape=n)
        T_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        a_ti.from_numpy(a)
        b_ti.from_numpy(b)
        c_ti.from_numpy(c)
        T_ti.from_numpy(T)

        stats_py = get_function_benchmark(
            lambda: get_antoine_vapor_pressure(a, b, c, T), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_get_antoine_vapor_pressure(a, b, c, T), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_antoine_vapor_pressure(a_ti, b_ti, c_ti, T_ti, res_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "antoine_vapor_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Antoine vapor pressure throughput benchmark",
        os.path.join(out_dir, "antoine_vapor_pressure_benchmark.png"),
    )

def benchmark_clausius_clapeyron_vapor_pressure_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel for Clausius-Clapeyron vapor pressure
    over ARRAY_LENGTHS, then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)
    gas_constant = 8.31446261815324

    for n in ARRAY_LENGTHS:
        latent_heat = rng.uniform(20000.0, 50000.0, n)
        T0 = rng.uniform(250.0, 400.0, n)
        P0 = rng.uniform(1e4, 1e5, n)
        T = rng.uniform(250.0, 400.0, n)

        lh_ti = ti.ndarray(dtype=ti.f64, shape=n)
        T0_ti = ti.ndarray(dtype=ti.f64, shape=n)
        P0_ti = ti.ndarray(dtype=ti.f64, shape=n)
        T_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        lh_ti.from_numpy(latent_heat)
        T0_ti.from_numpy(T0)
        P0_ti.from_numpy(P0)
        T_ti.from_numpy(T)

        stats_py = get_function_benchmark(
            lambda: get_clausius_clapeyron_vapor_pressure(latent_heat, T0, P0, T, gas_constant=gas_constant), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_get_clausius_clapeyron_vapor_pressure(latent_heat, T0, P0, T, gas_constant=gas_constant), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_clausius_clapeyron_vapor_pressure(lh_ti, T0_ti, P0_ti, T_ti, gas_constant, res_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "clausius_clapeyron_vapor_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Clausius-Clapeyron vapor pressure throughput benchmark",
        os.path.join(out_dir, "clausius_clapeyron_vapor_pressure_benchmark.png"),
    )

def benchmark_buck_vapor_pressure_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel for Buck vapor pressure
    over ARRAY_LENGTHS, then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        T = rng.uniform(200.0, 350.0, n)

        T_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        T_ti.from_numpy(T)

        stats_py = get_function_benchmark(
            lambda: get_buck_vapor_pressure(T), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_get_buck_vapor_pressure(T), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_buck_vapor_pressure(T_ti, res_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "buck_vapor_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Buck vapor pressure throughput benchmark",
        os.path.join(out_dir, "buck_vapor_pressure_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_antoine_vapor_pressure_csv()
    benchmark_clausius_clapeyron_vapor_pressure_csv()
    benchmark_buck_vapor_pressure_csv()
