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

# functions to benchmark
from particula.particles.properties.convert_mass_concentration import (
    get_mole_fraction_from_mass        as py_func_mole,
    get_volume_fraction_from_mass      as py_func_vol,
    get_mass_fraction_from_mass        as py_func_mass
)
from particula.backend.taichi.particles.properties.ti_convert_mass_concentration_module import (
    ti_get_mole_fraction_from_mass      as ti_func_mole,
    ti_get_volume_fraction_from_mass    as ti_func_vol,
    ti_get_mass_fraction_from_mass      as ti_func_mass,
    kget_mole_fraction_from_mass        as ti_kernel_mole,
    kget_volume_fraction_from_mass      as ti_kernel_vol,
    kget_mass_fraction_from_mass        as ti_kernel_mass,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_mole_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        m = rng.random(n, dtype=np.float64) + 1e-9
        mm = rng.random(n, dtype=np.float64) + 1e-9

        m_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mm_ti = ti.ndarray(dtype=ti.f64, shape=n)
        out_ti = ti.ndarray(dtype=ti.f64, shape=n)
        m_ti.from_numpy(m)
        mm_ti.from_numpy(mm)

        stats_py     = get_function_benchmark(
            lambda: py_func_mole(m, mm), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func_mole(m, mm), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel_mole(m_ti, mm_ti, out_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "convert_mass_concentration_mole_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Mole fraction throughput benchmark",
        os.path.join(out_dir, "convert_mass_concentration_mole_benchmark.png"),
    )

def benchmark_volume_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        m = rng.random(n, dtype=np.float64) + 1e-9
        rho = rng.random(n, dtype=np.float64) + 1e-9

        m_ti = ti.ndarray(dtype=ti.f64, shape=n)
        rho_ti = ti.ndarray(dtype=ti.f64, shape=n)
        out_ti = ti.ndarray(dtype=ti.f64, shape=n)
        m_ti.from_numpy(m)
        rho_ti.from_numpy(rho)

        stats_py     = get_function_benchmark(
            lambda: py_func_vol(m, rho), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func_vol(m, rho), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel_vol(m_ti, rho_ti, out_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "convert_mass_concentration_volume_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Volume fraction throughput benchmark",
        os.path.join(out_dir, "convert_mass_concentration_volume_benchmark.png"),
    )

def benchmark_mass_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        m = rng.random(n, dtype=np.float64) + 1e-9

        m_ti = ti.ndarray(dtype=ti.f64, shape=n)
        out_ti = ti.ndarray(dtype=ti.f64, shape=n)
        m_ti.from_numpy(m)

        stats_py     = get_function_benchmark(
            lambda: py_func_mass(m), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func_mass(m), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel_mass(m_ti, out_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "convert_mass_concentration_mass_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Mass fraction throughput benchmark",
        os.path.join(out_dir, "convert_mass_concentration_mass_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_mole_csv()
    benchmark_volume_csv()
    benchmark_mass_csv()
