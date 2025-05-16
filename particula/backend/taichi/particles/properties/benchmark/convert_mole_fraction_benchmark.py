"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel for
mass fraction conversion from mole fractions, in both 1D and 2D cases.
"""
import os
import json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)
from particula.particles.properties.convert_mole_fraction import (
    get_mass_fractions_from_moles as py_func,
)
from particula.backend.taichi.particles.properties.ti_convert_mole_fraction_module import (
    ti_get_mass_fractions_from_moles as ti_func,
    kget_mass_fractions_1d, kget_mass_fractions_2d,
)

RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 7, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_mass_fractions_from_moles_1d_csv():
    """Benchmark 1D mass fraction conversion and save CSV, JSON, and PNG outputs."""
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)
    for n in ARRAY_LENGTHS:
        mole = rng.random(n)
        mole /= mole.sum()
        mw   = rng.uniform(10.0, 200.0, n)
        mole = mole.astype(np.float64)
        mw = mw.astype(np.float64)

        mole_ti = ti.ndarray(dtype=ti.f64, shape=mole.shape)
        mw_ti   = ti.ndarray(dtype=ti.f64, shape=mw.shape)
        out_ti  = ti.ndarray(dtype=ti.f64, shape=mole.shape)
        mole_ti.from_numpy(mole)
        mw_ti.from_numpy(mw)

        stats_py     = get_function_benchmark(
            lambda: py_func(mole, mw), ops_per_call=mole.size
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(mole, mw), ops_per_call=mole.size
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_mass_fractions_1d(mole_ti, mw_ti, out_ti), ops_per_call=mole.size
        )

        rows.append([
            mole.size,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "mass_fractions_from_moles_1d_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "Mass Fractions From Moles 1D throughput benchmark",
        os.path.join(out_dir, "mass_fractions_from_moles_1d_benchmark.png"),
    )

def benchmark_mass_fractions_from_moles_2d_csv():
    """Benchmark 2D mass fraction conversion and save CSV, JSON, and PNG outputs."""
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)
    N_COLS = 8
    for n in ARRAY_LENGTHS:
        mole = rng.random((n, N_COLS))
        mole /= mole.sum(axis=1, keepdims=True)
        mw   = rng.uniform(10.0, 200.0, N_COLS)
        mole = mole.astype(np.float64)
        mw = mw.astype(np.float64)

        mole_ti = ti.ndarray(dtype=ti.f64, shape=mole.shape)
        mw_ti   = ti.ndarray(dtype=ti.f64, shape=mw.shape)
        out_ti  = ti.ndarray(dtype=ti.f64, shape=mole.shape)
        mole_ti.from_numpy(mole)
        mw_ti.from_numpy(mw)

        stats_py     = get_function_benchmark(
            lambda: py_func(mole, mw), ops_per_call=mole.size
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(mole, mw), ops_per_call=mole.size
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_mass_fractions_2d(mole_ti, mw_ti, out_ti), ops_per_call=mole.size
        )

        rows.append([
            mole.size,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "mass_fractions_from_moles_2d_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "Mass Fractions From Moles 2D throughput benchmark",
        os.path.join(out_dir, "mass_fractions_from_moles_2d_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_mass_fractions_from_moles_1d_csv()
    benchmark_mass_fractions_from_moles_2d_csv()
