"""Benchmarks get_partial_pressure and get_saturation_ratio_from_pressure."""
import os, json, numpy as np, taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)
#  Python reference versions
from particula.gas.properties.pressure_function import (
    get_partial_pressure                as py_partial,
    get_saturation_ratio_from_pressure  as py_sat_ratio,
)
#  Taichi wrapper + raw kernel
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    ti_get_partial_pressure                       as ti_partial,
    kget_partial_pressure                         as k_partial,
    ti_get_saturation_ratio_from_pressure         as ti_sat_ratio,
    kget_saturation_ratio_from_pressure           as k_sat_ratio,
)

RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_partial_pressure_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel for get_partial_pressure
    over ARRAY_LENGTHS, then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # Random positive input data
        concentration = rng.random(n, dtype=np.float64) + 1e-9
        molar_mass = rng.random(n, dtype=np.float64) + 0.01  # avoid zero
        temperature = rng.random(n, dtype=np.float64) * 300 + 200  # 200-500K

        # Taichi buffers
        conc_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mm_ti = ti.ndarray(dtype=ti.f64, shape=n)
        temp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        conc_ti.from_numpy(concentration)
        mm_ti.from_numpy(molar_mass)
        temp_ti.from_numpy(temperature)

        # Timing
        stats_py = get_function_benchmark(
            lambda: py_partial(concentration, molar_mass, temperature), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_partial(concentration, molar_mass, temperature), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: k_partial(conc_ti, mm_ti, temp_ti, res_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "get_partial_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "get_partial_pressure throughput benchmark",
        os.path.join(out_dir, "get_partial_pressure_benchmark.png"),
    )


def benchmark_saturation_ratio_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel for get_saturation_ratio_from_pressure
    over ARRAY_LENGTHS, then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # Random positive input data
        partial_pressure = rng.random(n, dtype=np.float64) * 1000 + 1e-6
        pure_vapor_pressure = rng.random(n, dtype=np.float64) * 1000 + 1e-6

        # Taichi buffers
        pp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pvp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pp_ti.from_numpy(partial_pressure)
        pvp_ti.from_numpy(pure_vapor_pressure)

        # Timing
        stats_py = get_function_benchmark(
            lambda: py_sat_ratio(partial_pressure, pure_vapor_pressure), ops_per_call=n
        )
        stats_ti = get_function_benchmark(
            lambda: ti_sat_ratio(partial_pressure, pure_vapor_pressure), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: k_sat_ratio(pp_ti, pvp_ti, res_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "get_saturation_ratio_from_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "get_saturation_ratio_from_pressure throughput benchmark",
        os.path.join(out_dir, "get_saturation_ratio_from_pressure_benchmark.png"),
    )


if __name__ == "__main__":
    benchmark_partial_pressure_csv()
    benchmark_saturation_ratio_csv()
