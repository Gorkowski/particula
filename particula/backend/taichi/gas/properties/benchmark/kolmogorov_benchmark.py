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
from particula.gas.properties.kolmogorov_module import (
    get_kolmogorov_time as py_time,
    get_kolmogorov_length as py_length,
    get_kolmogorov_velocity as py_velocity,
)
from particula.backend.taichi.gas.properties.ti_kolmogorov_module import (
    ti_get_kolmogorov_time as ti_time,
    ti_get_kolmogorov_length as ti_length,
    ti_get_kolmogorov_velocity as ti_velocity,
    kget_kolmogorov_time as k_time,
    kget_kolmogorov_length as k_length,
    kget_kolmogorov_velocity as k_velocity,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_kolmogorov_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows_time: list[list[float]] = []
    rows_length: list[list[float]] = []
    rows_velocity: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    # ------------------------------------------------------------------ #
    #  LOOP OVER ARRAY LENGTHS – *no separate helper function*           #
    # ------------------------------------------------------------------ #
    for n in ARRAY_LENGTHS:
        # ----- random input data ---------------------------------------
        v = rng.random(n, dtype=np.float64) + 1e-9
        eps = rng.random(n, dtype=np.float64) + 1e-9

        # ----- Taichi buffers (create once per length) -----------------
        v_ti = ti.ndarray(dtype=ti.f64, shape=n)
        eps_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        v_ti.from_numpy(v)
        eps_ti.from_numpy(eps)

        # ----- timing: kolmogorov_time ---------------------------------
        stats_py_time     = get_function_benchmark(
            lambda: py_time(v, eps), ops_per_call=n
        )
        stats_ti_time     = get_function_benchmark(
            lambda: ti_time(v, eps), ops_per_call=n
        )
        stats_kernel_time = get_function_benchmark(
            lambda: k_time(v_ti, eps_ti, res_ti), ops_per_call=n
        )
        rows_time.append([
            n,
            *stats_py_time["array_stats"],
            *stats_ti_time["array_stats"],
            *stats_kernel_time["array_stats"],
        ])

        # ----- timing: kolmogorov_length -------------------------------
        stats_py_length     = get_function_benchmark(
            lambda: py_length(v, eps), ops_per_call=n
        )
        stats_ti_length     = get_function_benchmark(
            lambda: ti_length(v, eps), ops_per_call=n
        )
        stats_kernel_length = get_function_benchmark(
            lambda: k_length(v_ti, eps_ti, res_ti), ops_per_call=n
        )
        rows_length.append([
            n,
            *stats_py_length["array_stats"],
            *stats_ti_length["array_stats"],
            *stats_kernel_length["array_stats"],
        ])

        # ----- timing: kolmogorov_velocity -----------------------------
        stats_py_velocity     = get_function_benchmark(
            lambda: py_velocity(v, eps), ops_per_call=n
        )
        stats_ti_velocity     = get_function_benchmark(
            lambda: ti_velocity(v, eps), ops_per_call=n
        )
        stats_kernel_velocity = get_function_benchmark(
            lambda: k_velocity(v_ti, eps_ti, res_ti), ops_per_call=n
        )
        rows_velocity.append([
            n,
            *stats_py_velocity["array_stats"],
            *stats_ti_velocity["array_stats"],
            *stats_kernel_velocity["array_stats"],
        ])

    # ------------------------ header construction ----------------------
    python_hdr  = ["python_"        + h for h in stats_py_time["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti_time["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel_time["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ------------------------ output directory -------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------ CSV --------------------------------------
    save_combined_csv(
        os.path.join(out_dir, "kolmogorov_time_benchmark.csv"),
        header, rows_time
    )
    save_combined_csv(
        os.path.join(out_dir, "kolmogorov_length_benchmark.csv"),
        header, rows_length
    )
    save_combined_csv(
        os.path.join(out_dir, "kolmogorov_velocity_benchmark.csv"),
        header, rows_velocity
    )

    # ------------------------ system info JSON -------------------------
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ------------------------ throughput plot --------------------------
    plot_throughput_vs_array_length(
        header,
        rows_time,
        "Kolmogorov time throughput benchmark",
        os.path.join(out_dir, "kolmogorov_time_benchmark.png"),
    )
    plot_throughput_vs_array_length(
        header,
        rows_length,
        "Kolmogorov length throughput benchmark",
        os.path.join(out_dir, "kolmogorov_length_benchmark.png"),
    )
    plot_throughput_vs_array_length(
        header,
        rows_velocity,
        "Kolmogorov velocity throughput benchmark",
        os.path.join(out_dir, "kolmogorov_velocity_benchmark.png"),
    )

if __name__ == "__main__":

    benchmark_kolmogorov_csv()
