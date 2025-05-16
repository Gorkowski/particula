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
from particula.gas.properties.taylor_microscale import (
    get_lagrangian_taylor_microscale_time  as py_tau,
    get_taylor_microscale                  as py_lam,
    get_taylor_microscale_reynolds_number  as py_reλ,
)
from particula.backend.taichi.gas.properties.ti_taylor_microscale_module import (
    ti_get_lagrangian_taylor_microscale_time  as ti_tau,
    ti_get_taylor_microscale                  as ti_lam,
    ti_get_taylor_microscale_reynolds_number  as ti_reλ,
    kget_lagrangian_taylor_microscale_time    as k_tau,
    kget_taylor_microscale                    as k_lam,
    kget_taylor_microscale_reynolds_number    as k_reλ,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_lagrangian_taylor_microscale_time_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ----- random input data ---------------------------------------
        kt = rng.random(n, dtype=np.float64) + 1e-9
        rl = rng.random(n, dtype=np.float64) * 1e3 + 100.
        av = rng.random(n, dtype=np.float64) + 1e-9

        # ----- Taichi buffers (create once per length) -----------------
        kt_ti = ti.ndarray(dtype=ti.f64, shape=n)
        rl_ti = ti.ndarray(dtype=ti.f64, shape=n)
        av_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        kt_ti.from_numpy(kt)
        rl_ti.from_numpy(rl)
        av_ti.from_numpy(av)

        # ----- timing --------------------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_tau(kt, rl, av), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_tau(kt, rl, av), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: k_tau(kt_ti, rl_ti, av_ti, res_ti), ops_per_call=n
        )

        # ----- collect one CSV row ------------------------------------
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

    csv_path = os.path.join(out_dir, "lagrangian_taylor_microscale_time_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # system info JSON (shared)
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Lagrangian Taylor-microscale time throughput benchmark",
        os.path.join(out_dir, "lagrangian_taylor_microscale_time_benchmark.png"),
    )

def benchmark_taylor_microscale_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        u   = rng.random(n, dtype=np.float64) + 1e-9
        nu  = rng.random(n, dtype=np.float64) * 1e-4 + 1e-9
        eps = rng.random(n, dtype=np.float64) + 1e-9

        u_ti   = ti.ndarray(dtype=ti.f64, shape=n)
        nu_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        eps_ti = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        u_ti.from_numpy(u)
        nu_ti.from_numpy(nu)
        eps_ti.from_numpy(eps)

        stats_py     = get_function_benchmark(
            lambda: py_lam(u, nu, eps), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_lam(u, nu, eps), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: k_lam(u_ti, nu_ti, eps_ti, res_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "taylor_microscale_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # system info JSON (shared)
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Taylor-microscale throughput benchmark",
        os.path.join(out_dir, "taylor_microscale_benchmark.png"),
    )

def benchmark_taylor_microscale_reynolds_number_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        u   = rng.random(n, dtype=np.float64) + 1e-9
        nu  = rng.random(n, dtype=np.float64) * 1e-4 + 1e-9
        eps = rng.random(n, dtype=np.float64) + 1e-9
        lam = py_lam(u, nu, eps)

        u_ti   = ti.ndarray(dtype=ti.f64, shape=n)
        lam_ti = ti.ndarray(dtype=ti.f64, shape=n)
        nu_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        u_ti.from_numpy(u)
        lam_ti.from_numpy(lam)
        nu_ti.from_numpy(nu)

        stats_py     = get_function_benchmark(
            lambda: py_reλ(u, lam, nu), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_reλ(u, lam, nu), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: k_reλ(u_ti, lam_ti, nu_ti, res_ti), ops_per_call=n
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

    csv_path = os.path.join(out_dir, "taylor_microscale_reynolds_number_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # system info JSON (shared)
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Taylor-microscale Reynolds number throughput benchmark",
        os.path.join(out_dir, "taylor_microscale_reynolds_number_benchmark.png"),
    )

if __name__ == "__main__":

    benchmark_lagrangian_taylor_microscale_time_csv()
    benchmark_taylor_microscale_csv()
    benchmark_taylor_microscale_reynolds_number_csv()
