"""Benchmarks Kolmogorov time/length/velocity (Python vs. Taichi)."""

# --- std / 3rd-party -------------------------------------------------------
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)

# --- functions under test --------------------------------------------------
from particula.gas.properties.kolmogorov_module import (
    get_kolmogorov_time, get_kolmogorov_length, get_kolmogorov_velocity,
)
from particula.backend.taichi.gas.properties.ti_kolmogorov_module import (
    ti_get_kolmogorov_time,  ti_get_kolmogorov_length,  ti_get_kolmogorov_velocity,
    kget_kolmogorov_time,    kget_kolmogorov_length,    kget_kolmogorov_velocity,
)

# --- benchmark config ------------------------------------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)      # 10² … 10⁸
ti.init(arch=ti.cpu)                                  # fixed backend

# -------------------------------------------------------------------------- #
#  Helper: single-routine benchmark (NO separate helper for length loop)     #
# -------------------------------------------------------------------------- #
def _one_benchmark(name, py_func, ti_func, ti_kernel, file_stub):
    rows, rng = [], np.random.default_rng(seed=RNG_SEED)
    for n in ARRAY_LENGTHS:
        ν  = rng.random(n, dtype=np.float64) + 1e-12   # kinematic viscosity
        ε  = rng.random(n, dtype=np.float64) + 1e-12   # dissipation rate

        ν_ti, ε_ti, res_ti = (ti.ndarray(dtype=ti.f64, shape=n) for _ in range(3))
        ν_ti.from_numpy(ν); ε_ti.from_numpy(ε)

        stats_py     = get_function_benchmark(lambda: py_func(ν, ε),                ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda: ti_func(ν, ε),                ops_per_call=n)
        stats_kernel = get_function_benchmark(lambda: ti_kernel(ν_ti, ε_ti, res_ti), ops_per_call=n)

        rows.append([n, *stats_py["array_stats"], *stats_ti["array_stats"],
                        *stats_kernel["array_stats"]])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{file_stub}.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows, f"{name} throughput benchmark",
        os.path.join(out_dir, f"{file_stub}.png"),
    )

# -------------------------------------------------------------------------- #
#  Three public benchmark entry points                                       #
# -------------------------------------------------------------------------- #
def benchmark_kolmogorov_time_csv():
    """Benchmark Kolmogorov *time*."""
    _one_benchmark(
        "Kolmogorov time",
        get_kolmogorov_time,  ti_get_kolmogorov_time,  kget_kolmogorov_time,
        "kolmogorov_time_benchmark",
    )

def benchmark_kolmogorov_length_csv():
    """Benchmark Kolmogorov *length*."""
    _one_benchmark(
        "Kolmogorov length",
        get_kolmogorov_length, ti_get_kolmogorov_length, kget_kolmogorov_length,
        "kolmogorov_length_benchmark",
    )

def benchmark_kolmogorov_velocity_csv():
    """Benchmark Kolmogorov *velocity*."""
    _one_benchmark(
        "Kolmogorov velocity",
        get_kolmogorov_velocity, ti_get_kolmogorov_velocity, kget_kolmogorov_velocity,
        "kolmogorov_velocity_benchmark",
    )

# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_kolmogorov_time_csv()
    benchmark_kolmogorov_length_csv()
    benchmark_kolmogorov_velocity_csv()
