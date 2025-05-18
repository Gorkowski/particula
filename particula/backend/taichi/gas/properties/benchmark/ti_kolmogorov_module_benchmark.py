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
RANDOM_SEED   = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)      # 10² … 10⁸
ti.init(arch=ti.cpu)                                  # fixed backend

# -------------------------------------------------------------------------- #
#  Helper: single-routine benchmark (NO separate helper for length loop)     #
# -------------------------------------------------------------------------- #
def _one_benchmark(
    benchmark_name, python_func, taichi_func, taichi_kernel, file_stub
):
    rows = []
    random_generator = np.random.default_rng(seed=RANDOM_SEED)
    for array_length in ARRAY_LENGTHS:
        kinematic_viscosity = (
            random_generator.random(array_length, dtype=np.float64) + 1e-12
        )
        dissipation_rate = (
            random_generator.random(array_length, dtype=np.float64) + 1e-12
        )

        kinematic_viscosity_ti, dissipation_rate_ti, result_ti = (
            ti.ndarray(dtype=ti.f64, shape=array_length) for _ in range(3)
        )
        kinematic_viscosity_ti.from_numpy(kinematic_viscosity)
        dissipation_rate_ti.from_numpy(dissipation_rate)

        statistics_python = get_function_benchmark(
            lambda: python_func(kinematic_viscosity, dissipation_rate),
            ops_per_call=array_length,
        )
        statistics_taichi = get_function_benchmark(
            lambda: taichi_func(kinematic_viscosity, dissipation_rate),
            ops_per_call=array_length,
        )
        statistics_kernel = get_function_benchmark(
            lambda: taichi_kernel(
                kinematic_viscosity_ti, dissipation_rate_ti, result_ti
            ),
            ops_per_call=array_length,
        )

        rows.append(
            [
                array_length,
                *statistics_python["array_stats"],
                *statistics_taichi["array_stats"],
                *statistics_kernel["array_stats"],
            ]
        )

    python_header = [
        "python_" + h for h in statistics_python["array_headers"]
    ]
    taichi_header = [
        "taichi_" + h for h in statistics_taichi["array_headers"]
    ]
    kernel_header = [
        "taichi_kernel_" + h for h in statistics_kernel["array_headers"]
    ]
    header = ["array_length", *python_header, *taichi_header, *kernel_header]

    output_directory = os.path.join(
        os.path.dirname(__file__), "benchmark_outputs"
    )
    os.makedirs(output_directory, exist_ok=True)

    csv_path = os.path.join(output_directory, f"{file_stub}.csv")
    save_combined_csv(csv_path, header, rows)

    with open(
        os.path.join(output_directory, "system_info.json"),
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(get_system_info(), file_handle, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        f"{benchmark_name} throughput benchmark",
        os.path.join(output_directory, f"{file_stub}.png"),
    )

# -------------------------------------------------------------------------- #
#  Three public benchmark entry points                                       #
# -------------------------------------------------------------------------- #
def benchmark_kolmogorov_time_csv():
    """Benchmark Kolmogorov *time*."""
    _one_benchmark(
        "Kolmogorov time",
        get_kolmogorov_time,
        ti_get_kolmogorov_time,
        kget_kolmogorov_time,
        "kolmogorov_time_benchmark",
    )

def benchmark_kolmogorov_length_csv():
    """Benchmark Kolmogorov *length*."""
    _one_benchmark(
        "Kolmogorov length",
        get_kolmogorov_length,
        ti_get_kolmogorov_length,
        kget_kolmogorov_length,
        "kolmogorov_length_benchmark",
    )

def benchmark_kolmogorov_velocity_csv():
    """Benchmark Kolmogorov *velocity*."""
    _one_benchmark(
        "Kolmogorov velocity",
        get_kolmogorov_velocity,
        ti_get_kolmogorov_velocity,
        kget_kolmogorov_velocity,
        "kolmogorov_velocity_benchmark",
    )

# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    benchmark_kolmogorov_time_csv()
    benchmark_kolmogorov_length_csv()
    benchmark_kolmogorov_velocity_csv()
