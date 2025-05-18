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
from particula.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity as python_get_dynamic_viscosity,
)
from particula.backend.taichi.gas.properties.ti_dynamic_viscosity_module import (
    ti_get_dynamic_viscosity   as taichi_get_dynamic_viscosity,
    kget_dynamic_viscosity     as taichi_kernel_get_dynamic_viscosity,
)
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
)

# -- fixed random seed and Taichi backend for reproducibility -----------
RANDOM_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)  # 10²–10⁸
ti.init(arch=ti.cpu)

def benchmark_dynamic_viscosity_csv():
    """
    Run throughput benchmarks for dynamic-viscosity calculations.

    For each value in ARRAY_LENGTHS this function:
        1. Generates random temperature input plus reference arrays.
        2. Benchmarks three implementations
           – pure-Python,
           – Taichi wrapper,
           – Taichi raw kernel.
        3. Writes the results to
           ‘dynamic_viscosity_benchmark.csv’, ‘system_info.json’,
           and ‘dynamic_viscosity_benchmark.png’ in ./benchmark_outputs/.

    Returns:
        None

    Examples:
        >>> benchmark_dynamic_viscosity_csv()
    """
    rows = []
    random_generator = np.random.default_rng(seed=RANDOM_SEED)

    for array_length in ARRAY_LENGTHS:
        # ----- random input data -----------------------------------------
        temperatures = (
            random_generator.random(array_length, dtype=np.float64) * 150.0 + 200.0
        )
        reference_viscosity_array = np.full(
            array_length, REF_VISCOSITY_AIR_STP, dtype=np.float64
        )
        reference_temperature_array = np.full(
            array_length, REF_TEMPERATURE_STP, dtype=np.float64
        )

        # ----- Taichi buffers --------------------------------------------
        temperatures_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        reference_viscosity_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        reference_temperature_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        result_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperatures_ti.from_numpy(temperatures)
        reference_viscosity_ti.from_numpy(reference_viscosity_array)
        reference_temperature_ti.from_numpy(reference_temperature_array)

        # ----- timing ----------------------------------------------------
        python_statistics = get_function_benchmark(
            lambda: python_get_dynamic_viscosity(
                temperatures,
                reference_viscosity=reference_viscosity_array,
                reference_temperature=reference_temperature_array,
            ),
            ops_per_call=array_length,
        )

        taichi_statistics = get_function_benchmark(
            lambda: taichi_get_dynamic_viscosity(
                temperatures,
                reference_viscosity_array,
                reference_temperature_array,
            ),
            ops_per_call=array_length,
        )
        taichi_kernel_statistics = get_function_benchmark(
            lambda: taichi_kernel_get_dynamic_viscosity(
                temperatures_ti,
                reference_viscosity_ti,
                reference_temperature_ti,
                result_ti,
            ),
            ops_per_call=array_length,
        )

        # ----- collect row -----------------------------------------------
        rows.append([
            array_length,
            *python_statistics["array_stats"],
            *taichi_statistics["array_stats"],
            *taichi_kernel_statistics["array_stats"],
        ])

    # ---------------- header --------------------------------------------
    python_header = [
        "python_" + h for h in python_statistics["array_headers"]
    ]
    taichi_header = [
        "taichi_" + h for h in taichi_statistics["array_headers"]
    ]
    taichi_kernel_header = [
        "taichi_kernel_" + h for h in taichi_kernel_statistics["array_headers"]
    ]
    header = [
        "array_length", *python_header, *taichi_header, *taichi_kernel_header
    ]

    # ---------------- output dir ----------------------------------------
    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)

    # ---------------- CSV -----------------------------------------------
    save_combined_csv(
        os.path.join(output_directory, "dynamic_viscosity_benchmark.csv"),
        header, rows
    )

    # ---------------- system info JSON ----------------------------------
    with open(
        os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8"
    ) as file_handle:
        json.dump(get_system_info(), file_handle, indent=2)

    # ---------------- throughput plot -----------------------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        "Dynamic viscosity throughput benchmark",
        os.path.join(output_directory, "dynamic_viscosity_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_dynamic_viscosity_csv()
