"""
Benchmark Python and Taichi thermal conductivity implementations.

This module benchmarks the throughput of three implementations of the
thermal conductivity calculation: a pure-Python function, a Taichi
vectorized wrapper, and a raw Taichi kernel. For each array length in
ARRAY_LENGTHS, it generates random temperature arrays (200–400 K),
times each function, and collects performance statistics. The results
are saved as a CSV file, system information as JSON, and a throughput
plot as PNG in the benchmark_outputs directory.

Examples:
    ```sh
    python -m particula.backend.taichi.gas.properties.benchmark.ti_thermal_conductivity_benchmark
    ```
    This will run the benchmark and write outputs to the appropriate
    directory.

References:
    - Yuanming Hu et al., "Taichi: A Language for High-Performance
      Numerical Computation," ACM Transactions on Graphics (TOG), 2019.
      [https://taichi.graphics/](https://taichi.graphics/)
"""
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
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity as python_get_thermal_conductivity,
)
from particula.backend.taichi.gas.properties.ti_thermal_conductivity_module import (
    ti_get_thermal_conductivity as taichi_get_thermal_conductivity,
    kget_thermal_conductivity   as taichi_kernel_get_thermal_conductivity,
)

# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_thermal_conductivity_csv() -> None:
    """
    Benchmark and save throughput for three thermal conductivity methods.

    For each array length in ARRAY_LENGTHS, this function generates a
    random temperature array, times the pure-Python, Taichi wrapper,
    and Taichi kernel implementations, and collects performance
    statistics. It writes a CSV file with all results, a JSON file
    with system information, and a PNG plot of throughput versus array
    length to the benchmark_outputs directory.

    Returns:
        - None : This function writes files and does not return a value.

    Examples:
        ```py
        from particula.backend.taichi.gas.properties.benchmark.ti_thermal_conductivity_benchmark import (
            benchmark_thermal_conductivity_csv
        )
        benchmark_thermal_conductivity_csv()
        ```
    """
    rows: list[list[float]] = []
    random_generator = np.random.default_rng(seed=RNG_SEED)

    for array_length in ARRAY_LENGTHS:
        # ----- random input data (uniform 200–400 K) --------------------
        temperature_array = (
            random_generator.random(array_length, dtype=np.float64) * 200.0 + 200.0
        )

        # ----- Taichi buffers ------------------------------------------
        temperature_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        result_ti      = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_ti.from_numpy(temperature_array)

        # ----- timing --------------------------------------------------
        stats_python = get_function_benchmark(
            lambda: python_get_thermal_conductivity(temperature_array),
            ops_per_call=array_length,
        )
        stats_taichi = get_function_benchmark(
            lambda: taichi_get_thermal_conductivity(temperature_array),
            ops_per_call=array_length,
        )
        stats_taichi_kernel = get_function_benchmark(
            lambda: taichi_kernel_get_thermal_conductivity(
                temperature_ti, result_ti
            ),
            ops_per_call=array_length,
        )

        # ----- collect one CSV row ------------------------------------
        rows.append([
            array_length,
            *stats_python["array_stats"],
            *stats_taichi["array_stats"],
            *stats_taichi_kernel["array_stats"],
        ])

    # ------------------------ header construction ----------------------
    python_header        = ["python_" + h for h in stats_python["array_headers"]]
    taichi_header        = ["taichi_" + h for h in stats_taichi["array_headers"]]
    taichi_kernel_header = [
        "taichi_kernel_" + h for h in stats_taichi_kernel["array_headers"]
    ]
    header = ["array_length", *python_header, *taichi_header, *taichi_kernel_header]

    # ------------------------ output directory -------------------------
    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)

    # ------------------------ CSV --------------------------------------
    csv_file_path = os.path.join(
        output_directory, "thermal_conductivity_benchmark.csv"
    )
    save_combined_csv(csv_file_path, header, rows)

    # ------------------------ system info JSON -------------------------
    with open(
        os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8"
    ) as file_handle:
        json.dump(get_system_info(), file_handle, indent=2)

    # ------------------------ throughput plot --------------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        "Thermal-conductivity throughput benchmark",
        os.path.join(
            output_directory, "thermal_conductivity_benchmark.png"
        ),
    )

if __name__ == "__main__":
    benchmark_thermal_conductivity_csv()
