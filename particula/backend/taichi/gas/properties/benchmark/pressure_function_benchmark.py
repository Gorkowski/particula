"""Benchmarks get_partial_pressure and get_saturation_ratio_from_pressure."""
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
#  Python reference versions
from particula.gas.properties.pressure_function import (
    get_partial_pressure as python_get_partial_pressure,
    get_saturation_ratio_from_pressure
    as python_get_saturation_ratio_from_pressure,
)
#  Taichi wrapper + raw kernel
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    ti_get_partial_pressure,
    kget_partial_pressure,
    ti_get_saturation_ratio_from_pressure,
    kget_saturation_ratio_from_pressure,
)

RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_partial_pressure_csv():
    """
    Benchmark the throughput of get_partial_pressure implementations.

    Times the pure-Python, Taichi wrapper, and raw kernel versions of
    get_partial_pressure over a range of array lengths. Results are saved
    as CSV, JSON, and PNG files in the ./benchmark_outputs/ directory.

    Returns:
        - None : Results are written to disk; nothing is returned.

    Examples:
        ```py
        benchmark_partial_pressure_csv()
        # Output: Files saved in ./benchmark_outputs/
        ```
    """
    result_rows = []
    random_generator = np.random.default_rng(seed=RNG_SEED)

    for array_length in ARRAY_LENGTHS:
        # Random positive input data
        concentration = (
            random_generator.random(array_length, dtype=np.float64) + 1.0e-9
        )
        molar_mass = random_generator.random(array_length, dtype=np.float64) + 0.01
        temperature = (
            random_generator.random(array_length, dtype=np.float64) * 300 + 200
        )

        # Taichi buffers
        concentration_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        molar_mass_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        result_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        concentration_field.from_numpy(concentration)
        molar_mass_field.from_numpy(molar_mass)
        temperature_field.from_numpy(temperature)

        # Timing
        python_stats = get_function_benchmark(
            lambda: python_get_partial_pressure(
                concentration, molar_mass, temperature
            ),
            ops_per_call=array_length,
        )
        taichi_stats = get_function_benchmark(
            lambda: ti_get_partial_pressure(
                concentration, molar_mass, temperature
            ),
            ops_per_call=array_length,
        )
        kernel_stats = get_function_benchmark(
            lambda: kget_partial_pressure(
                concentration_field,
                molar_mass_field,
                temperature_field,
                result_field,
            ),
            ops_per_call=array_length,
        )

        result_rows.append(
            [
                array_length,
                *python_stats["array_stats"],
                *taichi_stats["array_stats"],
                *kernel_stats["array_stats"],
            ]
        )

    python_header = [
        f"python_{header_name}"
        for header_name in python_stats["array_headers"]
    ]
    taichi_header = [
        f"taichi_{header_name}"
        for header_name in taichi_stats["array_headers"]
    ]
    kernel_header = [
        f"taichi_kernel_{header_name}"
        for header_name in kernel_stats["array_headers"]
    ]
    header = ["array_length", *python_header, *taichi_header, *kernel_header]

    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)

    csv_file_path = os.path.join(
        output_directory,
        "get_partial_pressure_benchmark.csv",
    )
    save_combined_csv(csv_file_path, header, result_rows)

    with open(
        os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        result_rows,
        "get_partial_pressure throughput benchmark",
        os.path.join(
            output_directory,
            "get_partial_pressure_benchmark.png",
        ),
    )




def benchmark_saturation_ratio_csv():
    """
    Benchmark the throughput of get_saturation_ratio_from_pressure.

    Times the pure-Python, Taichi wrapper, and raw kernel versions of
    get_saturation_ratio_from_pressure over a range of array lengths.
    Results are saved as CSV, JSON, and PNG files in the
    ./benchmark_outputs/ directory.

    Returns:
        - None : Results are written to disk; nothing is returned.

    Examples:
        ```py
        benchmark_saturation_ratio_csv()
        # Output: Files saved in ./benchmark_outputs/
        ```
    """
    result_rows = []
    random_generator = np.random.default_rng(seed=RNG_SEED)

    for array_length in ARRAY_LENGTHS:
        # Random positive input data
        partial_pressure = (
            random_generator.random(array_length, dtype=np.float64) * 1000 + 1.0e-6
        )
        pure_vapor_pressure = (
            random_generator.random(array_length, dtype=np.float64) * 1000 + 1.0e-6
        )

        # Taichi buffers
        partial_pressure_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        pure_vapor_pressure_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        result_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        partial_pressure_field.from_numpy(partial_pressure)
        pure_vapor_pressure_field.from_numpy(pure_vapor_pressure)

        # Timing
        python_stats = get_function_benchmark(
            lambda: python_get_saturation_ratio_from_pressure(
                partial_pressure, pure_vapor_pressure
            ),
            ops_per_call=array_length,
        )
        taichi_stats = get_function_benchmark(
            lambda: ti_get_saturation_ratio_from_pressure(
                partial_pressure, pure_vapor_pressure
            ),
            ops_per_call=array_length,
        )
        kernel_stats = get_function_benchmark(
            lambda: kget_saturation_ratio_from_pressure(
                partial_pressure_field,
                pure_vapor_pressure_field,
                result_field,
            ),
            ops_per_call=array_length,
        )

        result_rows.append(
            [
                array_length,
                *python_stats["array_stats"],
                *taichi_stats["array_stats"],
                *kernel_stats["array_stats"],
            ]
        )

    python_header = [
        f"python_{header_name}"
        for header_name in python_stats["array_headers"]
    ]
    taichi_header = [
        f"taichi_{header_name}"
        for header_name in taichi_stats["array_headers"]
    ]
    kernel_header = [
        f"taichi_kernel_{header_name}"
        for header_name in kernel_stats["array_headers"]
    ]
    header = ["array_length", *python_header, *taichi_header, *kernel_header]

    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)

    csv_file_path = os.path.join(
        output_directory,
        "get_saturation_ratio_from_pressure_benchmark.csv",
    )
    save_combined_csv(csv_file_path, header, result_rows)

    with open(
        os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        result_rows,
        "get_saturation_ratio_from_pressure throughput benchmark",
        os.path.join(
            output_directory,
            "get_saturation_ratio_from_pressure_benchmark.png",
        ),
    )


    

if __name__ == "__main__":
    benchmark_partial_pressure_csv()
    benchmark_saturation_ratio_csv()
