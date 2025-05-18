"""
Benchmarking suite for vapor pressure routines (Antoine, Clausius-Clapeyron,
Buck) in both Python and Taichi implementations.

Benchmarks:
    - benchmark_antoine_vapor_pressure_csv
    - benchmark_clausius_clapeyron_vapor_pressure_csv
    - benchmark_buck_vapor_pressure_csv

Examples
--------
>>> benchmark_antoine_vapor_pressure_csv()
>>> benchmark_clausius_clapeyron_vapor_pressure_csv()
>>> benchmark_buck_vapor_pressure_csv()

References
----------
"""

# --- standard & required imports -----------------------------------------
import os, json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)

# --- functions to benchmark ----------------------------------------------
from particula.gas.properties.vapor_pressure_module import (
    get_antoine_vapor_pressure as python_get_antoine_vapor_pressure,
    get_clausius_clapeyron_vapor_pressure
        as python_get_clausius_clapeyron_vapor_pressure,
    get_buck_vapor_pressure as python_get_buck_vapor_pressure,
)
from particula.backend.taichi.gas.properties.ti_vapor_pressure_module import (
    ti_get_antoine_vapor_pressure
        as taichi_get_antoine_vapor_pressure,
    ti_get_clausius_clapeyron_vapor_pressure
        as taichi_get_clausius_clapeyron_vapor_pressure,
    ti_get_buck_vapor_pressure
        as taichi_get_buck_vapor_pressure,
    kget_antoine_vapor_pressure,
    kget_clausius_clapeyron_vapor_pressure,
    kget_buck_vapor_pressure,
)

# --- reproducibility ------------------------------------------------------
RANDOM_SEED   = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)     # 10² … 10⁸
ti.init(arch=ti.cpu)
GAS_CONSTANT = 8.31446261815324  # Universal gas constant [J mol⁻¹ K⁻¹]

# -------------------------------------------------------------------------
# Benchmark #1 – Antoine vapour pressure
# -------------------------------------------------------------------------
def benchmark_antoine_vapor_pressure_csv():
    """
    Benchmark the Antoine pure-Python, Taichi-Python, and Taichi-kernel
    implementations over array sizes 10²–10⁸.

    The routine writes a CSV file, stores system information, and creates an
    annotated throughput plot in ./benchmark_outputs.

    Returns
    -------
    None

    Examples
    --------
    >>> benchmark_antoine_vapor_pressure_csv()

    """
    result_rows: list[list] = []
    random_generator = np.random.default_rng(seed=RANDOM_SEED)
    constant_a_value, constant_b_value, constant_c_value = 8.07131, 1730.63, 233.426   # fixed water params
    for array_length in ARRAY_LENGTHS:
        # ------------ input ------------------------------------------------
        temperature = (
            random_generator.random(array_length, dtype=np.float64) * 200.0 + 250.0
        )  # 250–450 K
        constant_a_array = np.full(array_length, constant_a_value)
        constant_b_array = np.full(array_length, constant_b_value)
        constant_c_array = np.full(array_length, constant_c_value)
        # ------------ Ti buffers -------------------------------------------
        constant_a_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        constant_a_ti.from_numpy(constant_a_array)
        constant_b_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        constant_b_ti.from_numpy(constant_b_array)
        constant_c_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        constant_c_ti.from_numpy(constant_c_array)
        temperature_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_ti.from_numpy(temperature)
        result_array = ti.ndarray(dtype=ti.f64, shape=array_length)
        # ------------ timing -----------------------------------------------
        python_stats = get_function_benchmark(
            lambda: python_get_antoine_vapor_pressure(
                constant_a_value,
                constant_b_value,
                constant_c_value,
                temperature,
            ),
            ops_per_call=array_length,
        )
        taichi_stats = get_function_benchmark(
            lambda: taichi_get_antoine_vapor_pressure(
                constant_a_value,
                constant_b_value,
                constant_c_value,
                temperature,
            ),
            ops_per_call=array_length,
        )
        kernel_stats = get_function_benchmark(
            lambda: kget_antoine_vapor_pressure(
                constant_a_ti,
                constant_b_ti,
                constant_c_ti,
                temperature_ti,
                result_array,
            ),
            ops_per_call=array_length,
        )
        result_rows.append([
            array_length,
            *python_stats["array_stats"],
            *taichi_stats["array_stats"],
            *kernel_stats["array_stats"],
        ])
    # ------------------------------ IO + plotting --------------------------
    python_hdr = ["python_"        + h for h in python_stats["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in taichi_stats["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in kernel_stats["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)
    save_combined_csv(
        os.path.join(output_directory, "antoine_vapor_pressure_benchmark.csv"),
        header, result_rows
    )
    with open(
        os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header, result_rows, "Antoine vapor-pressure throughput benchmark",
        os.path.join(output_directory, "antoine_vapor_pressure_benchmark.png"),
    )

# -------------------------------------------------------------------------
# Benchmark #2 – Clausius-Clapeyron vapour pressure
# -------------------------------------------------------------------------
def benchmark_clausius_clapeyron_vapor_pressure_csv():
    """
    Benchmark the Clausius-Clapeyron pure-Python, Taichi-Python, and
    Taichi-kernel implementations over array sizes 10²–10⁸.

    The routine writes a CSV file, stores system information, and creates an
    annotated throughput plot in ./benchmark_outputs.

    Returns
    -------
    None

    Examples
    --------
    >>> benchmark_clausius_clapeyron_vapor_pressure_csv()

    """
    result_rows: list[list] = []
    random_generator = np.random.default_rng(seed=RANDOM_SEED)
    latent_heat_value, temperature_initial_value, pressure_initial_value = (
        4.066e4, 373.15, 1.01325e5
    )
    for array_length in ARRAY_LENGTHS:
        temperature = (
            random_generator.random(array_length, dtype=np.float64) * 200.0 + 250.0
        )
        latent_heat_array = np.full(array_length, latent_heat_value)
        temperature_initial_array = np.full(array_length, temperature_initial_value)
        pressure_initial_array = np.full(array_length, pressure_initial_value)
        latent_heat_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        latent_heat_ti.from_numpy(latent_heat_array)
        temperature_initial_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_initial_ti.from_numpy(temperature_initial_array)
        pressure_initial_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        pressure_initial_ti.from_numpy(pressure_initial_array)
        temperature_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_ti.from_numpy(temperature)
        result_array = ti.ndarray(dtype=ti.f64, shape=array_length)

        python_stats = get_function_benchmark(
            lambda: python_get_clausius_clapeyron_vapor_pressure(
                latent_heat_value,
                temperature_initial_value,
                pressure_initial_value,
                temperature,
            ),
            ops_per_call=array_length,
        )
        taichi_stats = get_function_benchmark(
            lambda: taichi_get_clausius_clapeyron_vapor_pressure(
                latent_heat_value,
                temperature_initial_value,
                pressure_initial_value,
                temperature,
            ),
            ops_per_call=array_length,
        )
        kernel_stats = get_function_benchmark(
            lambda: kget_clausius_clapeyron_vapor_pressure(
                latent_heat_ti,
                temperature_initial_ti,
                pressure_initial_ti,
                temperature_ti,
                GAS_CONSTANT,
                result_array,
            ),
            ops_per_call=array_length,
        )
        result_rows.append([
            array_length,
            *python_stats["array_stats"],
            *taichi_stats["array_stats"],
            *kernel_stats["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in python_stats["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in taichi_stats["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in kernel_stats["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)
    save_combined_csv(
        os.path.join(output_directory, "clausius_clapeyron_benchmark.csv"),
        header, result_rows
    )
    with open(
        os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header, result_rows, "Clausius-Clapeyron throughput benchmark",
        os.path.join(output_directory, "clausius_clapeyron_benchmark.png"),
    )

# -------------------------------------------------------------------------
# Benchmark #3 – Buck vapour pressure
# -------------------------------------------------------------------------
def benchmark_buck_vapor_pressure_csv():
    """
    Benchmark the Buck pure-Python, Taichi-Python, and Taichi-kernel
    implementations over array sizes 10²–10⁸.

    The routine writes a CSV file, stores system information, and creates an
    annotated throughput plot in ./benchmark_outputs.

    Returns
    -------
    None

    Examples
    --------
    >>> benchmark_buck_vapor_pressure_csv()

    """
    result_rows: list[list] = []
    random_generator = np.random.default_rng(seed=RANDOM_SEED)
    for array_length in ARRAY_LENGTHS:
        temperature = (
            random_generator.random(array_length, dtype=np.float64) * 200.0 + 250.0
        )
        temperature_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_ti.from_numpy(temperature)
        result_array = ti.ndarray(dtype=ti.f64, shape=array_length)

        python_stats = get_function_benchmark(
            lambda: python_get_buck_vapor_pressure(temperature),
            ops_per_call=array_length,
        )
        taichi_stats = get_function_benchmark(
            lambda: taichi_get_buck_vapor_pressure(temperature),
            ops_per_call=array_length,
        )
        kernel_stats = get_function_benchmark(
            lambda: kget_buck_vapor_pressure(temperature_ti, result_array),
            ops_per_call=array_length,
        )
        result_rows.append([
            array_length,
            *python_stats["array_stats"],
            *taichi_stats["array_stats"],
            *kernel_stats["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in python_stats["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in taichi_stats["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in kernel_stats["array_headers"]]
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)
    save_combined_csv(
        os.path.join(output_directory, "buck_vapor_pressure_benchmark.csv"),
        header, result_rows
    )
    with open(
        os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header, result_rows, "Buck vapor-pressure throughput benchmark",
        os.path.join(output_directory, "buck_vapor_pressure_benchmark.png"),
    )

# -------------------------------------------------------------------------
# Entrypoint guard
# -------------------------------------------------------------------------
if __name__ == "__main__":
    benchmark_antoine_vapor_pressure_csv()
    benchmark_clausius_clapeyron_vapor_pressure_csv()
    benchmark_buck_vapor_pressure_csv()
