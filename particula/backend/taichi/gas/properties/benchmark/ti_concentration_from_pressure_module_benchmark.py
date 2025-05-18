"""
Benchmark throughput of concentration-from-pressure routines implemented
in pure-Python and Taichi.

The script measures element-throughput for three variants
(Python, Taichi wrapper, raw Taichi kernel) across several
input-array lengths and stores the results as CSV, JSON and PNG
in ``./benchmark_outputs``.

References:
    - Y. Hu et al., “Taichi: A Unified, Efficient, and Portable
      Programming Framework”, ACM TOG 38 (3), 2019.
"""
# ── imports ────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (                       # utilities
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

# functions to benchmark
from particula.gas.properties.concentration_function import (
    get_concentration_from_pressure as python_get_concentration_from_pressure,
)
from particula.backend.taichi.gas.properties.ti_concentration_from_pressure_module import (  # noqa: E501
    ti_get_concentration_from_pressure
    as taichi_get_concentration_from_pressure,
    kget_concentration_from_pressure
    as taichi_kernel_get_concentration_from_pressure,
)

# ── benchmark configuration ────────────────────────────────────────────────
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)  # 10² … 10⁸
ti.init(arch=ti.cpu)


def benchmark_concentration_from_pressure_to_csv() -> None:
    """
    Execute the benchmark and write artefacts to ``./benchmark_outputs``.

    The routine iterates over ``ARRAY_LENGTHS``, builds random yet
    physically plausible test vectors, measures throughput for the three
    back-ends via ``get_function_benchmark``, aggregates the results, and
    saves them as CSV, JSON, and PNG.

    Arguments:
        - None

    Returns:
        - None

    Examples:
        ```py
        from particula.backend.taichi.gas.properties.benchmark import (
            ti_concentration_from_pressure_module_benchmark as bm,
        )
        bm.benchmark_concentration_from_pressure_to_csv()
        ```
    """
    rows: list[list[float]] = []
    random_generator = np.random.default_rng(seed=RNG_SEED)

    # ── loop over array lengths ────────────────────────────────────────────
    for array_length in ARRAY_LENGTHS:
        # random input data (physically sensible ranges)
        partial_pressure_array = (
            random_generator.random(array_length, dtype=np.float64) * 1.0e5
            + 1.0
        )  # Pa
        molar_mass_array = (
            random_generator.random(array_length, dtype=np.float64) * 0.04
            + 0.002
        )  # kg mol⁻¹
        temperature_array = (
            random_generator.random(array_length, dtype=np.float64) * 300.0
            + 200.0
        )  # K

        # Taichi buffers
        partial_pressure_field = ti.ndarray(dtype=ti.f64, shape=array_length)
        molar_mass_field       = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_field      = ti.ndarray(dtype=ti.f64, shape=array_length)
        concentration_field    = ti.ndarray(dtype=ti.f64, shape=array_length)

        partial_pressure_field.from_numpy(partial_pressure_array)
        molar_mass_field.from_numpy(molar_mass_array)
        temperature_field.from_numpy(temperature_array)

        # timing
        stats_python = get_function_benchmark(
            lambda: python_get_concentration_from_pressure(
                partial_pressure_array, molar_mass_array, temperature_array
            ),
            ops_per_call=array_length,
        )
        stats_taichi = get_function_benchmark(
            lambda: taichi_get_concentration_from_pressure(
                partial_pressure_array, molar_mass_array, temperature_array
            ),
            ops_per_call=array_length,
        )
        stats_taichi_kernel = get_function_benchmark(
            lambda: taichi_kernel_get_concentration_from_pressure(
                partial_pressure_field,
                molar_mass_field,
                temperature_field,
                concentration_field,
            ),
            ops_per_call=array_length,
        )

        # collect CSV row
        rows.append(
            [
                array_length,
                *stats_python["array_stats"],
                *stats_taichi["array_stats"],
                *stats_taichi_kernel["array_stats"],
            ]
        )

    # ── header construction ────────────────────────────────────────────────
    python_headers        = ["python_" + h for h in stats_python["array_headers"]]
    taichi_headers        = ["taichi_" + h for h in stats_taichi["array_headers"]]
    taichi_kernel_headers = ["taichi_kernel_" + h for h in stats_taichi_kernel["array_headers"]]
    csv_header = ["array_length", *python_headers, *taichi_headers, *taichi_kernel_headers]

    # ── output directory ───────────────────────────────────────────────────
    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)

    # ── CSV ────────────────────────────────────────────────────────────────
    csv_file_path    = os.path.join(output_directory, "concentration_from_pressure_benchmark.csv")
    save_combined_csv(csv_file_path, csv_header, rows)

    # ── system-info JSON ───────────────────────────────────────────────────
    with open(os.path.join(output_directory, "system_info.json"), "w", encoding="utf-8") as file_handle:
        json.dump(get_system_info(), file_handle, indent=2)

    # ── throughput plot ────────────────────────────────────────────────────
    plot_throughput_vs_array_length(
        csv_header,
        rows,
        "Concentration-from-pressure throughput benchmark",
        os.path.join(output_directory, "concentration_from_pressure_benchmark.png"),
    )


if __name__ == "__main__":        # entry-point guard
    benchmark_concentration_from_pressure_to_csv()
