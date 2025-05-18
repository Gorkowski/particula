"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel."""
# ------------------------- 1. Imports --------------------------------------
import os
import json
import numpy as np
import taichi as ti
from particula.backend.benchmark import (                 # helper utilities
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

# pure-Python reference functions
from particula.gas import (
    get_lagrangian_taylor_microscale_time      as python_get_lagrangian_taylor_microscale_time,
    get_taylor_microscale                      as python_get_taylor_microscale,
    get_taylor_microscale_reynolds_number      as python_get_taylor_microscale_reynolds_number,
)

# Taichi wrapper functions + raw kernels
from particula.backend.taichi.gas.properties.ti_taylor_microscale_module import (
    ti_get_lagrangian_taylor_microscale_time   as taichi_get_lagrangian_taylor_microscale_time,
    ti_get_taylor_microscale                   as taichi_get_taylor_microscale,
    ti_get_taylor_microscale_reynolds_number   as taichi_get_taylor_microscale_reynolds_number,
    kget_lagrangian_taylor_microscale_time,
    kget_taylor_microscale,
    kget_taylor_microscale_reynolds_number,
)

# ------------------------- 2. Benchmark config -----------------------------
RANDOM_SEED   = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)     # 10² … 10⁸
ti.init(arch=ti.cpu)


# ------------------------- 3. Helpers --------------------------------------
def _benchmark_loop(generate_input_data, python_call, taichi_call, kernel_call):
    """Run benchmark loop for one function, return CSV rows & last stats."""
    rows = []
    for array_length in ARRAY_LENGTHS:
        numpy_args, taichi_args, result_ti_field = generate_input_data(array_length)
        stats_python = get_function_benchmark(
            lambda: python_call(*numpy_args), ops_per_call=array_length
        )
        stats_taichi = get_function_benchmark(
            lambda: taichi_call(*numpy_args), ops_per_call=array_length
        )
        stats_kernel = get_function_benchmark(
            lambda: kernel_call(*taichi_args, result_ti_field),
            ops_per_call=array_length,
        )
        rows.append([
            array_length,
            *stats_python["array_stats"],
            *stats_taichi["array_stats"],
            *stats_kernel["array_stats"],
        ])
    return rows, stats_python, stats_taichi, stats_kernel


def _save_outputs(benchmark_stem, rows, stats_python, stats_taichi, stats_kernel):
    """Write CSV, JSON, and PNG for one benchmark."""
    python_header = ["python_"        + h for h in stats_python["array_headers"]]
    taichi_header = ["taichi_"        + h for h in stats_taichi["array_headers"]]
    kernel_header = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_header, *taichi_header, *kernel_header]

    output_directory = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_directory, exist_ok=True)

    save_combined_csv(
        os.path.join(output_directory, f"{benchmark_stem}_benchmark.csv"),
        header,
        rows,
    )
    with open(
        os.path.join(output_directory, "system_info.json"), "w",
        encoding="utf-8",
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        f"{benchmark_stem} throughput benchmark",
        os.path.join(output_directory, f"{benchmark_stem}_benchmark.png"),
    )


# ------------------------- 4. Benchmarks -----------------------------------
def benchmark_lagrangian_taylor_microscale_time_csv():
    """Benchmark get_lagrangian_taylor_microscale_time."""
    random_generator = np.random.default_rng(RANDOM_SEED)

    def generate_input_data(array_length):
        kinetic_energy_array = (
            random_generator.random(array_length, dtype=np.float64) + 1e-9
        )
        length_scale_array = (
            random_generator.random(array_length, dtype=np.float64) * 1e3 + 100.0
        )
        auxiliary_value_array = (
            random_generator.random(array_length, dtype=np.float64) + 1e-9
        )

        kinetic_energy_ti_field = ti.ndarray(ti.f64, array_length)
        kinetic_energy_ti_field.from_numpy(kinetic_energy_array)
        length_scale_ti_field = ti.ndarray(ti.f64, array_length)
        length_scale_ti_field.from_numpy(length_scale_array)
        auxiliary_value_ti_field = ti.ndarray(ti.f64, array_length)
        auxiliary_value_ti_field.from_numpy(auxiliary_value_array)

        result_ti_field = ti.ndarray(ti.f64, array_length)

        numpy_args = (
            kinetic_energy_array,
            length_scale_array,
            auxiliary_value_array,
        )
        taichi_args = (
            kinetic_energy_ti_field,
            length_scale_ti_field,
            auxiliary_value_ti_field,
        )
        return numpy_args, taichi_args, result_ti_field

    rows, stats_python, stats_taichi, stats_kernel = _benchmark_loop(
        generate_input_data,
        python_get_lagrangian_taylor_microscale_time,
        taichi_get_lagrangian_taylor_microscale_time,
        kget_lagrangian_taylor_microscale_time,
    )
    _save_outputs(
        "lagrangian_taylor_microscale_time",
        rows,
        stats_python,
        stats_taichi,
        stats_kernel,
    )


def benchmark_taylor_microscale_csv():
    """Benchmark get_taylor_microscale."""
    random_generator = np.random.default_rng(RANDOM_SEED)

    def generate_input_data(array_length):
        velocity_array = (
            random_generator.random(array_length, dtype=np.float64) + 1e-9
        )
        kinematic_viscosity_array = (
            random_generator.random(array_length, dtype=np.float64) * 1e-5 + 1e-7
        )
        energy_dissipation_array = (
            random_generator.random(array_length, dtype=np.float64) + 1e-9
        )

        velocity_ti_field = ti.ndarray(ti.f64, array_length)
        velocity_ti_field.from_numpy(velocity_array)
        kinematic_viscosity_ti_field = ti.ndarray(ti.f64, array_length)
        kinematic_viscosity_ti_field.from_numpy(kinematic_viscosity_array)
        energy_dissipation_ti_field = ti.ndarray(ti.f64, array_length)
        energy_dissipation_ti_field.from_numpy(energy_dissipation_array)

        result_ti_field = ti.ndarray(ti.f64, array_length)

        numpy_args = (
            velocity_array,
            kinematic_viscosity_array,
            energy_dissipation_array,
        )
        taichi_args = (
            velocity_ti_field,
            kinematic_viscosity_ti_field,
            energy_dissipation_ti_field,
        )
        return numpy_args, taichi_args, result_ti_field

    rows, stats_python, stats_taichi, stats_kernel = _benchmark_loop(
        generate_input_data,
        python_get_taylor_microscale,
        taichi_get_taylor_microscale,
        kget_taylor_microscale,
    )
    _save_outputs(
        "taylor_microscale",
        rows,
        stats_python,
        stats_taichi,
        stats_kernel,
    )


def benchmark_taylor_microscale_reynolds_number_csv():
    """Benchmark get_taylor_microscale_reynolds_number."""
    random_generator = np.random.default_rng(RANDOM_SEED)

    def generate_input_data(array_length):
        velocity_array = (
            random_generator.random(array_length, dtype=np.float64) + 1e-9
        )
        taylor_microscale_array = (
            random_generator.random(array_length, dtype=np.float64) + 1e-9
        )
        kinematic_viscosity_array = (
            random_generator.random(array_length, dtype=np.float64) * 1e-5 + 1e-7
        )

        velocity_ti_field = ti.ndarray(ti.f64, array_length)
        velocity_ti_field.from_numpy(velocity_array)
        taylor_microscale_ti_field = ti.ndarray(ti.f64, array_length)
        taylor_microscale_ti_field.from_numpy(taylor_microscale_array)
        kinematic_viscosity_ti_field = ti.ndarray(ti.f64, array_length)
        kinematic_viscosity_ti_field.from_numpy(kinematic_viscosity_array)

        result_ti_field = ti.ndarray(ti.f64, array_length)

        numpy_args = (
            velocity_array,
            taylor_microscale_array,
            kinematic_viscosity_array,
        )
        taichi_args = (
            velocity_ti_field,
            taylor_microscale_ti_field,
            kinematic_viscosity_ti_field,
        )
        return numpy_args, taichi_args, result_ti_field

    rows, stats_python, stats_taichi, stats_kernel = _benchmark_loop(
        generate_input_data,
        python_get_taylor_microscale_reynolds_number,
        taichi_get_taylor_microscale_reynolds_number,
        kget_taylor_microscale_reynolds_number,
    )
    _save_outputs(
        "taylor_microscale_reynolds_number",
        rows,
        stats_python,
        stats_taichi,
        stats_kernel,
    )


# ------------------------- 5. Entrypoint -----------------------------------
if __name__ == "__main__":
    benchmark_lagrangian_taylor_microscale_time_csv()
    benchmark_taylor_microscale_csv()
    benchmark_taylor_microscale_reynolds_number_csv()
