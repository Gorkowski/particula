"""Benchmarks partial-pressure and saturation-ratio routines
(pure-Python, Taichi wrapper, Taichi kernel)."""
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

# ── reference (pure-Python) implementations ────────────────────────────────
from particula.gas.properties.pressure_function import (
    get_partial_pressure                as python_get_partial_pressure,
    get_saturation_ratio_from_pressure  as python_get_saturation_ratio,
)

# ── Taichi wrapper & raw-kernel implementations ────────────────────────────
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    ti_get_partial_pressure,
    ti_get_saturation_ratio_from_pressure,
    kget_partial_pressure,
    kget_saturation_ratio_from_pressure,
)

# ── reproducibility / Taichi backend ───────────────────────────────────────
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)  # 10² … 10⁸
ti.init(arch=ti.cpu)


# ═══════════════════════════════════════════════════════════════════════════
def benchmark_partial_pressure_csv() -> None:
    """Benchmark *get_partial_pressure* variants and save CSV + PNG."""
    rows: list[list[float]] = []
    random_generator = np.random.default_rng(seed=RNG_SEED)

    for array_length in ARRAY_LENGTHS:
        # ── random input data ────────────────────────────────────────────
        concentration = (
            random_generator.random(array_length, dtype=np.float64) + 1e-9
        )  # kg m⁻³
        molar_mass = (
            random_generator.random(array_length, dtype=np.float64) * 0.05 + 0.01
        )  # kg mol⁻¹
        temperature = (
            random_generator.random(array_length, dtype=np.float64) * 100.0 + 273.0
        )  # K

        # ── Taichi buffers (create once per length) ─────────────────────
        concentration_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        concentration_ti.from_numpy(concentration)
        molar_mass_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        molar_mass_ti.from_numpy(molar_mass)
        temperature_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        temperature_ti.from_numpy(temperature)
        result_ti = ti.ndarray(dtype=ti.f64, shape=array_length)

        # ── timing ───────────────────────────────────────────────────────
        stats_python = get_function_benchmark(
            lambda: python_get_partial_pressure(concentration, molar_mass, temperature),
            array_length,
        )
        stats_taichi = get_function_benchmark(
            lambda: ti_get_partial_pressure(concentration, molar_mass, temperature),
            array_length,
        )
        stats_taichi_kernel = get_function_benchmark(
            lambda: kget_partial_pressure(
                concentration_ti, molar_mass_ti, temperature_ti, result_ti
            ),
            array_length,
        )

        # ── collect CSV row ──────────────────────────────────────────────
        rows.append([
            array_length,
            *stats_python["array_stats"],
            *stats_taichi["array_stats"],
            *stats_taichi_kernel["array_stats"],
        ])

    # ── header construction ────────────────────────────────────────────────
    python_header = ["python_" + h for h in stats_python["array_headers"]]
    taichi_header = ["taichi_" + h for h in stats_taichi["array_headers"]]
    taichi_kernel_header = [
        "taichi_kernel_" + h for h in stats_taichi_kernel["array_headers"]
    ]
    header = ["array_length", *python_header, *taichi_header, *taichi_kernel_header]

    # ── output directory & CSV ─────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "partial_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # ── system-info JSON ───────────────────────────────────────────────────
    with open(
        os.path.join(output_dir, "system_info.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ── throughput figure ─────────────────────────────────────────────────
    plot_throughput_vs_array_length(
        header,
        rows,
        "Partial-pressure throughput benchmark",
        os.path.join(output_dir, "partial_pressure_benchmark.png"),
    )


# ═══════════════════════════════════════════════════════════════════════════
def benchmark_saturation_ratio_csv() -> None:
    """Benchmark *get_saturation_ratio_from_pressure* variants and save CSV + PNG."""
    rows: list[list[float]] = []
    random_generator = np.random.default_rng(seed=RNG_SEED)

    for array_length in ARRAY_LENGTHS:
        # ── random input data ────────────────────────────────────────────
        partial_pressure = (
            random_generator.random(array_length, dtype=np.float64) * 1.0e3 + 1.0
        )  # Pa
        pure_vapor_pressure = (
            partial_pressure
            + random_generator.random(array_length, dtype=np.float64) * 1.0e3
        )  # Pa

        # ── Taichi buffers ───────────────────────────────────────────────
        partial_pressure_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        partial_pressure_ti.from_numpy(partial_pressure)
        pure_vapor_pressure_ti = ti.ndarray(dtype=ti.f64, shape=array_length)
        pure_vapor_pressure_ti.from_numpy(pure_vapor_pressure)
        result_ti = ti.ndarray(dtype=ti.f64, shape=array_length)

        # ── timing ───────────────────────────────────────────────────────
        stats_python = get_function_benchmark(
            lambda: python_get_saturation_ratio(partial_pressure, pure_vapor_pressure),
            array_length,
        )
        stats_taichi = get_function_benchmark(
            lambda: ti_get_saturation_ratio_from_pressure(
                partial_pressure, pure_vapor_pressure
            ),
            array_length,
        )
        stats_taichi_kernel = get_function_benchmark(
            lambda: kget_saturation_ratio_from_pressure(
                partial_pressure_ti, pure_vapor_pressure_ti, result_ti
            ),
            array_length,
        )

        rows.append([
            array_length,
            *stats_python["array_stats"],
            *stats_taichi["array_stats"],
            *stats_taichi_kernel["array_stats"],
        ])

    python_header = ["python_" + h for h in stats_python["array_headers"]]
    taichi_header = ["taichi_" + h for h in stats_taichi["array_headers"]]
    taichi_kernel_header = [
        "taichi_kernel_" + h for h in stats_taichi_kernel["array_headers"]
    ]
    header = ["array_length", *python_header, *taichi_header, *taichi_kernel_header]

    output_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "saturation_ratio_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(
        os.path.join(output_dir, "system_info.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Saturation-ratio throughput benchmark",
        os.path.join(output_dir, "saturation_ratio_benchmark.png"),
    )


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    benchmark_partial_pressure_csv()
    benchmark_saturation_ratio_csv()
