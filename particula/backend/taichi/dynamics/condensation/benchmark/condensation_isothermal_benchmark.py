"""Benchmarks CondensationIsothermal.first_order_mass_transport."""
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

# reference Python implementation
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal as PyCondensationIsothermal,
)
# Taichi data-oriented replacement
from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies \
    import CondensationIsothermal as TiCondensationIsothermal

# mean-free-path helper needed for the raw kernel call
from particula.gas import get_molecule_mean_free_path

RNG_SEED        = 42
ARRAY_LENGTHS   = np.logspace(2, 8, 10, dtype=int)    # 10² … 10⁸
TEMPERATURE_K   = 298.15
PRESSURE_PA     = 101_325.0
MOLAR_MASS_KGPM = 0.018
ti.init(arch=ti.cpu)

def benchmark_first_order_mass_transport_csv():
    """
    Time pure-Python, Taichi wrapper, and raw Taichi kernel for
    first_order_mass_transport over ARRAY_LENGTHS and save results.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    # instantiate once – avoids counting constructor time
    py_impl = PyCondensationIsothermal(molar_mass=MOLAR_MASS_KGPM)
    ti_impl = TiCondensationIsothermal(molar_mass=MOLAR_MASS_KGPM)

    for n in ARRAY_LENGTHS:
        # ── random radii (0.5–1.5 × 10⁻⁷ m) ───────────────────────────
        radius_np = (0.5 + rng.random(n)) * 1e-7

        # ── Taichi buffers ───────────────────────────────────────────
        radius_ti = ti.ndarray(dtype=ti.f64, shape=n)
        result_ti = ti.ndarray(dtype=ti.f64, shape=n)
        radius_ti.from_numpy(radius_np)

        # mean free path for the raw kernel
        mfp = get_molecule_mean_free_path(
            molar_mass=MOLAR_MASS_KGPM,
            temperature=TEMPERATURE_K,
            pressure=PRESSURE_PA,
        )

        # ── timing ───────────────────────────────────────────────────
        stats_py = get_function_benchmark(
            lambda: py_impl.first_order_mass_transport(
                radius_np, TEMPERATURE_K, PRESSURE_PA
            ),
            ops_per_call=n,
        )

        stats_ti = get_function_benchmark(
            lambda: ti_impl.first_order_mass_transport(
                radius_np, TEMPERATURE_K, PRESSURE_PA
            ),
            ops_per_call=n,
        )

        stats_kernel = get_function_benchmark(
            lambda: ti_impl._kget_first_order_mass_transport(
                radius_ti, mfp, result_ti
            ),
            ops_per_call=n,
        )

        # ── CSV row ──────────────────────────────────────────────────
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ── column headers ───────────────────────────────────────────────
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ── outputs (./benchmark_outputs/) ───────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "condensation_isothermal_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w",
              encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "CondensationIsothermal throughput benchmark",
        os.path.join(out_dir, "condensation_isothermal_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_first_order_mass_transport_csv()
