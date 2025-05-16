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
    get_partial_pressure                as py_get_partial_pressure,
    get_saturation_ratio_from_pressure  as py_get_saturation_ratio,
)

# ── Taichi wrapper & raw-kernel implementations ────────────────────────────
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    ti_get_partial_pressure,
    ti_get_saturation_ratio_from_pressure,
    kget_partial_pressure,
    kget_saturation_ratio_from_pressure,
)

# ── reproducibility / Taichi backend ───────────────────────────────────────
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)      # 10² … 10⁸
ti.init(arch=ti.cpu)


# ═══════════════════════════════════════════════════════════════════════════
def benchmark_partial_pressure_csv() -> None:
    """Benchmark *get_partial_pressure* variants and save CSV + PNG."""
    rows: list[list[float]] = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ── random input data ────────────────────────────────────────────
        conc = rng.random(n, dtype=np.float64) + 1e-9                     # kg m⁻³
        mw   = rng.random(n, dtype=np.float64) * 0.05 + 0.01              # kg mol⁻¹
        temp = rng.random(n, dtype=np.float64) * 100.0 + 273.0            # K

        # ── Taichi buffers (create once per length) ─────────────────────
        conc_ti = ti.ndarray(dtype=ti.f64, shape=n); conc_ti.from_numpy(conc)
        mw_ti   = ti.ndarray(dtype=ti.f64, shape=n); mw_ti.from_numpy(mw)
        temp_ti = ti.ndarray(dtype=ti.f64, shape=n); temp_ti.from_numpy(temp)
        res_ti  = ti.ndarray(dtype=ti.f64, shape=n)

        # ── timing ───────────────────────────────────────────────────────
        stats_py     = get_function_benchmark(
            lambda: py_get_partial_pressure(conc, mw, temp), n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_partial_pressure(conc, mw, temp), n
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_partial_pressure(conc_ti, mw_ti, temp_ti, res_ti), n
        )

        # ── collect CSV row ──────────────────────────────────────────────
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ── header construction ────────────────────────────────────────────────
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ── output directory & CSV ─────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "partial_pressure_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # ── system-info JSON ───────────────────────────────────────────────────
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ── throughput figure ─────────────────────────────────────────────────
    plot_throughput_vs_array_length(
        header,
        rows,
        "Partial-pressure throughput benchmark",
        os.path.join(out_dir, "partial_pressure_benchmark.png"),
    )


# ═══════════════════════════════════════════════════════════════════════════
def benchmark_saturation_ratio_csv() -> None:
    """Benchmark *get_saturation_ratio_from_pressure* variants and save CSV + PNG."""
    rows: list[list[float]] = []
    rng = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        # ── random input data ────────────────────────────────────────────
        partial_p = rng.random(n, dtype=np.float64) * 1.0e3 + 1.0          # Pa
        pure_vp   = partial_p + rng.random(n, dtype=np.float64) * 1.0e3    # Pa

        # ── Taichi buffers ───────────────────────────────────────────────
        pp_ti  = ti.ndarray(dtype=ti.f64, shape=n); pp_ti.from_numpy(partial_p)
        vp_ti  = ti.ndarray(dtype=ti.f64, shape=n); vp_ti.from_numpy(pure_vp)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)

        # ── timing ───────────────────────────────────────────────────────
        stats_py     = get_function_benchmark(
            lambda: py_get_saturation_ratio(partial_p, pure_vp), n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_get_saturation_ratio_from_pressure(partial_p, pure_vp), n
        )
        stats_kernel = get_function_benchmark(
            lambda: kget_saturation_ratio_from_pressure(pp_ti, vp_ti, res_ti), n
        )

        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "saturation_ratio_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header,
        rows,
        "Saturation-ratio throughput benchmark",
        os.path.join(out_dir, "saturation_ratio_benchmark.png"),
    )


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    benchmark_partial_pressure_csv()
    benchmark_saturation_ratio_csv()
