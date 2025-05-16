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
    get_lagrangian_taylor_microscale_time      as py_tau_L,
    get_taylor_microscale                      as py_lambda,
    get_taylor_microscale_reynolds_number      as py_Re_lambda,
)

# Taichi wrapper functions + raw kernels
from particula.backend.taichi.gas.properties.ti_taylor_microscale_module import (
    ti_get_lagrangian_taylor_microscale_time   as ti_tau_L,
    ti_get_taylor_microscale                   as ti_lambda,
    ti_get_taylor_microscale_reynolds_number   as ti_Re_lambda,
    kget_lagrangian_taylor_microscale_time,
    kget_taylor_microscale,
    kget_taylor_microscale_reynolds_number,
)

# ------------------------- 2. Benchmark config -----------------------------
RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)     # 10² … 10⁸
ti.init(arch=ti.cpu)


# ------------------------- 3. Helpers --------------------------------------
def _bench_loop(gen_data, py_call, ti_call, kernel_call):
    """Run benchmark loop for one function, return CSV rows & last stats."""
    rows = []
    for n in ARRAY_LENGTHS:
        np_args, ti_args, res_ti = gen_data(n)
        stats_py = get_function_benchmark(lambda: py_call(*np_args), ops_per_call=n)
        stats_ti = get_function_benchmark(lambda: ti_call(*np_args), ops_per_call=n)
        stats_ke = get_function_benchmark(
            lambda: kernel_call(*ti_args, res_ti), ops_per_call=n
        )
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_ke["array_stats"],
        ])
    return rows, stats_py, stats_ti, stats_ke


def _save_outputs(stem, rows, stats_py, stats_ti, stats_ke):
    """Write CSV, JSON, and PNG for one benchmark."""
    python_hdr = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr = ["taichi_kernel_" + h for h in stats_ke["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    save_combined_csv(
        os.path.join(out_dir, f"{stem}_benchmark.csv"), header, rows
    )
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows, f"{stem} throughput benchmark",
        os.path.join(out_dir, f"{stem}_benchmark.png"),
    )


# ------------------------- 4. Benchmarks -----------------------------------
def benchmark_lagrangian_taylor_microscale_time_csv():
    """Benchmark get_lagrangian_taylor_microscale_time."""
    rng = np.random.default_rng(RNG_SEED)

    def gen(n):
        kt = rng.random(n, dtype=np.float64) + 1e-9
        rl = rng.random(n, dtype=np.float64) * 1e3 + 100.0
        av = rng.random(n, dtype=np.float64) + 1e-9
        kt_ti = ti.ndarray(ti.f64, n); kt_ti.from_numpy(kt)
        rl_ti = ti.ndarray(ti.f64, n); rl_ti.from_numpy(rl)
        av_ti = ti.ndarray(ti.f64, n); av_ti.from_numpy(av)
        res_ti = ti.ndarray(ti.f64, n)
        return (kt, rl, av), (kt_ti, rl_ti, av_ti), res_ti

    rows, s_py, s_ti, s_ke = _bench_loop(
        gen, py_tau_L, ti_tau_L, kget_lagrangian_taylor_microscale_time
    )
    _save_outputs("lagrangian_taylor_microscale_time", rows, s_py, s_ti, s_ke)


def benchmark_taylor_microscale_csv():
    """Benchmark get_taylor_microscale."""
    rng = np.random.default_rng(RNG_SEED)

    def gen(n):
        u   = rng.random(n, dtype=np.float64) + 1e-9
        nu  = rng.random(n, dtype=np.float64) * 1e-5 + 1e-7
        eps = rng.random(n, dtype=np.float64) + 1e-9
        u_ti   = ti.ndarray(ti.f64, n); u_ti.from_numpy(u)
        nu_ti  = ti.ndarray(ti.f64, n); nu_ti.from_numpy(nu)
        eps_ti = ti.ndarray(ti.f64, n); eps_ti.from_numpy(eps)
        res_ti = ti.ndarray(ti.f64, n)
        return (u, nu, eps), (u_ti, nu_ti, eps_ti), res_ti

    rows, s_py, s_ti, s_ke = _bench_loop(
        gen, py_lambda, ti_lambda, kget_taylor_microscale
    )
    _save_outputs("taylor_microscale", rows, s_py, s_ti, s_ke)


def benchmark_taylor_microscale_reynolds_number_csv():
    """Benchmark get_taylor_microscale_reynolds_number."""
    rng = np.random.default_rng(RNG_SEED)

    def gen(n):
        u   = rng.random(n, dtype=np.float64) + 1e-9
        lam = rng.random(n, dtype=np.float64) + 1e-9
        nu  = rng.random(n, dtype=np.float64) * 1e-5 + 1e-7
        u_ti   = ti.ndarray(ti.f64, n); u_ti.from_numpy(u)
        lam_ti = ti.ndarray(ti.f64, n); lam_ti.from_numpy(lam)
        nu_ti  = ti.ndarray(ti.f64, n); nu_ti.from_numpy(nu)
        res_ti = ti.ndarray(ti.f64, n)
        return (u, lam, nu), (u_ti, lam_ti, nu_ti), res_ti

    rows, s_py, s_ti, s_ke = _bench_loop(
        gen, py_Re_lambda, ti_Re_lambda, kget_taylor_microscale_reynolds_number
    )
    _save_outputs("taylor_microscale_reynolds_number", rows, s_py, s_ti, s_ke)


# ------------------------- 5. Entrypoint -----------------------------------
if __name__ == "__main__":
    benchmark_lagrangian_taylor_microscale_time_csv()
    benchmark_taylor_microscale_csv()
    benchmark_taylor_microscale_reynolds_number_csv()
