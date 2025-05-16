# standard libs
import os, json
import numpy as np
import taichi as ti

# benchmark helpers
from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

# reference-python function
from particula.particles import get_mixing_state_index as py_func

# Taichi wrapper + raw kernel
from particula.backend.taichi.particles.properties.ti_mixing_state_index_module import (
    ti_get_mixing_state_index as ti_func,
    kget_mixing_state_index   as ti_kernel,
)

RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 7, 6, dtype=int)   # 10² … 10⁷  (2-D input)
N_SPECIES = 3
ti.init(arch=ti.cpu)

def _write_results(name: str, header, rows):
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}_benchmark.csv")
    save_combined_csv(csv_path, header, rows)
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)
    plot_throughput_vs_array_length(
        header, rows,
        f"{name.replace('_',' ').title()} throughput benchmark",
        os.path.join(out_dir, f"{name}_benchmark.png")
    )

def benchmark_mixing_state_index_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    for n in ARRAY_LENGTHS:
        m = rng.random((n, N_SPECIES), dtype=np.float64) + 1e-15   # avoid zeroes

        # Taichi buffers
        m_ti   = ti.ndarray(dtype=ti.f64, shape=(n, N_SPECIES))
        out_ti = ti.ndarray(dtype=ti.f64, shape=1)
        m_ti.from_numpy(m)

        stats_py     = get_function_benchmark(
            lambda: py_func(m),                       ops_per_call=n * N_SPECIES
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(m),                       ops_per_call=n * N_SPECIES
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(m_ti, n, N_SPECIES, out_ti),
            ops_per_call=n * N_SPECIES
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
    header     = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    _write_results("mixing_state_index", header, rows)

if __name__ == "__main__":
    benchmark_mixing_state_index_csv()
