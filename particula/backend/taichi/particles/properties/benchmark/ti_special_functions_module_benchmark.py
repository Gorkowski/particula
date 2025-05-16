import os, json, numpy as np, taichi as ti
from particula.backend.benchmark import (
    get_function_benchmark, get_system_info,
    save_combined_csv, plot_throughput_vs_array_length,
)
from particula.particles.properties.special_functions import (
    get_debye_function as py_func,
)
from particula.backend.taichi.particles.properties.ti_special_functions import (
    ti_get_debye_function as ti_func,
    kget_debye_function  as ti_kernel,
)

RNG_SEED      = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)

def benchmark_debye_function_csv():
    """Benchmark pure-Python, Taichi wrapper, and raw kernel for Debye function."""
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    exponent = 1.5

    for n in ARRAY_LENGTHS:
        x = rng.uniform(0.05, 5.0, size=n).astype(np.float64)
        x_ti   = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        x_ti.from_numpy(x)

        stats_py     = get_function_benchmark(lambda: py_func(x, n=exponent), ops_per_call=n)
        stats_ti     = get_function_benchmark(lambda: ti_func(x, n=exponent), ops_per_call=n)
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(x_ti, exponent, res_ti), ops_per_call=n)

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

    csv_path = os.path.join(out_dir, "debye_function_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    plot_throughput_vs_array_length(
        header, rows,
        "Debye-function throughput benchmark",
        os.path.join(out_dir, "debye_function_benchmark.png"),
    )

if __name__ == "__main__":
    benchmark_debye_function_csv()
