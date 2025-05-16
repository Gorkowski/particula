# Benchmark-Script Development Guide

How to build a **stand-alone benchmark** that times three
implementations of the same numeric routine—**pure Python**, **Taichi wrapper**,
and the **raw Taichi kernel**—and then stores timing data and a throughput
figure.

---

## 1. File location & name

Use the unit test file <name>_test.py as a template for the benchmark script.

Create the script in the `../benchmark/` folder, next to the module <name> you are benchmarking and prefix it with
`<name>_benchmark.py`.

Example

```
particula/backend_/particles/properties/test/ti_knudsen_number_test.py
→ particula/backend/taichi/particles/properties/benchmark/ti_knudsen_number_benchmark.py
```

---

## 2. Required imports

```python
"""Benchmarks the reference Python, Taichi wrapper, and raw Taichi kernel."""
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

# functions to benchmark
from particula.particles.properties.<name> import get_<name> as py_func
from particula.backend.taichi.particles.properties.ti_<name> import (
    ti_get_<name> as ti_func,
    kget_<name>   as ti_kernel,
)
```

---

## 3. Benchmark configuration

The benchmark script must evaulate the function over a range of input sizes.
This changes depending on the function, but a good starting point is to use
`np.logspace(2, 8, 10, dtype=int)` to get 10 different array lengths from 10² to 10⁸.
If the input is a 2D array us a smaller range.


```python
# -- fixed RNG and Taichi backend for reproducibility -------------------
RNG_SEED = 42
ARRAY_LENGTHS = np.logspace(2, 8, 10, dtype=int)   # 10² … 10⁸
ti.init(arch=ti.cpu)
```


## 5. Main benchmark routine

Example of a benchmark function for the `ti_knudsen_number_module_test.py`.

```python
def benchmark_<name>_csv():
    """
    Time pure-Python, Taichi wrapper, and raw kernel over ARRAY_LENGTHS,
    then save CSV, JSON, and PNG into ./benchmark_outputs/.
    """
    rows: list[list[float]] = []
    rng  = np.random.default_rng(seed=RNG_SEED)

    # ------------------------------------------------------------------ #
    #  LOOP OVER ARRAY LENGTHS – *no separate helper function*           #
    # ------------------------------------------------------------------ #
    for n in ARRAY_LENGTHS:
        # ----- random input data ---------------------------------------
        mfp = rng.random(n, dtype=np.float64) + 1e-9
        pr  = rng.random(n, dtype=np.float64) + 1e-9

        # ----- Taichi buffers (create once per length) -----------------
        mfp_ti = ti.ndarray(dtype=ti.f64, shape=n)
        pr_ti  = ti.ndarray(dtype=ti.f64, shape=n)
        res_ti = ti.ndarray(dtype=ti.f64, shape=n)
        mfp_ti.from_numpy(mfp)
        pr_ti.from_numpy(pr)

        # ----- timing --------------------------------------------------
        stats_py     = get_function_benchmark(
            lambda: py_func(mfp, pr), ops_per_call=n
        )
        stats_ti     = get_function_benchmark(
            lambda: ti_func(mfp, pr), ops_per_call=n
        )
        stats_kernel = get_function_benchmark(
            lambda: ti_kernel(mfp_ti, pr_ti, res_ti), ops_per_call=n
        )

        # ----- collect one CSV row ------------------------------------
        rows.append([
            n,
            *stats_py["array_stats"],
            *stats_ti["array_stats"],
            *stats_kernel["array_stats"],
        ])

    # ------------------------ header construction ----------------------
    python_hdr  = ["python_"        + h for h in stats_py["array_headers"]]
    taichi_hdr  = ["taichi_"        + h for h in stats_ti["array_headers"]]
    kernel_hdr  = ["taichi_kernel_" + h for h in stats_kernel["array_headers"]]
    header = ["array_length", *python_hdr, *taichi_hdr, *kernel_hdr]

    # ------------------------ output directory -------------------------
    out_dir = os.path.join(os.path.dirname(__file__), "benchmark_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------ CSV --------------------------------------
    csv_path = os.path.join(out_dir, "<name>_benchmark.csv")
    save_combined_csv(csv_path, header, rows)

    # ------------------------ system info JSON -------------------------
    with open(os.path.join(out_dir, "system_info.json"),
              "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)

    # ------------------------ throughput plot --------------------------
    plot_throughput_vs_array_length(
        header,
        rows,
        "<Name> throughput benchmark",
        os.path.join(out_dir, "<name>_benchmark.png"),
    )

```

---

## 5. More Functions

If the module has more than one function, add them to the benchmark script
as separate benchmark functions.

---

## 6. Entrypoint guard

```python
if __name__ == "__main__":


    benchmark_<name>_csv()
    # additional benchmark functions if needed
    # e.g. benchmark_<other_name>_csv()
```

---

## 7. Code-quality checklist

| ✓ | Requirement                                                 |
| - | ----------------------------------------------------------- |
|   | Public functions have concise one-sentence docstrings.      |
|   | Only **NumPy**, **Taichi**, and **Particula** are imported. |
|   | Script runs on CPU without modification.                    |

---

## 8. Run & verify

```bash
python particula/.../<name>_benchmark.py
ls particula/.../benchmark_outputs/
# → <name>_benchmark.csv  system_info.json  <name>_benchmark.png
```


---

## Help

If the function calls is not clear from the test, then read the module
directly. But do not modify the module itself.