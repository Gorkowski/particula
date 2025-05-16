"""
Utility that discovers every *benchmark_* module inside
`particula.backend.taichi.particles.properties.benchmark`
and executes any public function whose name follows
`benchmark_*_csv` or equals `run_benchmark`.

Run it with:
    python run_benchmarks.py
"""

import importlib
import pkgutil
import inspect
from types import ModuleType

# package to scan -----------------------------------------------------------
PKG_PATH = "particula.backend.taichi"
FUNC_SUFFIX = "_csv"          # each benchmark function must end with this


def _run_functions_in_module(mod: ModuleType) -> None:
    """Invoke every selected benchmark function inside *mod*."""
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        if (
            name == "run_benchmark"
            or (name.startswith("benchmark_") and name.endswith(FUNC_SUFFIX))
        ):
            print(f"[benchmark] {mod.__name__}.{name}()")
            obj()


def run_all_benchmarks() -> None:
    """Import each benchmark_* module and execute its benchmark functions."""
    package = importlib.import_module(PKG_PATH)
    for mod_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if mod_info.ispkg:
            continue                       # ignore sub-packages
        if mod_info.name.split(".")[-1].startswith("benchmark_"):
            module = importlib.import_module(mod_info.name)
            _run_functions_in_module(module)


if __name__ == "__main__":
    run_all_benchmarks()
