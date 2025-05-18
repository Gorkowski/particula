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
    import TiCondensationIsothermal
