### **Particula → Taichi Class Conversion Guide**

*Turn any NumPy-based “model” class into a high-performance, data-oriented Taichi class while preserving the original public API, using constructor-time dependency injection, and re-using existing `fget_…` element-wise helpers and `kget_…` vectorized kernels.*

---

## 0 · Concepts in One Minute

| Term                     | What it is                                                                                                                                           | Key rule inside Taichi                                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Dependency injection** | The class receives fully-initialised collaborator objects (e.g. `TiParticleSystem`, `TiGasSpecies`) via `__init__`, instead of creating them itself. | Collaborators **must store their mutable data in Taichi fields or ndarrays**, otherwise kernels cannot read them. |
| **`fget_*` helper**      | A small `@ti.func` that performs element-wise math.                                                                                                  | You can call it anywhere inside another `@ti.func` or a `@ti.kernel`.                                             |
| **`kget_*` helper**      | A vectorised `@ti.kernel` that operates on whole fields/ndarrays.                                                                                    | You can call it from Python *or* from another kernel.                                                             |
| **`snake_case` names**   | Variable or attribute names use plain English words with underscores.                                                                                | No abbreviations (`radius`, **not** `r`; `knudsen_number`, **not** `Kn`).                                         |
| **Mixins replace ABCs** | Taichi classes are not allowed to inherit from other classes.                                                                                       | Use mixins to share ti.func and ti.kernel code between classes.                                                                          |

---

## 1 · File & Naming Conventions

| Item                           | Rule                                                                                                                                                                                                                             |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Path**                       | Mirror the original module path under `particula/backend/taichi/…` and prepend `ti_` to the filename.<br>`particula/gas/vapor_pressure_strategies.py` → `particula/backend/taichi/gas/ti_vapor_pressure_strategies.py` |
| **Class name**                 | Keep the original class name. Add **TI** only if the original name is still imported elsewhere. If several subclasses share helpers, add one underscore-prefixed *mixin* class to hold those helpers (e.g. `_VaporPressureMixin`). Decorate every Taichi class with `@ti.data_oriented`. |
| **Variable / attribute names** | *Every* identifier must be a full English word or phrase in **`snake_case`**. Loop counters (`i`, `j`, `k`) are the only permitted one-letter names.                                                                             |
| **Helper imports**             | At module level, import any `fget_*` / `kget_*` helpers you have already written so the Taichi class can call them.                                                                                                              |

---

## 2 · Required Imports (template)

```python
"""Taichi implementation of particula.dynamics.condensation.Condensation."""

import taichi as ti
import numpy as np

# ─── Existing helpers you want to re-use ────────────────────────────────
from particula.backend.taichi.gas.properties.ti_vapor_pressure_module import (
    fget_antoine_vapor_pressure,
    fget_buck_vapor_pressure,
    fget_clausius_clapeyron_vapor_pressure,
)
from particula.backend.taichi.gas.properties.ti_pressure_function_module import (
    fget_partial_pressure,
    fget_concentration_from_pressure,
)

ti.init(default_fp=ti.f64)    # always work in float64 unless you *must* down-cast
```

---

## 3 · Skeleton of a Fully Injected Taichi Class (Mixin + Strategy)

```python
@ti.data_oriented
class _VaporPressureMixin:
    """
    Shared helpers reused by several pure-vapor-pressure strategies
    (Constant, Antoine, Clausius-Clapeyron, Buck, …).
    """

    # ── tiny element-wise helpers ─────────────────────────────────────
    @ti.func
    def _partial_pressure_func(self, concentration, molar_mass, temperature):
        return fget_partial_pressure(concentration, molar_mass, temperature)

    @ti.func
    def _concentration_func(self, partial_pressure, molar_mass, temperature):
        return fget_concentration_from_pressure(
            partial_pressure, molar_mass, temperature
        )

    # ── public wrappers identical to the NumPy API ────────────────────
    def partial_pressure(self, conc, m, T):
        return self._partial_pressure_kernel(conc, m, T)

    # … (omit remaining boilerplate for brevity)
```

```python
@ti.data_oriented
class ConstantVaporPressureStrategy(_VaporPressureMixin):
    """
    Taichi drop-in replacement for
    particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy
    """
    def __init__(self, vapor_pressure):
        self.vapor_pressure = ti.field(dtype=ti.f64, shape=())
        self.vapor_pressure[None] = vapor_pressure

    @ti.func
    def _pure_vp_func(self, temperature):
        return self.vapor_pressure[None]
```
(The Antoine / Buck / Clausius-Clapeyron subclasses follow the same
pattern, each calling its corresponding `fget_*` helper.)

---

## 4 · Method-Conversion Rules

| Original method did…                             | Convert to…                                                                        | Naming & notes                                              |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Pure element-wise math                           | `@ti.func`                                                                         | `kelvin_term`, `activity_coefficient`, etc.                 |
| Element-wise math already written as `fget_*`    | **Use it** – no rewrite                                                            | Wrap it if you need extra parameters.                       |
| Loops over arrays                                | Single `@ti.kernel` with nested `ti.ndrange`                                       | Gather all per-timestep work here.                          |
| Heavy vector physics already written in `kget_*` | Call it from your kernel via a thin wrapper                                        | Use full parameter names in wrapper arguments.              |
| Manipulates temporary arrays                     | Pre-allocate scratch `ti.field`s in `__init__` or a helper like `allocate_buffers` | Scratch names: `mass_rate_buffer`, `temperature_grid`, etc. |
| Needs new constant each call                     | Pass it as an argument to the kernel or store in a 0-D field                       | Never rely on Python-side globals inside kernels.           |
| ABCs to mixins (_underscore-prefixed_)             | Taichi classes cannot inherit from other classes                                   | Use mixins to share code between classes.                   |


---

## 5 · How to Wrap and Call an Existing `kget_*`

```python
@ti.kernel
def update_mass_transfer_rate(self):
    """
    Example: call vectorised helper on the whole particle set.
    """
    self.mass_transfer_rate(
        self.particle_system.radius,
        self.gas_species.diffusivity,
        self.mass_change,
    )
```

*If you need per-species calls, loop over `species_index` in a kernel and slice the appropriate field.*

---

## 6 · Unit-Testing Checklist

1. **Numerical parity**

   ```python
   np.testing.assert_allclose(
       numpy_impl.step(dt, **kwargs).get_state(),
       taichi_impl.advance(dt, **kwargs); taichi_impl.particle_system.export_state(),
       rtol=1e-12,
   )
   ```
2. **Dependency updates**

   ```python
   original = taichi_impl.advance(0.0).mass_change.to_numpy().copy()
   gas_species.molar_mass[0] *= 1.2          # modify in place
   taichi_impl.advance(1.0)
   assert not np.allclose(original, taichi_impl.mass_change.to_numpy())
   ```
3. **Direct kernel call** – run `_advance_one_timestep` on a 1-particle system and compare against a hand-computed answer.
4. **Isolated environment** – every test file must call

   ```python
   import taichi as ti
   ti.init(arch=ti.cpu, default_fp=ti.f64)
   ```

---

## 7 · Type & Style Rules

* **All numeric fields and kernel arguments are `ti.f64`.**
* Identifiers are full words in snake\_case.
  Examples: `mean_free_path`, `number_of_particles`, `delta_partial_pressure`.
* One-line docstring for every `@ti.kernel` or `@ti.func` describing *what it does*, not its implementation details.
* Avoid tables in docstrings unless they genuinely clarify behaviour.

---

### **Quick Reference Cheat-Sheet**

| Action                       | How                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| Inject collaborator          | `self.gas_species = gas_species`                                                    |
| Update a scalar at run-time  | `gas_species.molar_mass[None] = 0.020`                                              |
| Element-wise helper          | `@ti.func def kelvin_term(...): ...`                                                |
| Call `fget_*` inside helper  | `result = fget_kelvin_term(radius, temperature)`                                    |
| Call vectorised `kget_*`     | see `mass_transfer_rate` wrapper                                                    |
| Allocate scratch field later | `self.scratch = ti.field(ti.f64, shape=n_items)`                                    |
| Avoid recompilation          | Change field *values* only; never resize or change `dtype` without reinstantiating. |

---

**Follow these steps and you will obtain a one-to-one, dependency-injected Taichi class that:**

* Uses clean, descriptive `snake_case` names throughout.
* Re-uses your existing `fget_*` and `kget_*` code.
