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
| **Path**                       | Mirror the original module path under `particula/backend/taichi/…` and prepend `ti_` to the filename.<br>`particula/dynamics/condensation/condensation.py` → `particula/backend/taichi/dynamics/condensation/ti_condensation.py` |
| **Class name**                 | Keep the original class name. Add **TI** only if the original name is still imported elsewhere. Decorate with `@ti.data_oriented`.                                                                                               |
| **Variable / attribute names** | *Every* identifier must be a full English word or phrase in **`snake_case`**. Loop counters (`i`, `j`, `k`) are the only permitted one-letter names.                                                                             |
| **Helper imports**             | At module level, import any `fget_*` / `kget_*` helpers you have already written so the Taichi class can call them.                                                                                                              |

---

## 2 · Required Imports (template)

```python
"""Taichi implementation of particula.dynamics.condensation.Condensation."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

# ─── Existing helpers you want to re-use ────────────────────────────────
from particula.backend.taichi.dynamics.mass_transfer.ti_vapor import (
    fget_vapor_transition_correction,
)
from particula.backend.taichi.dynamics.mass_transfer.ti_mass_transfer import (
    kget_mass_transfer_rate,
)

ti.init(default_fp=ti.f64)    # always work in float64 unless you *must* down-cast
```

---

## 3 · Skeleton of a Fully Injected Taichi Class

```python
@ti.data_oriented
class TiCondensation:
    """
    Taichi drop-in replacement for particula.dynamics.condensation.Condensation.
    """

    # ── constructor ─────────────────────────────────────────────────────┐
    def __init__(
        self,
        particle_system,                     # TiParticleSystem instance
        gas_species,                         # TiGasSpecies instance
        *,
        accommodation_coefficient: float = 1.0,
    ):
        # store collaborators so kernels can access them
        self.particle_system = particle_system
        self.gas_species = gas_species

        # scalar parameters – zero-dimensional fields
        self.accommodation_coefficient = ti.field(dtype=ti.f64, shape=())
        self.accommodation_coefficient[None] = accommodation_coefficient

        # allocate any Condensation-specific persistent fields
        number_of_particles = self.particle_system.number_of_particles
        self.mass_change = ti.field(dtype=ti.f64, shape=number_of_particles)
    # ────────────────────────────────────────────────────────────────────┘

    # ── element-wise helpers (call existing fget_*) ─────────────────────
    @ti.func
    def transition_correction_factor(self, knudsen_number: ti.f64) -> ti.f64:
        return fget_vapor_transition_correction(
            knudsen_number,
            self.accommodation_coefficient[None],
        )

    # ── thin wrapper around an existing kget_* kernel ───────────────────
    @ti.func
    def mass_transfer_rate(
        self,
        radius_field: ti.template(),
        diffusivity_field: ti.template(),
        output_field: ti.template(),
    ):
        kget_mass_transfer_rate(radius_field, diffusivity_field, output_field)

    # ── main kernel: one loop, all math ─────────────────────────────────
    @ti.kernel
    def _advance_one_timestep(
        self,
        time_step: ti.f64,
        temperature: ti.f64,
        pressure: ti.f64,
        mean_free_path: ti.f64,
    ):
        """
        Compute condensation mass fluxes for every particle and species.
        """
        gas_constant = 8.314462618

        for particle_index, species_index in ti.ndrange(
            self.particle_system.number_of_particles,
            self.gas_species.number_of_species,
        ):
            radius = ti.max(self.particle_system.radius[particle_index], 1.0e-20)
            knudsen_number = mean_free_path / radius

            correction_factor = self.transition_correction_factor(knudsen_number)

            diffusivity = self.gas_species.diffusivity[species_index]
            mass_transfer_coefficient = (
                4.0 * ti.math.pi * radius * diffusivity * correction_factor
            )

            gas_partial_pressure = self.gas_species.partial_pressure(temperature)
            particle_partial_pressure = self.particle_system.partial_pressure(
                particle_index,
                species_index,
                temperature,
            )

            delta_partial_pressure = (
                gas_partial_pressure - particle_partial_pressure
            )

            delta_mass = (
                mass_transfer_coefficient
                * self.gas_species.molar_mass[species_index]
                * delta_partial_pressure
                / (gas_constant * temperature)
                * time_step
            )

            # update fields
            self.mass_change[particle_index] += delta_mass
            self.particle_system.mass[particle_index, species_index] += delta_mass

    # ── public API (mirrors NumPy version) ──────────────────────────────
    def advance(
        self,
        dt: float,
        *,
        temperature: float = 298.15,
        pressure: float = 101_325.0,
        mean_free_path: float = 66e-9,
    ):
        """Advance condensation by *dt* seconds."""
        self._advance_one_timestep(dt, temperature, pressure, mean_free_path)
```

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
| ABCs to mixins                                     | Taichi classes cannot inherit from other classes                                   | Use mixins to share code between classes.                   |


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
| Allocate scratch field later | `self.mass_rate_buffer = ti.field(ti.f64, shape=n_particles)`                       |
| Avoid recompilation          | Change field *values* only; never resize or change `dtype` without reinstantiating. |

---

**Follow these steps and you will obtain a one-to-one, dependency-injected Taichi class that:**

* Matches the original NumPy interface.
* Uses clean, descriptive `snake_case` names throughout.
* Re-uses your existing `fget_*` and `kget_*` code.
* Can be swapped in simply by requesting the `"taichi"` backend.
