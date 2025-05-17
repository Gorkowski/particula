### **Particula → Taichi Class Conversion Guide**

*Turn any NumPy-based “model” class into a high-performance, data-oriented Taichi class while re-using the `fget_…` / `kget_…` helpers you have already written.*

The high-level goal is to convert a numpy-class completely to a Taichi class, keep the same method names and signatures, and use the taichi data-oriented programming paradigm. This will allow us to keep the code clean and easy to read, while also making it more performant.
This will be a breaking change and will not be interoperable with the NumPy version.

---

## 1 · File & naming conventions

| What               | How                                                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **Location**       | Mirror the original path under `particula/backend/taichi/…` and prepend the file name with `ti_`.                               |
| **Class name**     | Keep the original class name; the `@ti.data_oriented` decorator makes it Taichi-aware. If you must disambiguate, append **TI**. |
| **Helper imports** | Import any previously created kernels / functions (`fget_*`, `kget_*`) at module level so every method can call them.           |

Example

```
# original
particula/dynamics/condensation/<name>.py

# taichi version
particula/backend/taichi/dynamics/condensation/ti_<name>.py
```

### Name convention

Use similar or same named variables convention of the original module.
To keep the code clear, and consistent with the rest of the codebase.
Use full names for variables in functions, and avoid abbreviations.

---

## 2 · Required imports

```python
"""Taichi implementation of <name>."""
import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

#  ─── Example of previously written helpers ─────────────────────────────────────────
from particula.backend.taichi.dynamics.mass_transfer.ti_vapor import (
    fget_vapor_transition_correction,
)            # ← element-wise
from particula.backend.taichi.dynamics.mass_transfer.ti_mass_transfer import (
    kget_mass_transfer_rate,
)            # ← vectorised
```

---

## 3 · Example Skeleton of a Taichi data-oriented class

* Allocate all persistent data (`particle_radius`, `mass`, etc.) as `ti.field`s in `__init__`.
* Keep element-wise math in `@ti.func`s or call existing `fget_*` helpers.
* Combine all per-step math into one heavy `@ti.kernel`; loop with `ti.ndrange`.
* Wrap compile-time constants with `ti.static(...)`.
* Preserve the original API naming (e.g., `step(dt, …)`).
* Allocate scratch fields only when needed (via helper like `allocate_scratch`).
* Use `ti.f64` everywhere and `ti.init(default_fp=ti.f64)`.
* Convert NumPy → Taichi once at the boundary, then stay in Taichi.
* Add one-line docstrings to every kernel and helper.

Example conversion:

```python
@ti.data_oriented
class Ti<name>:
    """
    Short description of the module version.
    Mirrors particula.<folder>.<name>.
    """

    # allocate persistent particle / species fields
    def __init__(self, n_particles: int, n_species: int = 1):
        self.particle_radius = ti.field(dtype=ti.f64, shape=n_particles)               # r_i
        self.mass   = ti.field(dtype=ti.f64, shape=(n_particles, n_species))  # m_{i,s}
        self.conc   = ti.field(dtype=ti.f64, shape=n_particles)               # N_i
        self.Di     = ti.field(dtype=ti.f64, shape=n_species)                 # diffusion
        self.Mi     = ti.field(dtype=ti.f64, shape=n_species)                 # molar mass
        self.alpha  = 1.0                                                     # scalar

    # dynamic scratch buffers can be allocated later
    def allocate_scratch(self, n):
        self.tmp = ti.field(ti.f64, shape=n)

    # helper func – can call imported fget_… without rewriting the math
    @ti.func
    def _transition(self, Kn: ti.f64) -> ti.f64:
        return fget_vapor_transition_correction(Kn, ti.static(self.alpha))

    # main kernel – heavy maths only
    @ti.kernel
    def _advance(
        self, T: ti.f64, P: ti.f64,
        dt: ti.f64, mean_free_path: ti.f64,
    ):
        R = 8.314462618
        for p, s in ti.ndrange(self.particle_radius.shape[0], self.Mi.shape[0]):
            r   = ti.max(self.particle_radius[p], 1e-20)
            Kn  = mean_free_path / r
            fkn = self._transition(Kn)                 # uses imported helper

            K   = 4.0 * ti.math.pi * r * self.Di[s] * fkn
            # (Δp computed elsewhere – shown here for completeness)
            delta_p = 1.0
            dm = K * self.Mi[s] * delta_p / (R * T) * dt
            self.mass[p, s] += dm

    # public API
    def step(self, dt: float, *, T=298.15, P=101325.0, λ=66e-9):
        """Advance the system by *dt* seconds."""
        self._advance(T, P, dt, λ)
```

---

## 4 · Guidelines for method conversion

| If the original method…                                     | Then in Taichi…                                                                                              |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Is pure math** and works element-wise                     | Convert to `@ti.func` or reuse an existing `fget_*`.                                                         |
| **Loops over an array**                                     | Convert to one `@ti.kernel` and, if helpful, call a vectorised `kget_*`.                                     |
| **Operates on scratch arrays**                              | Allocate with `FieldsBuilder` (see docs) or via a normal field attribute in a dedicated `allocate_*` helper. |


---

## 6 · Backend registration

If external code instantiates condensation strategies through a factory, expose a **wrapper constructor** and register it:

```python
@register("<name>", backend="taichi")
def Ti<name>(...):
```

---

## 7 · Unit-testing checklist

1. **Numerical parity**

   ```python
   np.testing.assert_allclose(
       numpy_impl.step(dt, **kw).get_state(),
       ti_impl.step(dt, **kw).get_state(),
       rtol=1e-12,
   )
   ```
2. **Kernel invocation** – call `_advance` or other kernels directly on small dummy data and compare.
3. Initialise Taichi explicitly in each test file:

   ```python
   import taichi as ti;  ti.init(arch=ti.cpu, default_fp=ti.f64)
   ```

---

## 8 · Type & style rules

* Use **`ti.f64` (float64)** for every numerical field or argument.
* Preserve the naming of the original class (method names, signatures).
* Describe every kernel/function with a one-line docstring *what it does*, not how.
* Prefer **full variable names** (`particle_radius`, `knudsen_number`) over abbreviations.
* Do not edit the original python class.

---

### **Cheat-sheet**

| Task                        | Syntax                                     |
| --------------------------- | ------------------------------------------ |
| Declare class               | `@ti.data_oriented`                        |
| Persistent field            | `self.x = ti.field(dtype=ti.f64, shape=n)` |
| Scratch field later         | `self.tmp = ti.field(ti.f64, shape=n)`     |
| Element-wise helper         | `@ti.func` (or import `fget_*`)            |
| Heavy loop                  | `@ti.kernel`                               |
| Static utility inside class | `@staticmethod + @ti.func`                 |
| Class-wide kernel           | `@classmethod + @ti.kernel`                |

Follow these steps and you will have a **one-to-one Taichi drop-in** for every performance-critical class, with minimal code duplication and full reuse of your existing Taichi math helpers.
