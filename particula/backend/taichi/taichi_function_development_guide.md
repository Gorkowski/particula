# Taichi Function Development Guide

This short guide describes how to port an existing *Particula* NumPy module to
a Taichi-accelerated implementation.  

---

## 1.  File location
Duplicate the original module’s relative path under the `backend/taichi` tree and
prepend the filename with `ti_`.

Example  
```
particula/particles/properties/knudsen_number_module.py
→ particula/backend/taichi/particles/properties/ti_knudsen_number_module.py
```

### Name convention

Use similar or same named variables convention of the original module.
To keep the code clear, and consistent with the rest of the codebase.
Use full names for variables in functions, and avoid abbreviations.

## 2.  Required imports
```python
"""Short description of the module version."""
import taichi as ti
import numpy as np
from particula.backend import register
```

## 3.  Element-wise Taichi function
```python
@ti.func
def fget_<name>(arg1: ti.f64, arg2: ti.f64) -> ti.f64:
    """Short description of the taichi version."""
    # scalar expression identical to the original NumPy code
    return <expression>
```
All arguments **and** the return type must be `ti.f64`.

## 4.  Vectorized Taichi kernel
```python
@ti.kernel
def kget_<name>(
    arg1: ti.types.ndarray(dtype=ti.f64, ndim=1),
    arg2: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Short description of the taichi version."""
    for i in range(result.shape[0]):
        result[i] = fget_<name>(arg1[i], arg2[i])
```

## 5.  Public wrapper with backend registration
```python
@register("get_<name>", backend="taichi")
def ti_get_<name>(arg1, arg2):
    """Short description of the taichi version."""
    # 5 a – type guard
    if not (isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")

    # 5 b – ensure 1-D NumPy arrays
    a1, a2 = np.atleast_1d(arg1), np.atleast_1d(arg2)
    n = a1.size

    # 5 c – allocate Taichi NDArray buffers
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)

    # 5 d – launch the kernel
    kget_<name>(variable_a1_ti, variable_a2_ti, result_ti)
    return result_ti.to_numpy()
```

## 6.  Expose the function
If the corresponding `__init__.py` under `backend/taichi/...` does not yet
import your new function, add it there.

## 7.  Unit tests
Create `ti_<name>_module_test.py` inside the same sub-package as the original
tests. Include two tests:

1. Compare the Taichi wrapper with the reference NumPy implementation using  
   `np.testing.assert_allclose`.
2. Invoke the kernel directly and compare its result with the expected output.

Remember to initialise Taichi:
```python
import taichi as ti
ti.init(arch=ti.cpu)
```

## 9.  Data types
Use `float64` (`ti.f64`) consistently for numerical parity with NumPy.

## 10.  Coding style
Keep the API identical to the original NumPy function.  
Suffix the public Taichi wrapper with `_taichi`, and prefix new files with
`ti_`.

