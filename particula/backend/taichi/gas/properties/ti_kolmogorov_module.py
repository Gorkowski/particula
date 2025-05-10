"""Taichi-accelerated Kolmogorov scales for gas properties."""

import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_kolmogorov_time(kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64) -> ti.f64:
    """Elementwise Kolmogorov time: sqrt(v / eps)"""
    return ti.sqrt(kinematic_viscosity / turbulent_dissipation)

@ti.func
def fget_kolmogorov_length(kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64) -> ti.f64:
    """Elementwise Kolmogorov length: sqrt(sqrt(v^3 / eps))"""
    return ti.sqrt(ti.sqrt(kinematic_viscosity * kinematic_viscosity * kinematic_viscosity / turbulent_dissipation))

@ti.func
def fget_kolmogorov_velocity(kinematic_viscosity: ti.f64, turbulent_dissipation: ti.f64) -> ti.f64:
    """Elementwise Kolmogorov velocity: sqrt(sqrt(v * eps))"""
    return ti.sqrt(ti.sqrt(kinematic_viscosity * turbulent_dissipation))

@ti.kernel
def kget_kolmogorov_time(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Kolmogorov time kernel."""
    for i in range(result.shape[0]):
        result[i] = fget_kolmogorov_time(kinematic_viscosity[i], turbulent_dissipation[i])

@ti.kernel
def kget_kolmogorov_length(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Kolmogorov length kernel."""
    for i in range(result.shape[0]):
        result[i] = fget_kolmogorov_length(kinematic_viscosity[i], turbulent_dissipation[i])

@ti.kernel
def kget_kolmogorov_velocity(
    kinematic_viscosity: ti.types.ndarray(dtype=ti.f64, ndim=1),
    turbulent_dissipation: ti.types.ndarray(dtype=ti.f64, ndim=1),
    result: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    """Vectorized Kolmogorov velocity kernel."""
    for i in range(result.shape[0]):
        result[i] = fget_kolmogorov_velocity(kinematic_viscosity[i], turbulent_dissipation[i])

@register("get_kolmogorov_time", backend="taichi")
def ti_get_kolmogorov_time(kinematic_viscosity, turbulent_dissipation):
    """Taichi wrapper for Kolmogorov time."""
    if not (isinstance(kinematic_viscosity, np.ndarray) and isinstance(turbulent_dissipation, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    a1, a2 = np.atleast_1d(kinematic_viscosity), np.atleast_1d(turbulent_dissipation)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    kget_kolmogorov_time(variable_a1_ti, variable_a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_kolmogorov_length", backend="taichi")
def ti_get_kolmogorov_length(kinematic_viscosity, turbulent_dissipation):
    """Taichi wrapper for Kolmogorov length."""
    if not (isinstance(kinematic_viscosity, np.ndarray) and isinstance(turbulent_dissipation, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    a1, a2 = np.atleast_1d(kinematic_viscosity), np.atleast_1d(turbulent_dissipation)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    kget_kolmogorov_length(variable_a1_ti, variable_a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np

@register("get_kolmogorov_velocity", backend="taichi")
def ti_get_kolmogorov_velocity(kinematic_viscosity, turbulent_dissipation):
    """Taichi wrapper for Kolmogorov velocity."""
    if not (isinstance(kinematic_viscosity, np.ndarray) and isinstance(turbulent_dissipation, np.ndarray)):
        raise TypeError("Taichi backend expects NumPy arrays for both inputs.")
    a1, a2 = np.atleast_1d(kinematic_viscosity), np.atleast_1d(turbulent_dissipation)
    n = a1.size
    variable_a1_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a2_ti = ti.ndarray(dtype=ti.f64, shape=n)
    result_ti = ti.ndarray(dtype=ti.f64, shape=n)
    variable_a1_ti.from_numpy(a1)
    variable_a2_ti.from_numpy(a2)
    kget_kolmogorov_velocity(variable_a1_ti, variable_a2_ti, result_ti)
    result_np = result_ti.to_numpy()
    return result_np.item() if result_np.size == 1 else result_np
