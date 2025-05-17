import taichi as ti
import numpy as np
from particula.backend.dispatch_register import register

@ti.func
def fget_surface_partial_pressure(pvp: ti.f64, act: ti.f64) -> ti.f64:
    return pvp * act

@ti.func
def fget_ideal_activity_mass(mass_single: ti.f64, total_mass: ti.f64) -> ti.f64:
    return 0.0 if total_mass == 0.0 else mass_single / total_mass

@ti.func
def fget_ideal_activity_volume(mass_single: ti.f64,
                               dens_single: ti.f64,
                               total_volume: ti.f64) -> ti.f64:
    return 0.0 if total_volume == 0.0 else (mass_single / dens_single) / total_volume

@ti.func
def fget_ideal_activity_molar(mass_single: ti.f64,
                              mm_single:   ti.f64,
                              total_moles: ti.f64) -> ti.f64:
    return 0.0 if total_moles == 0.0 else (mass_single / mm_single) / total_moles

# 1-D ---------------------------------------------------------------
@ti.kernel
def kget_surface_partial_pressure(
    pvp: ti.types.ndarray(dtype=ti.f64, ndim=1),
    act: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=1),
):
    for i in range(res.shape[0]):
        res[i] = fget_surface_partial_pressure(pvp[i], act[i])

# 2-D rows = particles / mixtures ; cols = species  -----------------
@ti.kernel
def kget_ideal_activity_mass(
    mc: ti.types.ndarray(dtype=ti.f64, ndim=2),
    res: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    for i in range(mc.shape[0]):
        row_sum = 0.0
        for s in range(mc.shape[1]):
            row_sum += mc[i, s]
        for s in range(mc.shape[1]):
            res[i, s] = fget_ideal_activity_mass(mc[i, s], row_sum)

@ti.kernel
def kget_ideal_activity_volume(
    mc: ti.types.ndarray(dtype=ti.f64, ndim=2),
    dens: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    ns = mc.shape[1]
    for i in range(mc.shape[0]):
        vol_sum = 0.0
        for s in range(ns):
            vol_sum += mc[i, s] / dens[s]
        for s in range(ns):
            res[i, s] = fget_ideal_activity_volume(mc[i, s], dens[s], vol_sum)

@ti.kernel
def kget_ideal_activity_molar(
    mc: ti.types.ndarray(dtype=ti.f64, ndim=2),
    mm: ti.types.ndarray(dtype=ti.f64, ndim=1),
    res: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    ns = mc.shape[1]
    for i in range(mc.shape[0]):
        mol_sum = 0.0
        for s in range(ns):
            mol_sum += mc[i, s] / mm[s]
        for s in range(ns):
            res[i, s] = fget_ideal_activity_molar(mc[i, s], mm[s], mol_sum)

@ti.kernel
def kget_kappa_activity(
    mc: ti.types.ndarray(dtype=ti.f64, ndim=2),
    kap: ti.types.ndarray(dtype=ti.f64, ndim=1),
    dens: ti.types.ndarray(dtype=ti.f64, ndim=1),
    mm: ti.types.ndarray(dtype=ti.f64, ndim=1),
    water_idx: ti.i32,
    res: ti.types.ndarray(dtype=ti.f64, ndim=2),
):
    ns = mc.shape[1]
    for i in range(mc.shape[0]):

        # mole-fraction part (all species first)
        mol_sum = 0.0
        for s in range(ns):
            mol_sum += mc[i, s] / mm[s]
        for s in range(ns):
            mol = mc[i, s] / mm[s]
            res[i, s] = 0.0 if mol_sum == 0.0 else mol / mol_sum

        # volume fractions ------------------------------------------
        vol_sum = 0.0
        for s in range(ns):
            vol_sum += mc[i, s] / dens[s]

        water_vf = 0.0
        if vol_sum > 0.0:
            water_vf = (mc[i, water_idx] / dens[water_idx]) / vol_sum

        sol_vol_sum = 1.0 - water_vf
        kappa_w = 0.0
        if sol_vol_sum > 0.0:
            # single-solute shortcut
            if ns == 2:
                sid = 1 if water_idx == 0 else 0
                kappa_w = kap[sid]
            else:
                for s in range(ns):
                    if s != water_idx:
                        vf_s = (mc[i, s] / dens[s]) / vol_sum
                        kappa_w += (vf_s / sol_vol_sum) * kap[s]

        vol_term = 0.0
        if water_vf > 0.0:
            vol_term = kappa_w * sol_vol_sum / water_vf
        water_activity = 0.0 if water_vf == 0.0 else 1.0 / (1.0 + vol_term)

        res[i, water_idx] = water_activity

@register("get_surface_partial_pressure", backend="taichi")
def ti_get_surface_partial_pressure(pure_vapor_pressure, activity):
    if isinstance(pure_vapor_pressure, float):
        return pure_vapor_pressure * activity  # scalar shortcut

    pv, act = np.atleast_1d(pure_vapor_pressure), np.atleast_1d(activity)
    n = pv.size
    pv_ti  = ti.ndarray(dtype=ti.f64, shape=n)
    act_ti = ti.ndarray(dtype=ti.f64, shape=n)
    res_ti = ti.ndarray(dtype=ti.f64, shape=n)
    pv_ti.from_numpy(pv)
    act_ti.from_numpy(act)
    kget_surface_partial_pressure(pv_ti, act_ti, res_ti)
    out = res_ti.to_numpy()
    return out.item() if out.size == 1 else out

@register("get_ideal_activity_mass", backend="taichi")
def ti_get_ideal_activity_mass(mass_concentration):
    if isinstance(mass_concentration, float):
        return 1.0
    mc = np.atleast_2d(mass_concentration)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mc_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    mc_ti.from_numpy(mc)
    kget_ideal_activity_mass(mc_ti, res_ti)
    out = res_ti.to_numpy()
    if single_row:
        return out[0]
    return out

@register("get_ideal_activity_volume", backend="taichi")
def ti_get_ideal_activity_volume(mass_concentration, density):
    if isinstance(mass_concentration, float):
        return 1.0
    mc = np.atleast_2d(mass_concentration)
    dens = np.atleast_1d(density)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mc_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    dens_ti = ti.ndarray(dtype=ti.f64, shape=dens.shape)
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    mc_ti.from_numpy(mc)
    dens_ti.from_numpy(dens)
    kget_ideal_activity_volume(mc_ti, dens_ti, res_ti)
    out = res_ti.to_numpy()
    if single_row:
        return out[0]
    return out

@register("get_ideal_activity_molar", backend="taichi")
def ti_get_ideal_activity_molar(mass_concentration, molar_mass):
    if isinstance(mass_concentration, float):
        return 1.0
    mc = np.atleast_2d(mass_concentration)
    mm = np.atleast_1d(molar_mass)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mc_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    mm_ti = ti.ndarray(dtype=ti.f64, shape=mm.shape)
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    mc_ti.from_numpy(mc)
    mm_ti.from_numpy(mm)
    kget_ideal_activity_molar(mc_ti, mm_ti, res_ti)
    out = res_ti.to_numpy()
    if single_row:
        return out[0]
    return out

@register("get_kappa_activity", backend="taichi")
def ti_get_kappa_activity(mass_concentration, kappa, density, molar_mass, water_index):
    if isinstance(mass_concentration, float):
        return 1.0
    mc = np.atleast_2d(mass_concentration)
    kap = np.atleast_1d(kappa)
    dens = np.atleast_1d(density)
    mm = np.atleast_1d(molar_mass)
    single_row = False
    if mass_concentration is not None and np.ndim(mass_concentration) == 1:
        single_row = True
    mc_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    kap_ti = ti.ndarray(dtype=ti.f64, shape=kap.shape)
    dens_ti = ti.ndarray(dtype=ti.f64, shape=dens.shape)
    mm_ti = ti.ndarray(dtype=ti.f64, shape=mm.shape)
    res_ti = ti.ndarray(dtype=ti.f64, shape=mc.shape)
    mc_ti.from_numpy(mc)
    kap_ti.from_numpy(kap)
    dens_ti.from_numpy(dens)
    mm_ti.from_numpy(mm)
    kget_kappa_activity(mc_ti, kap_ti, dens_ti, mm_ti, int(water_index), res_ti)
    out = res_ti.to_numpy()
    if single_row:
        return out[0]
    return out
