"""
Given a python aerosol object and condensation strategy,
convert it to a Taichi particle resolved condensation object.
"""

from __future__ import annotations

# --- std / third-party -------------------------------------------------
from typing import Any, Tuple, Optional
import numpy as np

# --- particula (python backend) ---------------------------------------
import particula as par
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)

# --- particula (taichi backend) ---------------------------------------
from particula.backend.taichi.aerosol.ti_particle_resolved import (
    TiAerosolParticleResolved,
)


# ----------------------------------------------------------------------
# Helper utilities (pure python, no Taichi inside)
# ----------------------------------------------------------------------
def _get_attr(obj: Any, *names: str) -> Any:
    """
    Return the first existing attribute in *names* from *obj*.

    Raises
    ------
    AttributeError
        If none of the names are present.
    """
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(f"None of {names} exist in {type(obj).__name__}")


def _compute_pure_vp(gas, temperature: float) -> np.ndarray:
    """Return pure-component vapour pressure [Pa] for every species."""
    # try dedicated API first
    if hasattr(gas, "get_vapor_pressure"):
        return np.asarray(gas.get_vapor_pressure(temperature))
    # else call a strategy array if present (handles the builder used in tests)
    if hasattr(gas, "_vapor_pressure_strategy"):
        return np.asarray([s.vapor_pressure(temperature) for s in gas._vapor_pressure_strategy])  # type: ignore[attr-defined]
    # fallback: constant 0 Pa
    return np.zeros(len(_get_attr(gas, "molar_mass", "_molar_mass")))


# ----------------------------------------------------------------------
# Conversion functions
# ----------------------------------------------------------------------
def build_ti_particle_resolved(
    aerosol: "par.aerosol.Aerosol",
    cond_py: CondensationIsothermal,
    *,
    temperature: float = 298.15,
    pressure: float = 101_325.0,
    time_step: float = 1.0,
    simulation_volume: float = 1.0,
    variant_count: int = 1,
) -> TiAerosolParticleResolved:
    """
    Build and return a TiAerosolParticleResolved object populated from a
    python-backend *aerosol* together with a CondensationIsothermal instance.
    """
    # ---- unpack python objects --------------------------------------
    particle = _get_attr(
        aerosol, "particle", "particles", "particle_representation"
    )
    gas = _get_attr(aerosol, "atmosphere", "gas_species", "gas")

    # per-species data -------------------------------------------------
    species_masses_np: np.ndarray = np.asarray(
        _get_attr(particle, "mass", "_mass")
    )  # [n_particles, n_species]
    density_np: np.ndarray = np.asarray(
        _get_attr(particle, "density", "_density")
    )  # [n_species]
    molar_mass_np: np.ndarray = np.asarray(
        _get_attr(gas, "molar_mass", "_molar_mass")
    )  # [n_species]

    pure_vp_np = _compute_pure_vp(gas, temperature)  # [Pa]
    vapor_conc_np: np.ndarray = np.asarray(
        _get_attr(gas, "concentration", "_concentration")
    )  # [kg m⁻³]

    # activity (kappa) & surface tension ------------------------------
    kappa_np = np.zeros_like(molar_mass_np)
    if hasattr(particle, "activity") and hasattr(particle.activity, "kappa"):
        kappa_np = np.asarray(particle.activity.kappa)

    surface_tension_np = np.full_like(molar_mass_np, 0.072)
    if hasattr(particle, "surface") and hasattr(
        particle.surface, "surface_tension"
    ):
        surface_tension_np = np.asarray(particle.surface.surface_tension)

    # gas-phase mass (kg) ---------------------------------------------
    gas_mass_np = vapor_conc_np * simulation_volume

    # per-particle concentration (# m⁻³) – fall back to ones
    n_particles = species_masses_np.shape[0]
    particle_conc_np = np.ones(n_particles)
    if hasattr(particle, "concentration"):
        particle_conc_np = np.asarray(particle.concentration)

    # ---- create & load Taichi object --------------------------------
    sim = TiAerosolParticleResolved(
        particle_count=n_particles,
        species_count=species_masses_np.shape[1],
        variant_count=variant_count,
        time_step=time_step,
        simulation_volume=simulation_volume,
        temperature=temperature,
        pressure=pressure,
    )

    sim.setup(
        variant_index=0,
        species_masses_np=species_masses_np,
        density_np=density_np,
        molar_mass_np=molar_mass_np,
        pure_vapor_pressure_np=pure_vp_np,
        vapor_concentration_np=vapor_conc_np,
        kappa_value_np=kappa_np,
        surface_tension_np=surface_tension_np,
        gas_mass_np=gas_mass_np,
        particle_concentration_np=particle_conc_np,
    )
    return sim


def update_python_aerosol_from_ti(
    sim: TiAerosolParticleResolved,
    aerosol: "par.aerosol.Aerosol",
    variant_index: int = 0,
) -> None:
    """
    Copy species-mass (particles) and gas-mass data from *sim* back into the
    supplied python-backend *aerosol* object (in-place).
    """
    particle = _get_attr(
        aerosol, "particle", "particles", "particle_representation"
    )
    gas = _get_attr(aerosol, "atmosphere", "gas_species", "gas")

    # species-mass -----------------------------------------------------
    species_mass_new = sim.get_species_masses(variant_index)  # [n_p, n_s]
    if hasattr(particle, "mass"):
        particle.mass[:] = species_mass_new
    elif hasattr(particle, "_mass"):
        particle._mass[:] = species_mass_new
    else:
        raise AttributeError("Cannot locate mass array in python particle")

    # gas-mass → concentration ----------------------------------------
    gas_mass_new = sim.get_gas_mass(variant_index)  # [n_s]
    volume = float(sim.env.simulation_volume)
    concentration_new = gas_mass_new / volume
    if hasattr(gas, "concentration"):
        gas.concentration[:] = concentration_new
    elif hasattr(gas, "_concentration"):
        gas._concentration[:] = concentration_new
    else:
        raise AttributeError("Cannot locate concentration array in python gas")

