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
from particula.backend.taichi.aerosol.ti_environmental_conditions_builder import (
    EnvironmentalConditions,
)


# ----------------------------------------------------------------------
# Conversion functions
# ----------------------------------------------------------------------
def build_ti_particle_resolved(
    aerosol: "par.Aerosol",
    cond_py: CondensationIsothermal,
    time_step: float = 1.0,
    variant_count: int = 1,
) -> TiAerosolParticleResolved:
    """
    Build and return a TiAerosolParticleResolved object populated from a
    python-backend *aerosol* together with a CondensationIsothermal instance.
    """
    temperature = aerosol.atmosphere.temperature

    # per-species data -------------------------------------------------
    species_masses_np = aerosol.particles.distribution
    density_np = aerosol.particles.density
    molar_mass_np = cond_py.molar_mass
    particle_conc_np = aerosol.particles.concentration

    pure_vapor_pressure_np = aerosol.atmosphere.partitioning_species.get_pure_vapor_pressure(
        temperature=temperature
    )
    vapor_conc_np = aerosol.atmosphere.partitioning_species.get_concentration()

    # activity (kappa) & surface tension ------------------------------
    kappa_np = aerosol.particles.activity.kappa
    surface_tension_np = aerosol.particles.surface.surface_tension

    # single values
    pressure = aerosol.atmosphere.total_pressure
    simulation_volume = aerosol.particles.volume

    env = EnvironmentalConditions(
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        mass_accommodation=cond_py.accommodation_coefficient,
        diffusion_coefficient=cond_py.diffusion_coefficient,
        simulation_volume=simulation_volume,
        dynamic_viscosity=par.gas.get_dynamic_viscosity(
            temperature=temperature,
        )
    )

    # ---- create & load Taichi object --------------------------------
    sim = TiAerosolParticleResolved(
        particle_count=species_masses_np.shape[0],
        species_count=species_masses_np.shape[1],
        variant_count=variant_count,
        environmental_conditions=env,
    )

    sim.setup(
        variant_index=0,
        species_masses_np=species_masses_np,
        density_np=density_np,
        molar_mass_np=molar_mass_np,
        pure_vapor_pressure_np=pure_vapor_pressure_np,
        vapor_concentration_np=vapor_conc_np,
        kappa_value_np=kappa_np,
        surface_tension_np=surface_tension_np,
        gas_mass_np=vapor_conc_np,
        particle_concentration_np=particle_conc_np,
    )
    return sim


def update_python_aerosol_from_ti(
    sim: TiAerosolParticleResolved,
    aerosol: "par.Aerosol",
    variant_index: int = 0,
) -> None:
    """
    Copy species-mass (particles) and gas-mass data from *sim* back into the
    supplied python-backend *aerosol* object (in-place).
    """
    # species-mass -----------------------------------------------------
    species_mass_new = sim.get_species_masses(variant_index)  # [n_p, n_s]
    aerosol.particles.distribution = species_mass_new
    
    # gas-mass → concentration ----------------------------------------
    gas_mass_new = sim.get_gas_mass(variant_index)  # [n_s]
    aerosol.atmosphere.partitioning_species.set_concentration(
        new_concentration=gas_mass_new,
    )
