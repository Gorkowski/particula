"""Benchmark CondensationIsothermal.step for an ever-increasing number of
particles (10 species fixed)."""
import os
import json
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64)

from particula.backend.benchmark import (
    get_function_benchmark,
    get_system_info,
    save_combined_csv,
    plot_throughput_vs_array_length,
)

from particula.backend.taichi.particles.ti_distribution_strategies import (
    TiParticleResolvedSpeciatedMass,
)
from particula.backend.taichi.particles.ti_activity_strategies import (
    ActivityKappaParameter,
)
from particula.backend.taichi.particles.ti_surface_strategies import (
    TiSurfaceStrategyMolar,
)
from particula.backend.taichi.particles.ti_representation import (
    TiParticleRepresentation,
)
from particula.backend.taichi.gas.ti_species import TiGasSpecies
from particula.backend.taichi.gas.ti_vapor_pressure_strategies import (
    WaterBuckStrategy,
    ConstantVaporPressureStrategy,
)
from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies \
    import TiCondensationIsothermal

def _build_taichi_condensation_isothermal(
    molar_mass: np.ndarray,
    diffusion_coefficient: float = 2.0e-5,
    accommodation: float = 1.0,
):
    mm_ti = ti.ndarray(dtype=ti.f64, shape=molar_mass.shape)
    mm_ti.from_numpy(molar_mass)

    diff_coeff_ti = ti.field(ti.f64, shape=())
    diff_coeff_ti[None] = diffusion_coefficient

    alpha_ti = ti.ndarray(dtype=ti.f64, shape=molar_mass.shape)
    alpha_ti.from_numpy(np.full_like(molar_mass, accommodation))

    cond = TiCondensationIsothermal(
        molar_mass=mm_ti,
        diffusion_coefficient=diff_coeff_ti,
        accommodation_coefficient=alpha_ti,
        update_gases=False,
    )
    cond.molar_mass = mm_ti
    cond.diffusion_coefficient = diff_coeff_ti
    cond.accommodation_coefficient = alpha_ti
    return cond

def _build_particle_and_gas(n_particles: int, n_species: int = 10):
    # --- particle block --------------------------------------------------
    rng = np.random.default_rng(0)
    distribution = np.abs(rng.standard_normal((n_particles, n_species))) * 1e-18
    densities = np.linspace(1_000.0, 1_500.0, n_species)
    concentration = np.ones(n_particles)
    charge = np.zeros(n_particles)

    strategy = TiParticleResolvedSpeciatedMass()
    activity = ActivityKappaParameter(
        kappa=np.zeros(n_species),
        density=densities,
        molar_mass=np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species),
        water_index=0,
    )
    surface = TiSurfaceStrategyMolar(
        surface_tension=np.full(n_species, 0.072),
        density=densities,
        molar_mass=np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species),
    )
    particle = TiParticleRepresentation(
        strategy,
        activity,
        surface,
        distribution,
        densities,
        concentration,
        charge,
        volume=1.0,
    )

    # --- gas block -------------------------------------------------------
    gas = TiGasSpecies(
        name=np.array([f"X{i}" for i in range(n_species)]),
        molar_mass=np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species),
        vapor_pressure_strategy=[
            WaterBuckStrategy() if i == 0 else ConstantVaporPressureStrategy(100.0 + i * 50.0)
            for i in range(n_species)
        ],
        partitioning=True,
        concentration=np.ones(n_species),
    )
    return particle, gas

def make_step_callable(particle, gas, cond):
    def _inner():
        cond.step(
            particle=particle,
            gas_species=gas,
            temperature=298.15,
            pressure=101_325.0,
            time_step=1.0,
        )
    return _inner

if __name__ == "__main__":
    N_SPECIES = 10
    PARTICLE_COUNTS = [10, 100, 1_000, 10_000, 50_000]  # adjust/extend as desired

    # build a single condensation object (species count is fixed)
    molar_mass_vec = np.linspace(0.018, 0.018 + 0.002 * (N_SPECIES - 1), N_SPECIES)
    condensation_ti = _build_taichi_condensation_isothermal(molar_mass_vec)

    csv_header = ["array_length",
                  "ti_step_mean_s", "ti_step_std_s",
                  "ti_step_throughput_ops", "ti_step_gmean_ops",
                  "ti_step_gstd_ops"]
    rows = []

    for n_particles in PARTICLE_COUNTS:
        particle, gas = _build_particle_and_gas(n_particles, N_SPECIES)
        stats_ti = get_function_benchmark(
            make_step_callable(particle, gas, condensation_ti),
            ops_per_call=1,
        )
        rows.append([
            n_particles,
            *stats_ti["array_stats"],
        ])

    out_dir = os.path.dirname(__file__)
    csv_file = os.path.join(out_dir, "condensation_isothermal_step_benchmark.csv")
    save_combined_csv(csv_file, csv_header, rows)

    plot_throughput_vs_array_length(
        csv_header,
        rows,
        "TiCondensation.step throughput vs #particles (10 species)",
        os.path.join(out_dir, "condensation_isothermal_step_benchmark.png"),
    )

    # also stash the machine / environment info
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(get_system_info(), fh, indent=2)
