"""Benchmark CondensationIsothermal.step for an ever-increasing number of
particles (10 species fixed)."""

import os
import json
import numpy as np
import particula as par  # << NEW – pure-python helpers
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)

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
from particula.backend.taichi.dynamics.condensation.ti_condensation_strategies import (
    TiCondensationIsothermal,
)
from particula.backend.taichi.aerosol.ti_particle_resolved import (
    TiAerosolParticleResolved,
)

# python (NumPy-only) condensation
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal as PyCondensationIsothermal,
)


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
    distribution = (
        np.abs(rng.standard_normal((n_particles, n_species))) * 1e-18
    )
    densities = np.linspace(1_000.0, 1_500.0, n_species)
    concentration = np.ones(n_particles)
    charge = np.zeros(n_particles)

    strategy = TiParticleResolvedSpeciatedMass()
    activity = ActivityKappaParameter(
        kappa=np.zeros(n_species),
        density=densities,
        molar_mass=np.linspace(
            0.018, 0.018 + 0.002 * (n_species - 1), n_species
        ),
        water_index=0,
    )
    surface = TiSurfaceStrategyMolar(
        surface_tension=np.full(n_species, 0.072),
        density=densities,
        molar_mass=np.linspace(
            0.018, 0.018 + 0.002 * (n_species - 1), n_species
        ),
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
        molar_mass=np.linspace(
            0.018, 0.018 + 0.002 * (n_species - 1), n_species
        ),
        vapor_pressure_strategy=[
            (
                WaterBuckStrategy()
                if i == 0
                else ConstantVaporPressureStrategy(100.0 + i * 50.0)
            )
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


# ── fused particle-resolved aerosol (new benchmark) ────────────────────
def _build_ti_particle_resolved(n_particles: int, n_species: int = 10):
    """Create and populate a TiAerosolParticleResolved instance."""
    rng = np.random.default_rng(0)

    species_masses = np.abs(rng.standard_normal((n_particles, n_species))) * 1e-18
    density = np.linspace(1_000.0, 1_500.0, n_species)
    molar_mass = np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species)
    pure_vp = np.full(n_species, 50.0)              # dummy, Pa
    vapor_conc = np.ones(n_species) * 1.0e-3        # mol m⁻3
    kappa = np.zeros(n_species)
    surface_tension = np.full(n_species, 0.072)     # N m⁻1
    gas_mass = np.ones(n_species) * 1.0e-6          # kg
    particle_conc = np.ones(n_particles)            # # m⁻3

    sim = TiAerosolParticleResolved(
        particle_count=n_particles,
        species_count=n_species,
        time_step=1.0,
        simulation_volume=1.0,
    )
    sim.setup(
        species_masses,
        density,
        molar_mass,
        pure_vp,
        vapor_conc,
        kappa,
        surface_tension,
        gas_mass,
        particle_conc,
    )
    return sim


def make_fused_step_callable(sim_obj: TiAerosolParticleResolved):
    """Return a callable that executes sim_obj.fused_step once."""
    def _inner():
        sim_obj.fused_step()
    return _inner

def _build_particle_and_gas_python(n_particles: int, n_species: int = 10):
    """
    Pure-Python analogue of _build_particle_and_gas for the NumPy backend.
    Creates particle and gas objects compatible with the python
    CondensationIsothermal.step signature.
    """
    rng = np.random.default_rng(0)
    # particle mass [kg] per species; positive values only
    mass = np.abs(rng.standard_normal((n_particles, n_species))) * 1.0e-18

    densities = np.linspace(1_000.0, 1_500.0, n_species)
    molar_mass = np.linspace(0.018, 0.018 + 0.002 * (n_species - 1), n_species)

    # ---------- particle ----------
    activity = (
        par.particles.ActivityKappaParameterBuilder()
        .set_density(densities, "kg/m^3")
        .set_kappa(np.zeros(n_species))
        .set_molar_mass(molar_mass, "kg/mol")
        .set_water_index(0)
        .build()
    )
    surface = (
        par.particles.SurfaceStrategyVolumeBuilder()
        .set_density(densities, "kg/m^3")
        .set_surface_tension(np.full(n_species, 0.072), "N/m")
        .build()
    )
    particle = (
        par.particles.ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(
            par.particles.ParticleResolvedSpeciatedMass()
        )
        .set_activity_strategy(activity)
        .set_surface_strategy(surface)
        .set_mass(mass, "kg")
        .set_density(densities, "kg/m^3")
        .set_charge(0)
        .set_volume(1.0, "m^3")  # arbitrary parcel volume
        .build()
    )

    # ---------- gas ----------
    vp_strategies = [
        par.gas.VaporPressureFactory().get_strategy(
            "water_buck" if i == 0 else "constant",
            (
                None
                if i == 0
                else {
                    "vapor_pressure": 100.0 + i * 50.0,
                    "vapor_pressure_units": "Pa",
                }
            ),
        )
        for i in range(n_species)
    ]
    gas_species = (
        par.gas.GasSpeciesBuilder()
        .set_name([f"X{i}" for i in range(n_species)])
        .set_molar_mass(molar_mass, "kg/mol")
        .set_vapor_pressure_strategy(vp_strategies)
        .set_concentration(np.ones(n_species), "kg/m^3")
        .set_partitioning(True)
        .build()
    )

    return particle, gas_species


def make_python_step_callable(particle, gas, cond):
    """Return a callable that executes cond.step once."""

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
    PARTICLE_COUNTS = np.logspace(
        1, 6, num=10, dtype=np.int64
    )  # 10^1 to 10^5 particles

    # build a single condensation object (species count is fixed)
    molar_mass_vec = np.linspace(
        0.018, 0.018 + 0.002 * (N_SPECIES - 1), N_SPECIES
    )
    condensation_ti = _build_taichi_condensation_isothermal(molar_mass_vec)

    # build a single Python-side CondensationIsothermal object
    condensation_py = PyCondensationIsothermal(
        molar_mass=molar_mass_vec,
        diffusion_coefficient=2.0e-5,
        accommodation_coefficient=1.0,
    )

    rows = []
    csv_header = None  # will be created in first loop

    for n_particles in PARTICLE_COUNTS:
        # ----- Taichi objects & stats ---------------------------------------
        # particle, gas = _build_particle_and_gas(n_particles, N_SPECIES)
        # stats_ti = get_function_benchmark(
        #     make_step_callable(particle, gas, condensation_ti),
        #     ops_per_call=1,
        #     max_run_time_s=3.0,
        # )

        # ----- Python objects & stats ---------------------------------------
        py_particle, py_gas = _build_particle_and_gas_python(
            n_particles, N_SPECIES
        )
        stats_py = get_function_benchmark(
            make_python_step_callable(py_particle, py_gas, condensation_py),
            ops_per_call=1,
        )

        # ----- Fused particle-resolved solver stats ------------------------
        pr_sim = _build_ti_particle_resolved(n_particles, N_SPECIES)
        stats_pr = get_function_benchmark(
            make_fused_step_callable(pr_sim),
            ops_per_call=1,
            max_run_time_s=3.0,
        )

        # ----- build header only once ---------------------------------------
        if csv_header is None:
            # taichi_headers = ["taichi_" + h for h in stats_ti["array_headers"]]
            python_headers = ["python_" + h for h in stats_py["array_headers"]]
            fused_headers  = ["fused_"  + h for h in stats_pr["array_headers"]]
            csv_header = ["array_length", *python_headers, *fused_headers]

        # ----- collect row ---------------------------------------------------
        rows.append(
            [
                n_particles,
                *stats_py["array_stats"],
                *stats_pr["array_stats"],
            ]
        )

    # ── output directory ───────────────────────────────────────────────────
    output_directory = os.path.join(
        os.path.dirname(__file__), "benchmark_outputs"
    )
    os.makedirs(output_directory, exist_ok=True)

    csv_file = os.path.join(
        output_directory, "condensation_isothermal_step_benchmark.csv"
    )
    save_combined_csv(csv_file, csv_header, rows)

    plot_throughput_vs_array_length(
        csv_header,
        rows,
        "TiCondensation.step throughput vs #particles (10 species)",
        os.path.join(
            output_directory, "condensation_isothermal_step_benchmark.png"
        ),
    )

    # also stash the machine / environment info
    with open(
        os.path.join(output_directory, "system_info.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(get_system_info(), fh, indent=2)

    print(f"Benchmark results saved to {csv_file}")
    print(stats_pr['report'])
