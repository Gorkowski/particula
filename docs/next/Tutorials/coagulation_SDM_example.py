# %% particle resolved coagulation example
from itertools import combinations
import numpy as np  # For numerical operations and array manipulations
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import pandas as pd  # For data manipulation and analysis
from scipy.integrate import (
    trapezoid,
)  # For numerical integration using the trapezoidal rule
from scipy.interpolate import RectBivariateSpline  # type: ignore

# Import specific modules from the particula package
from particula.next.dynamics.coagulation import brownian_kernel, rate
from particula.next.dynamics.coagulation import super_droplet_method

# The 'brownian_kernel' module calculates the Brownian coagulation kernel,
# which is used to determine coagulation rates between particles.
# The 'rate' module includes functions for calculating gain and loss rates during coagulation.
from particula.next.particles.properties.lognormal_size_distribution import (
    lognormal_pmf_distribution,
    lognormal_pdf_distribution,
    lognormal_sample_distribution,
)

# The 'lognormal_pmf_distribution' function generates a lognormal distribution
# represented as a Probability Mass Function (PMF).
# The 'lognormal_pdf_distribution' function generates a lognormal distribution
# represented as a Probability Density Function (PDF).
from particula.util.convert import distribution_convert_pdf_pms

# %% sample lognormal distribution

radius_bins = np.logspace(
    -9, -6, num=20
)  # Define the radius bins for the distribution
mass_bins = (
    4 / 3 * np.pi * radius_bins**3 * 1000
)  # Calculate the mass of the particles in the bins


kernel_bins = brownian_kernel.brownian_coagulation_kernel_via_system_state(
    radius_particle=radius_bins,
    mass_particle=mass_bins,
    temperature=298.15,
    pressure=101325,
)  # Calculate the Brownian coagulation kernel for the radius bins

rng = np.random.default_rng(12345)

# %% sample particle distribution
particle_radius = lognormal_sample_distribution(
    mode=np.array([1e-8, 1e-7]),
    geometric_standard_deviation=np.array([1.1, 1.1]),
    number_of_particles=np.array([5000, 1000]),
    number_of_samples=1000,
)
particle_radius = np.sort(particle_radius)
particles_original = particle_radius.copy()

fig, ax = plt.subplots()
ax.hist(
    particle_radius, bins=100, histtype="step", color="black", density=True
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Frequency")
plt.show()

# %% Coagulation step

# Sort particles by size and bin them
# overflow test for bins
# particle_radius[0] = 5e-10
# particle_radius[-1] = 5e-6
random_concentration = np.random.uniform(1, 10, size=particle_radius.size)
particle_concentration = (
    np.ones_like(particle_radius) * random_concentration * 1e6
)  # particles per m^3
particle_concentration = particle_concentration.astype(np.float64)
particle_concentration_original = particle_concentration.copy()
volume_sim = 1  # m^3
number_in_bins, bins = np.histogram(particle_radius, bins=radius_bins)
print(f"Total number of particles: {np.sum(particle_concentration)}")

# Get the bin index for each particle (0-based indexing)
bin_indices = np.digitize(particle_radius, bins, right=True)
# replace overflow bin with the last bin
bin_indices[bin_indices == len(bins)] = len(bins) - 1
# replace underflow bin with the first bin
bin_indices[bin_indices == 0] = 1
# change to 0-based indexing
bin_indices -= 1
# Precompute the bin pairs for vectorized operations
# change pairs for super-droplet method
unique_bins = np.unique(bin_indices)
# Generate unique pairs directly, no mirror pairs
pair_indices = list(combinations(unique_bins, 2))

# bin the particle concentrations
concentration_in_bins = np.zeros_like(number_in_bins)
for unique_bin in unique_bins:
    concentration_in_bins[unique_bin] = np.sum(
        particle_concentration[bin_indices == unique_bin]
    )
concentration_in_bins.astype(np.float64)


delta_t = 10  # time step in seconds

# Initialize spline for kernel interpolation
interp_kernel = RectBivariateSpline(
    x=radius_bins, y=radius_bins, z=kernel_bins
)
# Initialize spline for kernel interpolation
interp_kernel = RectBivariateSpline(
    x=radius_bins, y=radius_bins, z=kernel_bins
)


# def kernel_nearest_neighbor_extrapolation(
#     interp_func, x_vals, y_vals, x_bins, y_bins
# ):
#     # Clamping x_vals to be within the bounds of x_bins
#     x_vals_clamped = np.clip(x_vals, x_bins[0], x_bins[-1])

#     # Clamping y_vals to be within the bounds of y_bins
#     y_vals_clamped = np.clip(y_vals, y_bins[0], y_bins[-1])

#     # Use the interpolating function with clamped values
#     return interp_func.ev(x_vals_clamped, y_vals_clamped)


# # Use nearest_neighbor_extrapolation instead of interp_kernel.ev directly
# B_values = nearest_neighbor_extrapolation(
#     interp_kernel,
#     particle_radius[indices_i],
#     particle_radius[indices_j],
#     radius_bins,
#     radius_bins,
# )
loss = np.zeros_like(particle_radius)
gain = np.zeros_like(particle_radius)
single_event_counter = np.zeros_like(particle_radius, dtype=int)

# pair_indices.pop(-1)  # Remove the last pair (k, k)

# Super droplet method
# Vectorized operations over bin pairs
for k, l in pair_indices:
    # Kernel value at the max size for the pair of bins
    Kmax = kernel_bins[k, l + 1]

    event_pairs = super_droplet_method.event_pairs(
        lower_bin=k,
        upper_bin=l,
        kernel_max=Kmax,
        number_in_bins=number_in_bins,
        concentration_in_bins=concentration_in_bins,
    )

    # Determine the exact number of events and sample from a
    # Poisson distribution
    N_events = super_droplet_method.sample_events(
        events=event_pairs,
        volume=volume_sim,
        time_step=delta_t,
        generator=rng,
    )

    # Skip iteration if no events are expected
    if N_events == 0:
        continue

    # Limiter, may remove, else always max N_events
    N_events = number_in_bins[k] if N_events > number_in_bins[k] else N_events
    N_events = number_in_bins[l] if N_events > number_in_bins[l] else N_events

    # Get random indices for particles in the bins
    r_i_indices, r_j_indices = super_droplet_method.select_random_indices(
        lower_bin=k,
        upper_bin=l,
        events=N_events,
        number_in_bins=number_in_bins,
        generator=rng,
    )
    # Get the particle indices
    indices_i, indices_j = super_droplet_method.bin_to_particle_indices(
        lower_indices=r_i_indices,
        upper_indices=r_j_indices,
        lower_bin=k,
        upper_bin=l,
        bin_indices=bin_indices,
    )

    # Filter valid indices
    indices_i, indices_j = super_droplet_method.filter_valid_indices(
        small_index=indices_i,
        large_index=indices_j,
        particle_radius=particle_radius,
        single_event_counter=single_event_counter,
    )

    # Early exit for zero-sized indices
    if indices_i.size == 0:
        continue

    # Calulate or interpolate the kernel for the selected particles
    B_values = interp_kernel.ev(
        particle_radius[indices_i], particle_radius[indices_j]
    )

    # Calculate the coagulation events, with random selection
    indices_i, indices_j = super_droplet_method.coagulation_events(
        small_index=indices_i,
        large_index=indices_j,
        kernel_values=B_values,
        kernel_max=Kmax,
        generator=rng,
    )

    particle_radius, particle_concentration, single_event_counter = super_droplet_method.super_droplet_update_step(
        particle_radius=particle_radius,
        concentration=particle_concentration,
        single_event_counter=single_event_counter,
        small_index=indices_i,
        large_index=indices_j,
    )

volume_orginal = np.power(particles_original, 3)
volume_concentration_orginal = volume_orginal * particle_concentration_original
volume_total_original = np.sum(volume_concentration_orginal)

# final volume
volume_final = np.power(particle_radius, 3)
volume_concentration_final = volume_final * particle_concentration
volume_total_final = np.sum(volume_concentration_final)

print(f"Original volume: {volume_total_original}")
print(f"Final volume: {volume_total_final}")
print(f"Percent change in volume: {((volume_total_final - volume_total_original) / volume_total_original) * 100}%")
print(f"Final number of particles: {np.sum(particle_concentration)}")

# %% plot new distribution
fig, ax = plt.subplots()
ax.hist(
    particles_original, bins=100, histtype="step", color="blue", density=True
)
ax.hist(
    particle_radius, bins=100, histtype="step", color="black", density=True
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Frequency")
plt.show()



# %%  check volume conservation
# volume_orginal = particles_original**3
# volume_concentration_orginal = volume_orginal * particle_concentration_original
# volume_total_original = np.sum(volume_concentration_orginal)

# # final volume
# volume_final = particle_radius**3
# volume_concentration_final = volume_final * particle_concentration
# volume_total_final = np.sum(volume_concentration_final)

# print(f"Original volume: {volume_total_original}")
# print(f"Final volume: {volume_total_final}")

# %%
