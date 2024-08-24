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
    geometric_standard_deviation=np.array([1.3, 2.0]),
    number_of_particles=np.array([5000, 1000]),
    number_of_samples=1000,
)
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
random_concentration = np.random.randint(1, 10, size=particle_radius.size)
particle_concentration = (
    np.ones_like(particle_radius) * random_concentration * 1e4
)  # particles per m^3
volume_sim = 1  # m^3
particle_radius = np.sort(particle_radius)
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


delta_t = 1  # time step in seconds

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
single_event_counter = np.zeros_like(particle_radius)

# pair_indices.pop(-1)  # Remove the last pair (k, k)

# Vectorized operations over bin pairs
for k, l in pair_indices:
    # Kernel value at the max size for the pair of bins
    Kmax = kernel_bins[k, l + 1]

    # Calculate the number of particle pairs
    if k != l:
        N_pairs = (
            Kmax
            * number_in_bins[k]
            * number_in_bins[l]
            * concentration_in_bins[k]
            * concentration_in_bins[l]
        )
    else:
        N_pairs = (
            Kmax
            * 0.5
            * number_in_bins[k]
            * (number_in_bins[l] - 1)
            * concentration_in_bins[k]
            * (concentration_in_bins[l])
        )

    # Determine the exact number of events and sample from a Poisson distribution
    N_events_exact = N_pairs / volume_sim
    N_events = rng.poisson(N_events_exact * delta_t)

    # Skip iteration if no events are expected
    if N_events == 0:
        continue

    # Vectorized random particle selection, may remove if always max N_events
    N_events = number_in_bins[k] if N_events > number_in_bins[k] else N_events
    N_events = number_in_bins[l] if N_events > number_in_bins[l] else N_events

    r_i_indices = np.random.randint(
        0,
        number_in_bins[k],
        size=N_events,
        dtype=int,
    )
    r_j_indices = np.random.randint(
        0,
        number_in_bins[l],
        size=N_events,
        dtype=int,
    )

    # Get the indices in the particle array where the bins start
    start_index_k = np.searchsorted(bin_indices, k)
    start_index_l = np.searchsorted(bin_indices, l)

    # Actual indices in the particle array
    indices_i = start_index_k + r_i_indices
    indices_j = start_index_l + r_j_indices

    # Early exit for zero-sized particles due to previous coagulation
    valid_indices = (
        (particle_radius[indices_i] > 0)
        & (particle_radius[indices_j] > 0)
        & (single_event_counter[indices_i] < 1)
        & (single_event_counter[indices_j] < 1)
    )
    if not np.any(valid_indices):
        continue

    # process only valid indices
    indices_i = indices_i[valid_indices]
    indices_j = indices_j[valid_indices]

    # Calulate or interpolate the kernel for the selected particles
    B_values = interp_kernel.ev(
        particle_radius[indices_i], particle_radius[indices_j]
    )

    # Perform coagulation with probability B/Kmax
    coagulation_probabilities = B_values / Kmax
    print(coagulation_probabilities)
    coagulation_events = (
        rng.random(len(coagulation_probabilities)) < coagulation_probabilities
    )

    # process only coagulation events, i indices coagulate into j indices
    indices_i = indices_i[coagulation_events]
    indices_j = indices_j[coagulation_events]

    # Calculate new radii for coagulation events
    new_radii = (
        particle_radius[indices_i] ** 3 + particle_radius[indices_j] ** 3
    ) ** (1 / 3)

    # Save radii gain and loss for coagulation events
    loss[indices_i] = particle_radius[indices_i]
    loss[indices_j] = particle_radius[indices_j]
    gain[indices_j] = new_radii

    # Concentration change for coagulation events
    concentration_delta = (
        particle_concentration[indices_i] - particle_concentration[indices_j]
    )
    small_particle_concentration = concentration_delta > 0
    large_particle_concentration = concentration_delta < 0
    split_concentration = concentration_delta == 0

    # equal number of large and small particles
    if np.any(split_concentration):
        # Split the concentration for equal mass
        particle_concentration[indices_i[split_concentration]] /= 2
        particle_concentration[indices_j[split_concentration]] /= 2
        # Update the particle array
        particle_radius[indices_i[split_concentration]] = new_radii[
            split_concentration
        ]
        particle_radius[indices_j[split_concentration]] = new_radii[
            split_concentration
        ]
        # keep track of the number of events per particle
        single_event_counter[indices_i[split_concentration]] += 1
        single_event_counter[indices_j[split_concentration]] += 1

    # more large particles than small particles
    if np.any(large_particle_concentration):
        # remove the concentration from large particles, small concentration stays the same
        particle_concentration[indices_j[large_particle_concentration]] = (
            np.abs(concentration_delta[large_particle_concentration])
        )
        # Update the particle array, small concentration grow in size
        particle_radius[indices_i[large_particle_concentration]] = new_radii[large_particle_concentration]
        # indices j stay the same

    # more small particles than large particles
    if np.any(small_particle_concentration):
        # remove the concentration from small particles, large concentration stays the same
        particle_concentration[indices_i[small_particle_concentration]] = (
            np.abs(concentration_delta[small_particle_concentration])
        )
        # Update the particle array, large concentration grow in size
        particle_radius[indices_j[small_particle_concentration]] = new_radii[small_particle_concentration]
        # indices i stay the same

#
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
# %% plot loss and gain
loss = loss[loss != 0]
gain = gain[gain != 0]
fig, ax = plt.subplots()
ax.hist(loss, bins=100, histtype="step", color="red", density=True)
ax.hist(gain, bins=100, histtype="step", color="green", density=True)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Frequency")
plt.show()


# %%  check volume conservation
volume_loss = np.sum(loss**3)
volume_gain = np.sum(gain**3)
print(volume_loss, volume_gain)

# %%
