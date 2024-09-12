"""Particle resolved method for coagulation.
"""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type: ignore

from particula.next.dynamics.coagulation.super_droplet_method import (
    sort_particles,
    bin_particles,
    get_bin_pairs,
    filter_valid_indices,
    sample_events,
    event_pairs,
    coagulation_events,
    random_choice_indices,
)


def particle_resolved_update_step(
    particle_radius: NDArray[np.float64],
    loss: NDArray[np.float64],
    gain: NDArray[np.float64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Update the particle radii and concentrations after coagulation events.

    Args:
        particle_radius (NDArray[float64]): Array of particle radii.
        small_index (NDArray[int64]): Indices corresponding to smaller
            particles.
        large_index (NDArray[int64]): Indices corresponding to larger
            particles.

    Returns:
        - Updated array of particle radii.
        - Updated array for the radii of particles that were lost.
        - Updated array for the radii of particles that were gained.
    """
    # Step 1: Calculate the summed volumes of the smaller and larger particles
    # The volumes are obtained by cubing the radii of the particles.
    sum_radii_cubed = np.power(
        particle_radius[small_index], 3, dtype=np.float64
    ) + np.power(particle_radius[large_index], 3, dtype=np.float64)

    # Step 2: Calculate the new radii formed by the coagulation events
    # The new radius is the cube root of the summed volumes.
    new_radii = np.cbrt(sum_radii_cubed)

    # Step 3: Save out the loss and gain of particles
    loss[small_index] = particle_radius[small_index]
    gain[large_index] = particle_radius[large_index]

    # Step 4: Remove the small particles as they coagulated to the larger ones
    particle_radius[small_index] = 0

    # Step 5: Increase the radii of the large particles to the new radii
    particle_radius[large_index] = new_radii

    return particle_radius, loss, gain


# pylint: disable=too-many-arguments, too-many-locals
def particle_resolved_coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    """
    Perform a single step of particle coagulation, updating particle radii
    based on coagulation events.

    Args:
        particle_radius (NDArray[np.float64]): Array of particle radii.
        kernel (NDArray[np.float64]): Coagulation kernel as a 2D array where
            each element represents the probability of coagulation between
            particles of corresponding sizes.
        kernel_radius (NDArray[np.float64]): Array of radii corresponding to
            the kernel bins.
        volume (float): Volume of the system in which coagulation occurs.
        time_step (float): Time step over which coagulation is calculated.
        random_generator (np.random.Generator): Random number generator for
            stochastic processes.

    Returns:
        Tuple: Updated particle radii, and arrays representing the loss and
            gain in particle counts due to coagulation events.
    """

    _, bin_indices = bin_particles(
        particle_radius, kernel_radius
    )
    # Step 3: Precompute unique bin pairs for efficient coagulation
    pair_indices = get_bin_pairs(bin_indices=bin_indices)

    interp_kernel = RectBivariateSpline(
        x=kernel_radius, y=kernel_radius, z=kernel
    )

    small_index_total0 = np.array([], dtype=np.int64)
    large_index_total0 = np.array([], dtype=np.int64)
    loss = np.zeros_like(particle_radius, dtype=np.float64)
    gain = np.zeros_like(particle_radius, dtype=np.float64)

    # Iterate over each bin pair to calculate potential coagulation events
    for lower_bin, upper_bin in pair_indices:
        # get raidius indexes and filter out zeros
        small_indices = np.flatnonzero(
            (bin_indices == lower_bin) & (particle_radius > 0)
        )
        # filter small indices that are in small indices total
        small_indices = np.setdiff1d(small_indices, small_index_total0)

        large_indices = np.flatnonzero(
            (bin_indices == upper_bin) & (particle_radius > 0)
        )
        if np.size(small_indices) == 0 or np.size(large_indices) == 0:
            continue  # Skip to the next bin pair if no particles are present
        small_count = np.size(small_indices)
        large_count = np.size(large_indices)

        # Retrieve the maximum kernel value for the current bin pair
        kernel_values = interp_kernel.ev(
            np.min(particle_radius[small_indices]),
            np.max(particle_radius[large_indices]),
        )

        # Number of coagulation events
        events = small_count * large_count
        if lower_bin == upper_bin:
            events = small_count * (large_count - 1) / 2
        tests = np.ceil(kernel_values * time_step * events / volume).astype(
            int
        )

        if tests == 0 or events == 0:
            continue

        # Randomly select indices of particles involved in the coagulation
        small_replace = False if small_count > tests else True
        small_index = random_generator.choice(
            small_indices, tests, replace=small_replace
        )
        large_index = random_generator.choice(large_indices, tests)
        kernel_value = interp_kernel.ev(
            particle_radius[small_index], particle_radius[large_index]
        )
        # select diagonal
        if kernel_value.ndim > 1:
            kernel_value = np.diagonal(kernel_value)
        # print(f"kernel value: {kernel_value}")

        # Determine which coagulation events actually occur based on
        # interpolated kernel probabilities
        coagulation_probabilities = (
            kernel_value * time_step * events / (tests * volume)
        )
        # random number
        r = random_generator.uniform(size=tests)
        valid_indices = np.flatnonzero(r < coagulation_probabilities)
        # check if any valid indices are duplicate in small index
        # error of same small particle going to two different large particles
        _, unique_index = np.unique(
            small_index[valid_indices], return_index=True
        )
        small_index = small_index[valid_indices][unique_index]
        large_index = large_index[valid_indices][unique_index]

        # save the coagulation events
        small_index_total0 = np.append(small_index_total0, small_index)
        large_index_total0 = np.append(large_index_total0, large_index)

        # for i in range(tests):
        #     if np.size(small_indices) == 0 or np.size(large_indices) == 0:
        #         continue  # Skip to the next bin pair if no particles are present
        #     # Randomly select indices of particles involved in the coagulation
        #     # events within the current bins
        #     small_index = random_generator.choice(small_indices, 1)
        #     large_index = random_generator.choice(large_indices, 1)

        #     # Interpolate kernel values for the selected particle pairs
        #     kernel_value = interp_kernel.ev(
        #         particle_radius[small_index], particle_radius[large_index]
        #     )

        #     # Determine which coagulation events actually occur based on
        #     # interpolated kernel probabilities
        #     coagulation_probabilities = (
        #         kernel_value[0] * time_step * events / (tests * volume)
        #     )
        #     # random number
        #     r = random_generator.uniform()
        #     # print(f"propbability: {coagulation_probabilities}, random: {r}")

        #     if r < coagulation_probabilities:
        #         small_index_total0 = np.append(small_index_total0, small_index)
        #         large_index_total0 = np.append(large_index_total0, large_index)
        #         # pop out small indices
        #         small_indices = np.delete(
        #             small_indices, small_indices == small_index
        #         )

    commons, small_index_in_common, large_index_in_common = np.intersect1d(
        small_index_total0, large_index_total0, return_indices=True
    )
    # sort based on radius
    radius_argsort = np.argsort(particle_radius[commons])
    commons = commons[radius_argsort]
    small_index_in_common = small_index_in_common[radius_argsort]
    large_index_in_common = large_index_in_common[radius_argsort]

    # remap to largest particle in common
    for i, common in enumerate(commons):
        final_value = large_index_total0[small_index_in_common[i]]
        remap_index = np.flatnonzero(large_index_total0 == common)
        large_index_total0[remap_index] = final_value

    loss_gain_index = np.column_stack([small_index_total0, large_index_total0])

    return particle_radius, loss, gain, loss_gain_index
