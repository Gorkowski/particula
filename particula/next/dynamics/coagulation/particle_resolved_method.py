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
    select_random_indices,
    filter_valid_indices,
    sample_events,
    event_pairs,
    bin_to_particle_indices,
    coagulation_events,
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


def coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    # Step 1: Sort particles by size and obtain indices to revert sorting later
    unsort_indices, sorted_radius, _ = sort_particles(
        particle_radius=particle_radius,
    )
    # Step 2: Bin particles by size using the provided kernel radius bins
    number_in_bins, bin_indices = bin_particles(
        particle_radius=sorted_radius, radius_bins=kernel_radius
    )
    # Step 3: Precompute unique bin pairs for efficient coagulation
    # computations
    pair_indices = get_bin_pairs(bin_indices=bin_indices)

    # Step 4: Initialize a bivariate spline for interpolating kernel values
    # between bin radii
    interp_kernel = RectBivariateSpline(
        x=kernel_radius, y=kernel_radius, z=kernel)

    loss = np.zeros_like(particle_radius, dtype=np.float64)
    gain = np.zeros_like(particle_radius, dtype=np.float64)

    for lower_bin, upper_bin in pair_indices:
        # Step 7.1: Retrieve the maximum kernel value for the current bin pair
        # Note: The '+1' indexing assumes that 'kernel' has dimensions
        # accommodating this offset due to bin edges
        kernel_max = kernel[lower_bin, upper_bin + 1]

        # Step 7.2: Determine potential coagulation events between
        # particles in these bins
        events = event_pairs(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            kernel_max=kernel_max,
            number_in_bins=number_in_bins,
        )

        # Step 7.3: Sample the number of coagulation events from a
        # Poisson distribution
        num_events = sample_events(
            events=events,
            volume=volume,
            time_step=time_step,
            generator=random_generator,
        )

        # Step 7.4: If no events are expected, skip to the next bin pair
        if num_events == 0:
            continue

        # Step 7.6: Randomly select indices of particles involved in the
        # coagulation events within the current bins
        r_i_indices, r_j_indices = select_random_indices(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            events=num_events,
            number_in_bins=number_in_bins,
            generator=random_generator,
        )

        # Step 7.7: Convert bin-relative indices to actual particle indices
        # in the sorted arrays
        indices_i, indices_j = bin_to_particle_indices(
            lower_indices=r_i_indices,
            upper_indices=r_j_indices,
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            bin_indices=bin_indices,
        )

        # Step 7.8: Filter out invalid particle pairs based on their radii
        # and event counters
        indices_i, indices_j = filter_valid_indices(
            small_index=indices_i,
            large_index=indices_j,
            particle_radius=particle_radius,
        )

        # Step 7.9: If no valid indices remain after filtering, skip to
        # the next bin pair
        if indices_i.size == 0:
            continue

        # Step 7.10: Interpolate kernel values for the selected particle pairs
        kernel_values = interp_kernel.ev(
            particle_radius[indices_i], particle_radius[indices_j]
        )

        # Step 7.11: Determine which coagulation events actually occur based
        # on interpolated kernel probabilities
        indices_i, indices_j = coagulation_events(
            small_index=indices_i,
            large_index=indices_j,
            kernel_values=kernel_values,
            kernel_max=kernel_max,
            generator=random_generator,
        )

        # Evaluate the coagulation events
        particle_radius, loss, gain = particle_resolved_update_step(
            particle_radius=particle_radius,
            loss=loss,
            gain=gain,
            small_index=indices_i,
            large_index=indices_j,
        )

    # Step 8: Unsort the particle radii to match the original order
    particle_radius = particle_radius[unsort_indices]
    loss = loss[unsort_indices]  # type: ignore
    gain = gain[unsort_indices]  # type: ignore

    return particle_radius, loss, gain  # type: ignore
