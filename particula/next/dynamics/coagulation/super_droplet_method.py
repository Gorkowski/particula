"""
Super droplet method for coagulation dynamics.

Need to validate the code.
"""

from typing import Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray


def super_droplet_update_step(
    particle_radius: NDArray[np.float64],
    concentration: NDArray[np.float64],
    single_event_counter: NDArray[np.int64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """
    Update the particle radii and concentrations after coagulation events.

    Args:
        particle_radius : Array of particle radii.
        concentration : Array representing the concentration of particles.
        single_event_counter : Tracks the number of coagulation events for
            each particle.
        small_index : Indices corresponding to smaller particles.
        large_index : Indices corresponding to larger particles.

    Returns:
        Tuple :
        - Updated array of particle radii.
        - Updated array representing the concentration of particles.
        - Updated array tracking the number of coagulation events.
    """

    # Step 1: Calculate the summed volumes of the smaller and larger particles
    # The volumes are obtained by cubing the radii of the particles.
    sum_radii_cubed = np.power(
        particle_radius[small_index], 3, dtype=np.float64
    ) + np.power(particle_radius[large_index], 3, dtype=np.float64)

    # Step 2: Calculate the new radii formed by the coagulation events
    # The new radius is the cube root of the summed volumes.
    new_radii = np.cbrt(sum_radii_cubed)

    # Step 3: Determine the concentration differences between small and
    # large particles
    concentration_delta = (
        concentration[small_index] - concentration[large_index]
    )
    small_concentration = (
        concentration_delta > 0
    )  # More small particles than large
    large_concentration = (
        concentration_delta < 0
    )  # More large particles than small
    split_concentration = (
        concentration_delta == 0
    )  # Equal number of small and large particles

    # Step 4: Handle cases where small and large particle concentrations are
    # equal. In these cases, split the concentrations equally and update
    # both small and large particle radii.
    if np.any(split_concentration):
        # print("Handling equal concentration case (split)")
        concentration[small_index[split_concentration]] /= 2
        concentration[large_index[split_concentration]] /= 2

        particle_radius[small_index[split_concentration]] = new_radii[
            split_concentration
        ]
        particle_radius[large_index[split_concentration]] = new_radii[
            split_concentration
        ]

    # Step 5: Handle cases where there are more large particles than small ones
    # Update the concentration of large particles and adjust the radii of
    # small particles.
    if np.any(large_concentration):
        # print("Handling more large particles case")
        concentration[large_index[large_concentration]] = np.abs(
            concentration_delta[large_concentration]
        )
        particle_radius[small_index[large_concentration]] = new_radii[
            large_concentration
        ]

    # Step 6: Handle cases where there are more small particles than large ones
    # Update the concentration of small particles and adjust the radii of
    # large particles.
    if np.any(small_concentration):
        # print("Handling more small particles case")
        concentration[small_index[small_concentration]] = np.abs(
            concentration_delta[small_concentration]
        )
        particle_radius[large_index[small_concentration]] = new_radii[
            small_concentration
        ]

    # Increment event counters for both small and large particles
    single_event_counter[small_index] += 1
    single_event_counter[large_index] += 1

    return particle_radius, concentration, single_event_counter


def event_pairs(
    lower_bin: int,
    upper_bin: int,
    kernel_max: Union[float, NDArray[np.float64]],
    number_in_bins: NDArray[np.int64],
    concentration_in_bins: Optional[NDArray[np.float64]] = None
) -> float:
    """Calculate the number of particle pairs based on kernel value.

    Args:
        lower_bin : Lower bin index.
        upper_bin : Upper bin index.
        kernel_max : Maximum value of the kernel.
        number_in_bins : Number of particles in each bin.
        concentration_in_bins : Concentration of particles in each bin.
            Default is None.

    Returns:
        The number of particle pairs events based on the kernel and
        number of particles in the bins.
    """
    concentration_scaling = (
        concentration_in_bins[lower_bin] * concentration_in_bins[upper_bin]
        if concentration_in_bins is not None else 1.0
    )
    # Calculate the number of particle pairs based on the kernel value
    if lower_bin != upper_bin:
        return (
            kernel_max
            * number_in_bins[lower_bin]
            * number_in_bins[upper_bin]
            * concentration_scaling
        )
    return (
        kernel_max
        * 0.5
        * number_in_bins[lower_bin]
        * (number_in_bins[upper_bin] - 1)
        * concentration_scaling
    )


def sample_events(
    events: float,
    volume: float,
    time_step: float,
    generator: np.random.Generator,
) -> int:
    """
    Sample the number of coagulation events from a Poisson distribution.

    This function calculates the expected number of coagulation events based on
    the number of particle pairs, the simulation volume, and the time step. It
    then samples the actual number of events using a Poisson distribution.

    Args:
        events : The calculated number of particle pairs that could
            interact.
        volume : The volume of the simulation space.
        time_step : The time step over which the events are being simulated.
        generator : A NumPy random generator used to sample from the Poisson
            distribution.

    Returns:
        The sampled number of coagulation events as an integer.
    """
    # Calculate the expected number of events
    events_exact = events / volume

    # Sample the actual number of events from a Poisson distribution
    return generator.poisson(events_exact * time_step)


def select_random_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    number_in_bins: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Select random indices for particles involved in coagulation events.

    This function generates random indices for particles in the specified bins
    (`lower_bin` and `upper_bin`) that are involved in a specified number of
    events. The indices are selected based on the number of particles in
    each bin.

    Args:
        lower_bin : Index of the bin containing smaller particles.
        upper_bin : Index of the bin containing larger particles.
        events : The number of events to sample indices for.
        number_in_bins : Array representing the number of particles in
            each bin.
        generator: A NumPy random generator used to sample indices.

    Returns:
        Tuple :
            - Indices of particles from `lower_bin`.
            - Indices of particles from `upper_bin`.
    """
    # Select random indices for particles in the lower_bin
    lower_indices = generator.integers(  # type: ignore
        0, number_in_bins[lower_bin],
        size=events, endpoint=False, dtype=np.int64
    )
    # Select random indices for particles in the upper_bin
    upper_indices = generator.integers(  # type: ignore
        0, number_in_bins[upper_bin],
        size=events, endpoint=False, dtype=np.int64
    )
    return lower_indices, upper_indices


def bin_to_particle_indices(
    lower_indices: NDArray[np.int64],
    upper_indices: NDArray[np.int64],
    lower_bin: int,
    upper_bin: int,
    bin_indices: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Convert bin indices to actual particle indices in the particle array.

    This function calculates the actual indices in the particle array
    corresponding to the bins specified by `lower_bin` and `upper_bin`.
    The function adjusts the provided bin-relative indices to reflect
    their position in the full particle array.

    Args:
        lower_indices : Array of indices relative to the start of
            the `lower_bin`.
        upper_indices : Array of indices relative to the start of
            the `upper_bin`.
        lower_bin : Index of the bin containing smaller particles.
        upper_bin : Index of the bin containing larger particles.
        bin_indices : Array containing the start indices of each bin in the
            particle array.

    Returns:
        Tuple :
            - `small_index`: Indices of particles from the `lower_bin`.
            - `large_index`: Indices of particles from the `upper_bin`.
    """
    # Get the start index in the particle array for the lower_bin
    start_index_lower_bin = np.searchsorted(bin_indices, lower_bin)
    # Get the start index in the particle array for the upper_bin
    start_index_upper_bin = np.searchsorted(bin_indices, upper_bin)

    # Calculate the actual particle indices for the lower_bin and upper_bin
    small_index = start_index_lower_bin + lower_indices
    large_index = start_index_upper_bin + upper_indices

    return small_index, large_index


def filter_valid_indices(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
    single_event_counter: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Filter particles indices based on particle radius and event counters.

    This function filters out particle indices that are considered invalid
    based on two criteria:
    1. The particle radius must be greater than zero.
    2. If provided, the single event counter must be less than one.

    Args:
        small_index : Array of indices for particles in the smaller bin.
        large_index : Array of indices for particles in the larger bin.
        particle_radius : Array containing the radii of particles.
        single_event_counter (Optional) : Optional array tracking the
            number of events for each particle. If provided, only particles
            with a counter value less than one are valid.

    Returns:
        Tuple :
            - Filtered `small_index` array containing only valid indices.
            - Filtered `large_index` array containing only valid indices.
    """
    if single_event_counter is not None:
        # Both particle radius and event counter are used to determine
        # valid indices
        valid_indices = (
            (particle_radius[small_index] > 0)
            & (particle_radius[large_index] > 0)
            & (single_event_counter[small_index] < 1)
            & (single_event_counter[large_index] < 1)
        )
    else:
        # Only particle radius is used to determine valid indices
        valid_indices = (particle_radius[small_index] > 0) & (
            particle_radius[large_index] > 0
        )

    # Return the filtered indices
    return small_index[valid_indices], large_index[valid_indices]


def coagulation_events(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    kernel_values: NDArray[np.float64],
    kernel_max: float,
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Calculate coagulation probabilities and filter events based on them.

    This function calculates the probability of coagulation events occurring
    between pairs of particles, based on the ratio of the kernel value for
    each pair to the maximum kernel value for the bins. The function then
    randomly determines which events occur using these probabilities.

    Args:
        small_index : Array of indices for the first set of particles
            (smaller particles) involved in the events.
        large_index : Array of indices for the second set of particles
            (larger particles) involved in the events.
        kernel_values : Array of kernel values corresponding to the
            particle pairs.
        kernel_max : The maximum kernel value used for normalization
            of probabilities.
        generator : A NumPy random generator used to sample random numbers.

    Returns:
        Tuple:
            - Filtered `small_index` array containing indices where
                coagulation events occurred.
            - Filtered `large_index` array containing indices where
                coagulation events occurred.
    """
    # Calculate the coagulation probabilities for each particle pair
    coagulation_probabilities = kernel_values / kernel_max

    # Determine which events occur based on these probabilities
    coagulation_occurs = (
        generator.random(len(coagulation_probabilities))
        < coagulation_probabilities
    )

    # Return the indices of particles that underwent coagulation
    return small_index[coagulation_occurs], large_index[coagulation_occurs]
