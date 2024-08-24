# Super Droplet Method

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Super Droplet Method

> Auto-generated documentation for [particula.next.dynamics.coagulation.super_droplet_method](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py) module.

## bin_to_particle_indices

[Show source in super_droplet_method.py:222](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L222)

Convert bin indices to actual particle indices in the particle array.

This function calculates the actual indices in the particle array
corresponding to the bins specified by `lower_bin` and `upper_bin`.
The function adjusts the provided bin-relative indices to reflect
their position in the full particle array.

#### Arguments

lower_indices : Array of indices relative to the start of
    the `lower_bin`.
upper_indices : Array of indices relative to the start of
    the `upper_bin`.
lower_bin : Index of the bin containing smaller particles.
upper_bin : Index of the bin containing larger particles.
bin_indices : Array containing the start indices of each bin in the
    particle array.

#### Returns

Tuple :
    - `-` *`small_index`* - Indices of particles from the `lower_bin`.
    - `-` *`large_index`* - Indices of particles from the `upper_bin`.

#### Signature

```python
def bin_to_particle_indices(
    lower_indices: NDArray[np.int64],
    upper_indices: NDArray[np.int64],
    lower_bin: int,
    upper_bin: int,
    bin_indices: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## coagulation_events

[Show source in super_droplet_method.py:310](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L310)

Calculate coagulation probabilities and filter events based on them.

This function calculates the probability of coagulation events occurring
between pairs of particles, based on the ratio of the kernel value for
each pair to the maximum kernel value for the bins. The function then
randomly determines which events occur using these probabilities.

#### Arguments

small_index : Array of indices for the first set of particles
    (smaller particles) involved in the events.
large_index : Array of indices for the second set of particles
    (larger particles) involved in the events.
kernel_values : Array of kernel values corresponding to the
    particle pairs.
kernel_max : The maximum kernel value used for normalization
    of probabilities.
generator : A NumPy random generator used to sample random numbers.

#### Returns

Tuple:
    - Filtered `small_index` array containing indices where
        coagulation events occurred.
    - Filtered `large_index` array containing indices where
        coagulation events occurred.

#### Signature

```python
def coagulation_events(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    kernel_values: NDArray[np.float64],
    kernel_max: float,
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## event_pairs

[Show source in super_droplet_method.py:108](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L108)

Calculate the number of particle pairs based on kernel value.

#### Arguments

lower_bin : Lower bin index.
upper_bin : Upper bin index.
kernel_max : Maximum value of the kernel.
number_in_bins : Number of particles in each bin.
concentration_in_bins : Concentration of particles in each bin.
    Default is None.

#### Returns

The number of particle pairs events based on the kernel and
number of particles in the bins.

#### Signature

```python
def event_pairs(
    lower_bin: int,
    upper_bin: int,
    kernel_max: Union[float, NDArray[np.float64]],
    number_in_bins: NDArray[np.int64],
    concentration_in_bins: Optional[NDArray[np.float64]] = None,
) -> float: ...
```



## filter_valid_indices

[Show source in super_droplet_method.py:264](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L264)

Filter particles indices based on particle radius and event counters.

This function filters out particle indices that are considered invalid
based on two criteria:
1. The particle radius must be greater than zero.
2. If provided, the single event counter must be less than one.

#### Arguments

small_index : Array of indices for particles in the smaller bin.
large_index : Array of indices for particles in the larger bin.
particle_radius : Array containing the radii of particles.
single_event_counter (Optional) : Optional array tracking the
    number of events for each particle. If provided, only particles
    with a counter value less than one are valid.

#### Returns

Tuple :
    - Filtered `small_index` array containing only valid indices.
    - Filtered `large_index` array containing only valid indices.

#### Signature

```python
def filter_valid_indices(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
    single_event_counter: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## sample_events

[Show source in super_droplet_method.py:150](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L150)

Sample the number of coagulation events from a Poisson distribution.

This function calculates the expected number of coagulation events based on
the number of particle pairs, the simulation volume, and the time step. It
then samples the actual number of events using a Poisson distribution.

#### Arguments

events : The calculated number of particle pairs that could
    interact.
volume : The volume of the simulation space.
time_step : The time step over which the events are being simulated.
generator : A NumPy random generator used to sample from the Poisson
    distribution.

#### Returns

The sampled number of coagulation events as an integer.

#### Signature

```python
def sample_events(
    events: float, volume: float, time_step: float, generator: np.random.Generator
) -> int: ...
```



## select_random_indices

[Show source in super_droplet_method.py:181](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L181)

Select random indices for particles involved in coagulation events.

This function generates random indices for particles in the specified bins
(`lower_bin` and `upper_bin`) that are involved in a specified number of
events. The indices are selected based on the number of particles in
each bin.

#### Arguments

lower_bin : Index of the bin containing smaller particles.
upper_bin : Index of the bin containing larger particles.
events : The number of events to sample indices for.
number_in_bins : Array representing the number of particles in
    each bin.
- `generator` - A NumPy random generator used to sample indices.

#### Returns

Tuple :
    - Indices of particles from `lower_bin`.
    - Indices of particles from `upper_bin`.

#### Signature

```python
def select_random_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    number_in_bins: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## super_droplet_update_step

[Show source in super_droplet_method.py:12](https://github.com/Gorkowski/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L12)

Update the particle radii and concentrations after coagulation events.

#### Arguments

particle_radius : Array of particle radii.
concentration : Array representing the concentration of particles.
single_event_counter : Tracks the number of coagulation events for
    each particle.
small_index : Indices corresponding to smaller particles.
large_index : Indices corresponding to larger particles.

#### Returns

Tuple :
- Updated array of particle radii.
- Updated array representing the concentration of particles.
- Updated array tracking the number of coagulation events.

#### Signature

```python
def super_droplet_update_step(
    particle_radius: NDArray[np.float64],
    concentration: NDArray[np.float64],
    single_event_counter: NDArray[np.int64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]: ...
```
