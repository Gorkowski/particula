# Chamber Rate Fitting

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Chamber Rate Fitting

> Auto-generated documentation for [particula.data.process.chamber_rate_fitting](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py) module.

## ChamberParameters

[Show source in chamber_rate_fitting.py:381](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L381)

#### Signature

```python
class ChamberParameters: ...
```



## calculate_optimized_rates

[Show source in chamber_rate_fitting.py:480](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L480)

Calculate the coagulation rates using the optimized parameters and return
the rates and R2 score.

#### Arguments

- `radius_bins` - Array of particle radii in meters.
- `concentration_pmf` - 2D array of concentration PMF values.
- `wall_eddy_diffusivity` - Optimized wall eddy diffusivity.
- `alpha_collision_efficiency` - Optimized alpha collision efficiency.
- `chamber_params` - ChamberParameters object containing chamber-related parameters.
- `dN_dt_concentration_pmf` - Array of observed rate of change of the concentration PMF (optional).

#### Returns

- `coagulation_loss` - Loss rate due to coagulation.
- `coagulation_gain` - Gain rate due to coagulation.
- `dilution_loss` - Loss rate due to dilution.
- `wall_loss_rate` - Loss rate due to wall deposition.
- `net_rate` - Net rate considering all effects.
- `r2_value` - R2 score between the net rate and the observed rate.

#### Signature

```python
def calculate_optimized_rates(
    radius_bins: np.ndarray,
    concentration_pmf: np.ndarray,
    wall_eddy_diffusivity: float,
    alpha_collision_efficiency: float,
    chamber_params: ChamberParameters,
    dN_dt_concentration_pmf: Optional[NDArray[np.float64]] = None,
) -> Tuple[float, float, float, float, float, float]: ...
```

#### See also

- [ChamberParameters](#chamberparameters)



## calculate_pmf_rates

[Show source in chamber_rate_fitting.py:221](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L221)

Calculate the coagulation, dilution, and wall loss rates,
and return the net rate.

#### Arguments

- `radius_bins` - Array of particle radii.
- `concentration_pmf` - Array of particle concentration
    probability mass function.
- `temperature` - Temperature in Kelvin.
- `pressure` - Pressure in Pascals.
- `particle_density` - Density of the particles in kg/m^3.
- `alpha_collision_efficiency` - Collision efficiency factor.
- `volume` - Volume of the chamber in m^3.
- `input_flow_rate` - Input flow rate in m^3/s.
- `wall_eddy_diffusivity` - Eddy diffusivity for wall loss in m^2/s.
- `chamber_dimensions` - Dimensions of the chamber
    (length, width, height) in meters.

#### Returns

- `coagulation_loss` - Loss rate due to coagulation.
- `coagulation_gain` - Gain rate due to coagulation.
- `dilution_loss` - Loss rate due to dilution.
- `wall_loss_rate` - Loss rate due to wall deposition.
- `net_rate` - Net rate considering all effects.

#### Signature

```python
def calculate_pmf_rates(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    alpha_collision_efficiency: float = 1,
    volume: float = 1,
    input_flow_rate: float = 1.6e-07,
    wall_eddy_diffusivity: float = 0.1,
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```



## coagulation_rates_cost_function

[Show source in chamber_rate_fitting.py:320](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L320)

Cost function for the optimization of the eddy diffusivity
and alpha collision efficiency.

#### Signature

```python
def coagulation_rates_cost_function(
    parameters: NDArray[np.float64],
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    dN_dt_concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    volume: float = 1,
    input_flow_rate: float = 1.6e-07,
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),
) -> float: ...
```



## create_guess_and_bounds

[Show source in chamber_rate_fitting.py:390](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L390)

Create the initial guess array and bounds list for the optimization.

#### Arguments

- `guess_eddy_diffusivity` - Initial guess for eddy diffusivity.
- `guess_alpha_collision_efficiency` - Initial guess for alpha collision
    efficiency.
- `bounds_eddy_diffusivity` - Bounds for eddy diffusivity.
- `bounds_alpha_collision_efficiency` - Bounds for alpha collision
    efficiency.

#### Returns

- `initial_guess` - Numpy array of the initial guess values.
- `bounds` - List of tuples representing the bounds for each parameter.

#### Signature

```python
def create_guess_and_bounds(
    guess_eddy_diffusivity: float,
    guess_alpha_collision_efficiency: float,
    bounds_eddy_diffusivity: Tuple[float, float],
    bounds_alpha_collision_efficiency: Tuple[float, float],
) -> Tuple[NDArray[np.float64], List[Tuple[float, float]]]: ...
```



## create_lognormal_2mode_from_fit

[Show source in chamber_rate_fitting.py:105](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L105)

Create a fitted PMF stream and concentration matrix based on
optimized parameters.

#### Arguments

- `radius_min` - Log10 of the minimum radius value in meters (default: -9).
- `radius_max` - Log10 of the maximum radius value in meters (default: -6).
- `num_radius_bins` - Number of radius bins to create between radius_min and radius_max.

#### Returns

- `fitted_pmf_stream` - A Stream object containing the time and fitted concentration PMF data.
- `concentration_m3_pmf_fits` - A numpy array with the fitted concentration PMF values.

#### Signature

```python
def create_lognormal_2mode_from_fit(
    parameters_stream: Stream,
    radius_min: float = 1e-09,
    radius_max: float = 1e-06,
    num_radius_bins: int = 250,
) -> Tuple[Stream, NDArray[np.float64]]: ...
```

#### See also

- [Stream](../stream.md#stream)



## fit_lognormal_2mode_pdf

[Show source in chamber_rate_fitting.py:23](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L23)

Generate initial guesses using a machine learning model, optimize them,
and return a Stream object with the results.

#### Arguments

- `experiment_time` - Array of experiment time points.
- `radius_m` - Array of particle radii in meters.
- `concentration_m3_pdf` - 2D array of concentration PDFs for each
    time point.

#### Returns

- `fitted_stream` - A Stream object containing the initial guesses,
    optimized values, and R² scores.

#### Signature

```python
def fit_lognormal_2mode_pdf(
    experiment_time: np.ndarray, radius_m: np.ndarray, concentration_m3_pdf: np.ndarray
) -> Stream: ...
```

#### See also

- [Stream](../stream.md#stream)



## optimize_and_calculate_rates_looped

[Show source in chamber_rate_fitting.py:546](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L546)

Perform optimization and calculate rates for each time point in the stream.

#### Arguments

- `pmf_stream` - Stream object containing the fitted PMF data.
- `pmf_derivative_stream` - Stream object containing the derivative of the PMF data.
- `chamber_parameters` - ChamberParameters object containing chamber-related parameters.
- `fit_guess` - Initial guess for the optimization.
- `fit_bounds` - Bounds for the optimization parameters.

#### Returns

- `result_stream` - Stream containing the optimization results for each time point.
- `coagulation_loss_stream` - Stream containing coagulation loss rates.
- `coagulation_gain_stream` - Stream containing coagulation gain rates.
- `coagulation_net_stream` - Stream containing net coagulation rates.
- `dilution_loss_stream` - Stream containing dilution loss rates.
- `wall_loss_rate_stream` - Stream containing wall loss rates.
- `total_rate_stream` - Stream containing total rates.

#### Signature

```python
def optimize_and_calculate_rates_looped(
    pmf_stream: Stream,
    pmf_derivative_stream: Stream,
    chamber_parameters: ChamberParameters,
    fit_guess: NDArray[np.float64],
    fit_bounds: List[Tuple[float, float]],
) -> Tuple[Stream, Stream, Stream, Stream, Stream, Stream, Stream]: ...
```

#### See also

- [ChamberParameters](#chamberparameters)
- [Stream](../stream.md#stream)



## optimize_chamber_parameters

[Show source in chamber_rate_fitting.py:436](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L436)

Optimize the eddy diffusivity and alpha collision efficiency parameters.

#### Returns

- `wall_eddy_diffusivity_optimized` - Optimized wall eddy diffusivity.
- `alpha_collision_efficiency_optimized` - Optimized alpha collision
efficiency.

#### Signature

```python
def optimize_chamber_parameters(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    dN_dt_concentration_pmf: NDArray[np.float64],
    chamber_params: ChamberParameters,
    fit_guess: NDArray[np.float64] = np.array([0.1, 1]),
    fit_bounds: List[Tuple[float, float]] = [(0.01, 1), (0.1, 1)],
    minimize_method: str = "L-BFGS-B",
) -> Tuple[float, float]: ...
```

#### See also

- [ChamberParameters](#chamberparameters)



## optimize_parameters

[Show source in chamber_rate_fitting.py:420](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L420)

#### Signature

```python
def optimize_parameters(
    cost_function: callable,
    initial_guess: np.ndarray,
    bounds: List[Tuple[float, float]],
    method: str,
) -> Tuple[float, float]: ...
```



## time_derivative_of_pmf_fits

[Show source in chamber_rate_fitting.py:161](https://github.com/Gorkowski/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L161)

Calculate the rate of change of the concentration PMF over time and
return a new stream.

#### Arguments

- `pmf_fitted_stream` - Stream object containing the fitted concentration
    PMF data.
- `window_size` - Size of the time window for fitting the slope.

#### Returns

- `rate_of_change_stream` - Stream object containing the rate of
    change of the concentration PMF.

#### Signature

```python
def time_derivative_of_pmf_fits(
    pmf_fitted_stream: Stream, liner_slope_window_size: int = 12
) -> Stream: ...
```

#### See also

- [Stream](../stream.md#stream)
