# Generate And Train 2mode Sizer

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Data](../../index.md#data) / [Process](../index.md#process) / [Ml Analysis](./index.md#ml-analysis) / Generate And Train 2mode Sizer

> Auto-generated documentation for [particula.data.process.ml_analysis.generate_and_train_2mode_sizer](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py) module.

#### Attributes

- `logger` - Set up logging: logging.getLogger('particula')

- `TOTAL_NUMBER_SIMULATED` - Training parameters: 1000000


## create_pipeline

[Show source in generate_and_train_2mode_sizer.py:177](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L177)

Create a pipeline with normalization and MLPRegressor model.

#### Returns

A scikit-learn Pipeline object.

#### Signature

```python
def create_pipeline() -> Pipeline: ...
```



## evaluate_pipeline

[Show source in generate_and_train_2mode_sizer.py:310](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L310)

Evaluate the pipeline and print the mean squared error for each target.

#### Arguments

- `pipeline` - The trained pipeline.
- `X_test` - The test feature array.
- `y_test` - The test target array.

#### Signature

```python
def evaluate_pipeline(
    pipeline: Pipeline, x_test: NDArray[np.float64], y_test: NDArray[np.float64]
) -> None: ...
```



## generate_simulated_data

[Show source in generate_and_train_2mode_sizer.py:44](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L44)

Generate simulated lognormal aerosol particle size distributions.

#### Arguments

- `total_number_simulated` - Total number of simulated distributions.
- `number_of_modes_sim` - Number of modes to simulate (1, 2, or 3).
- `x_array_max_index` - Number of size bins in the particle size array.
- `lower_bound_gsd` - Lower bound for the geometric standard deviation.
- `upper_bound_gsd` - Upper bound for the geometric standard deviation.
- `seed` - Random seed for reproducibility.

#### Returns

- `x_values` - Array of particle sizes.
- `mode_index_sim` - Array of simulated mode indices.
- `geomertic_standard_deviation_sim` - Array of simulated geometric
    standard deviations (GSDs).
- `number_of_particles_sim` - Array of simulated relative number
    concentrations.
- `number_pdf_sim` - Array of simulated probability density
    functions (PDFs).

#### Signature

```python
def generate_simulated_data(
    total_number_simulated: int = 10000,
    number_of_modes_sim: int = 2,
    x_array_max_index: int = 128,
    lower_bound_gsd: float = 1.0,
    upper_bound_gsd: float = 2.0,
    seed: int = 0,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```



## load_and_cache_pipeline

[Show source in generate_and_train_2mode_sizer.py:362](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L362)

Load and cache the ML pipeline if not already loaded.

#### Arguments

- `filename` - Path to the pipeline file.

#### Returns

The loaded pipeline.

#### Signature

```python
def load_and_cache_pipeline(filename: str) -> Pipeline: ...
```



## load_pipeline

[Show source in generate_and_train_2mode_sizer.py:349](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L349)

Load a pipeline from a file.

#### Arguments

- `filename` - The filename to load the pipeline from.

#### Returns

The loaded pipeline.

#### Signature

```python
def load_pipeline(filename: str) -> Pipeline: ...
```



## lognormal_2mode_cost_function

[Show source in generate_and_train_2mode_sizer.py:532](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L532)

Cost function for the lognormal distribution with 2 modes.

#### Arguments

- `params` - Combined array of mode_values, geometric_standard_deviation,
    and number_of_particles.
- `x_values` - The x-values (particle sizes).
- `concentration_pdf` - The actual concentration PDF to fit.

#### Returns

The mean squared error between the actual and guessed concentration
    PDF.

#### Signature

```python
def lognormal_2mode_cost_function(
    params: NDArray[np.float64],
    x_values: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> float: ...
```



## lognormal_2mode_ml_guess

[Show source in generate_and_train_2mode_sizer.py:448](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L448)

Load the machine learning pipeline, interpolate the concentration PDF,
and predict lognormal parameters.

#### Arguments

- `file_name` - Path to the saved ML pipeline file.
- `x_values` - Array of x-values (particle sizes).
- `concentration_pdf` - Array of concentration PDF values.

#### Returns

- `mode_values_guess` - Predicted mode values after rescaling.
- `geometric_standard_deviation_guess` - Predicted geometric standard
    deviations after rescaling.
- `number_of_particles_guess` - Predicted number of particles after
    rescaling.

#### Signature

```python
def lognormal_2mode_ml_guess(
    logspace_x: NDArray[np.float64], concentration_pdf: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
```



## normalize_max

[Show source in generate_and_train_2mode_sizer.py:129](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L129)

Normalize each sample in X by dividing by its maximum value.

#### Arguments

- `X` - The input array to be normalized.

#### Returns

The normalized array.

#### Signature

```python
def normalize_max(x_input: NDArray[np.float64]) -> NDArray[np.float64]: ...
```



## normalize_targets

[Show source in generate_and_train_2mode_sizer.py:142](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L142)

Normalize the mode index, GSD, and relative number concentration.

#### Arguments

- `mode_index_sim` - Array of mode indices.
- `geomertic_standard_deviation_sim` - Array of geometric standard
    deviations (GSDs).
- `number_of_particles_sim` - Array of relative number concentrations.
- `x_array_max_index` - Maximum index for the mode.
- `lower_bound_gsd` - Lower bound for the geometric standard
    deviation (GSD).
- `upper_bound_gsd` - Upper bound for the geometric standard
    deviation (GSD).

#### Returns

- `y` - Normalized array combining mode indices, GSDs, and relative
    number concentrations.

#### Signature

```python
def normalize_targets(
    mode_index_sim: NDArray[np.int64],
    geomertic_standard_deviation_sim: NDArray[np.float64],
    number_of_particles_sim: NDArray[np.float64],
    x_array_max_index: int,
    lower_bound_gsd: float,
    upper_bound_gsd: float,
) -> NDArray[np.float64]: ...
```



## optimize_lognormal_2mode

[Show source in generate_and_train_2mode_sizer.py:584](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L584)

Optimize the lognormal 2-mode distribution parameters using multiple
optimization methods.

#### Arguments

- `initial_guess` - Initial guess for the optimization parameters.
- `x_values` - Array of x-values (particle sizes).
- `concentration_pdf` - The actual concentration PDF to fit.
- `bounds` - Bounds for the optimization parameters.
- `list_of_methods` - List of optimization methods to try.

#### Returns

A dictionary with the best optimization results, including:
    - `-` *best_method* - The optimization method that gave the best result.
    - `-` *optimized_mode_values* - Optimized mode values.
    - `-` *optimized_gsd* - Optimized geometric standard deviations.
    - `-` *optimized_number_of_particles* - Optimized number of particles.
    - `-` *r2_score* - The R² score of the best fit.
    - `-` *best_result* - The full result object from scipy.optimize.minimize.

#### Signature

```python
def optimize_lognormal_2mode(
    mode_guess: NDArray[np.float64],
    geometric_standard_deviation_guess: NDArray[np.float64],
    number_of_particles_in_mode_guess: NDArray[np.float64],
    x_values: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
    bounds: ignore = None,
    list_of_methods: List[str] = [
        "Nelder-Mead",
        "Powell",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
    ],
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, dict[Any, Any]
]: ...
```



## save_pipeline

[Show source in generate_and_train_2mode_sizer.py:338](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L338)

Save the trained pipeline to a file.

#### Arguments

- `pipeline` - The trained pipeline.
- `filename` - The filename to save the pipeline to.

#### Signature

```python
def save_pipeline(pipeline: Pipeline, filename: str) -> None: ...
```



## train_network_and_save

[Show source in generate_and_train_2mode_sizer.py:382](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L382)

Train the neural network and save the pipeline.

#### Signature

```python
def train_network_and_save(): ...
```



## train_pipeline

[Show source in generate_and_train_2mode_sizer.py:204](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L204)

Train the pipeline and return the trained model along with train/test data.

#### Arguments

- `X` - The feature array.
- `y` - The target array.
- `test_size` - The proportion of the dataset to include in the test split.
- `random_state` - Random seed for reproducibility.

#### Returns

- `pipeline` - The trained pipeline.
X_train, X_test, y_train, y_test: The training and testing data splits.

#### Signature

```python
def train_pipeline(
    x_input: NDArray[np.float64],
    y: NDArray[np.float64],
    test_split_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[
    Pipeline,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```



## train_pipeline_with_progress

[Show source in generate_and_train_2mode_sizer.py:243](https://github.com/Gorkowski/particula/blob/main/particula/data/process/ml_analysis/generate_and_train_2mode_sizer.py#L243)

Train the pipeline in batches with progress tracking, and return the
trained model along with train/test data.

#### Arguments

- `X` - The feature array.
- `y` - The target array.
- `test_size` - The proportion of the dataset to include in the test split.
- `random_state` - Random seed for reproducibility.
- `n_batches` - Number of batches to split the training into.

#### Returns

- `pipeline` - The trained pipeline.
X_train, X_test, y_train, y_test: The training and testing data splits.

#### Signature

```python
def train_pipeline_with_progress(
    x_input: NDArray[np.float64],
    y: NDArray[np.float64],
    test_split_size: float = 0.3,
    random_state: int = 42,
    n_batches: int = 10,
) -> Tuple[
    Pipeline,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```
