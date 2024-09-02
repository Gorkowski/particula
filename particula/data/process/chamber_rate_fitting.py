"""
Functions for fitting the chamber rates to the observed rates.
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from functools import partial


from sklearn.linear_model import LinearRegression
from particula.util import convert, time_manage

from numpy.typing import NDArray

import matplotlib.pyplot as plt

from particula.next.particles.properties import (
    lognormal_pdf_distribution,
    lognormal_pmf_distribution,
)
from particula.data.process.ml_analysis import generate_and_train_2mode_sizer
from particula.data.stream import Stream
from particula.data import loader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from scipy.optimize import minimize  # type: ignore

from particula.util.convert import distribution_convert_pdf_pms
from particula.next.dynamics import dilution, wall_loss, coagulation

from particula.util.input_handling import convert_units


def fit_lognormal_2mode_pdf(
    experiment_time: np.ndarray,
    radius_m: np.ndarray,
    concentration_m3_pdf: np.ndarray,
) -> Stream:
    """
    Generate initial guesses using a machine learning model, optimize them,
    and return a Stream object with the results.

    Arguments:
        experiment_time: Array of experiment time points.
        radius_m: Array of particle radii in meters.
        concentration_m3_pdf: 2D array of concentration PDFs for each
            time point.

    Returns:
        fitted_stream: A Stream object containing the initial guesses,
            optimized values, and R² scores.
    """
    # Get the initial guess with the ML model
    (
        mode_values_guess,
        geometric_standard_deviation_guess,
        number_of_particles_guess,
    ) = generate_and_train_2mode_sizer.looped_lognormal_2mode_ml_guess(
        logspace_x=radius_m,
        concentration_pdf=concentration_m3_pdf,
    )

    # Get the optimized values
    (
        mode_values_optimized,
        gsd_optimized,
        number_of_particles_optimized,
        r2_optimized,
    ) = generate_and_train_2mode_sizer.looped_optimize_lognormal_2mode(
        mode_guess=mode_values_guess,
        geometric_standard_deviation_guess=geometric_standard_deviation_guess,
        number_of_particles_in_mode_guess=number_of_particles_guess,
        logspace_x=radius_m,
        concentration_pdf=concentration_m3_pdf,
    )

    # Create and populate the Stream object
    fitted_stream = Stream()
    fitted_stream.time = experiment_time
    fitted_stream.header = [
        "ML_Mode_1",
        "ML_Mode_2",
        "ML_GSD_1",
        "ML_GSD_2",
        "ML_N_1",
        "ML_N_2",
        "Opt_Mode_1",
        "Opt_Mode_2",
        "Opt_GSD_1",
        "Opt_GSD_2",
        "Opt_N_1",
        "Opt_N_2",
        "R2",
    ]
    fitted_stream.data = np.array(
        [
            mode_values_guess[:, 0],
            mode_values_guess[:, 1],
            geometric_standard_deviation_guess[:, 0],
            geometric_standard_deviation_guess[:, 1],
            number_of_particles_guess[:, 0],
            number_of_particles_guess[:, 1],
            mode_values_optimized[:, 0],
            mode_values_optimized[:, 1],
            gsd_optimized[:, 0],
            gsd_optimized[:, 1],
            number_of_particles_optimized[:, 0],
            number_of_particles_optimized[:, 1],
            r2_optimized,
        ]
    ).T  # Transpose to match the shape expected by the Stream

    return fitted_stream


def create_lognormal_2mode_from_fit(
    parameters_stream: Stream,
    radius_min: float = 1e-9,
    radius_max: float = 1e-6,
    num_radius_bins: int = 250,
) -> Tuple[Stream, NDArray[np.float64]]:
    """
    Create a fitted PMF stream and concentration matrix based on
    optimized parameters.

    Arguments:
        radius_min: Log10 of the minimum radius value in meters (default: -9).
        radius_max: Log10 of the maximum radius value in meters (default: -6).
        num_radius_bins: Number of radius bins to create between radius_min and radius_max.

    Returns:
        fitted_pmf_stream: A Stream object containing the time and fitted concentration PMF data.
        concentration_m3_pmf_fits: A numpy array with the fitted concentration PMF values.
    """
    # Define the radius values
    radius_m_values = np.logspace(
        start=np.log10(radius_min),
        stop=np.log10(radius_max),
        num=num_radius_bins,
        dtype=np.float64,
        )

    # Initialize the concentration matrix
    concentration_m3_pmf_fits = np.zeros(
        (len(parameters_stream.time), len(radius_m_values))
    )
    mode_1 = parameters_stream["Opt_Mode_1"]  # Opt_Mode_1
    mode_2 = parameters_stream["Opt_Mode_2"]  # Opt_Mode_2
    gsd_1 = parameters_stream["Opt_GSD_1"]  # Opt_GSD_1
    gsd_2 = parameters_stream["Opt_GSD_2"]  # Opt_GSD_2
    n_1 = parameters_stream["Opt_N_1"]  # Opt_N_1
    n_2 = parameters_stream["Opt_N_2"]  # Opt_N_2

    # Calculate the fitted PMF for each set of optimized parameters
    for i in range(len(mode_1)):
        concentration_m3_pmf_fits[i] = lognormal_pmf_distribution(
            x_values=radius_m_values,
            mode=np.array([mode_1[i], mode_2[i]]),
            geometric_standard_deviation=np.array([gsd_1[i], gsd_2[i]]),
            number_of_particles=np.array([n_1[i], n_2[i]]),
        )

    # Create and populate the Stream object
    fitted_pmf_stream = Stream()
    fitted_pmf_stream.time = parameters_stream.time
    fitted_pmf_stream.header = np.array(radius_m_values, dtype=str).tolist()
    fitted_pmf_stream.data = concentration_m3_pmf_fits

    return fitted_pmf_stream, concentration_m3_pmf_fits


def time_derivative_of_pmf_fits(
    pmf_fitted_stream: Stream,
    liner_slope_window_size: int = 12
) -> Stream:
    """
    Calculate the rate of change of the concentration PMF over time and
    return a new stream.

    Arguments:
        pmf_fitted_stream: Stream object containing the fitted concentration
            PMF data.
        window_size: Size of the time window for fitting the slope.

    Returns:
        rate_of_change_stream: Stream object containing the rate of
            change of the concentration PMF.
    """
    # Extract necessary data from the input stream
    concentration_m3_pmf_fits = pmf_fitted_stream.data
    experiment_time_seconds = time_manage.relative_time(
        epoch_array=pmf_fitted_stream.time,
        units="sec",
    )

    n_rows = concentration_m3_pmf_fits.shape[0]
    dC_dt_smooth = np.zeros_like(concentration_m3_pmf_fits)
    half_window = liner_slope_window_size // 2

    # Iterate over each time point to fit the slope
    for i in range(n_rows):
        if i < half_window:  # Beginning edge case
            start_index = 0
            end_index = i + half_window + 1
        elif i > n_rows - half_window - 1:  # Ending edge case
            start_index = i - half_window
            end_index = n_rows
        else:  # General case
            start_index = i - half_window
            end_index = i + half_window + 1

        # Fit a linear model for each bin size over the current time window
        model = LinearRegression()
        model.fit(
            experiment_time_seconds[start_index:end_index].reshape(-1, 1),
            concentration_m3_pmf_fits[start_index:end_index, :],
        )

        # Store the slope (rate of change)
        dC_dt_smooth[i, :] = model.coef_.flatten()

    # Create a new stream for the rate of change
    pmf_derivative = Stream()
    pmf_derivative.time = pmf_fitted_stream.time
    pmf_derivative.header = pmf_fitted_stream.header
    pmf_derivative.data = dC_dt_smooth

    return pmf_derivative


# disable=too-many-arguments
def calculate_pmf_rates(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    alpha_collision_efficiency: float = 1,
    volume: float = 1,  # m^3
    input_flow_rate: float = 0.16e-6,  # m^3/s
    wall_eddy_diffusivity: float = 0.1,
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),  # m
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Calculate the coagulation, dilution, and wall loss rates,
    and return the net rate.

    Arguments:
        radius_bins: Array of particle radii.
        concentration_pmf: Array of particle concentration
            probability mass function.
        temperature: Temperature in Kelvin.
        pressure: Pressure in Pascals.
        particle_density: Density of the particles in kg/m^3.
        alpha_collision_efficiency: Collision efficiency factor.
        volume: Volume of the chamber in m^3.
        input_flow_rate: Input flow rate in m^3/s.
        wall_eddy_diffusivity: Eddy diffusivity for wall loss in m^2/s.
        chamber_dimensions: Dimensions of the chamber
            (length, width, height) in meters.

    Returns:
        coagulation_loss: Loss rate due to coagulation.
        coagulation_gain: Gain rate due to coagulation.
        dilution_loss: Loss rate due to dilution.
        wall_loss_rate: Loss rate due to wall deposition.
        net_rate: Net rate considering all effects.
    """
    # Mass of the particles in kg
    mass_particle = 4 / 3 * np.pi * radius_bins**3 * particle_density

    # Coagulation kernel
    kernel = coagulation.brownian_coagulation_kernel_via_system_state(
        radius_particle=radius_bins,
        mass_particle=mass_particle,
        temperature=temperature,
        pressure=pressure,
        alpha_collision_efficiency=alpha_collision_efficiency,
    )

    # Coagulation loss and gain
    coagulation_loss = coagulation.discrete_loss(
        concentration=concentration_pmf,
        kernel=kernel,
    )
    coagulation_gain = coagulation.discrete_gain(
        radius=radius_bins,
        concentration=concentration_pmf,
        kernel=kernel,
    )
    coagulation_net = coagulation_gain - coagulation_loss

    # Dilution loss rate
    dilution_coefficient = dilution.volume_dilution_coefficient(
        volume=volume, input_flow_rate=input_flow_rate
    )
    dilution_loss = dilution.dilution_rate(
        coefficient=dilution_coefficient,
        concentration=concentration_pmf,
    )

    # Wall loss rate
    wall_loss_rate = wall_loss.rectangle_wall_loss_rate(
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        particle_radius=radius_bins,
        particle_density=particle_density,
        particle_concentration=concentration_pmf,
        temperature=temperature,
        pressure=pressure,
        chamber_dimensions=chamber_dimensions,
    )

    # Net rate considering coagulation, dilution, and wall loss
    total_rate = coagulation_net + dilution_loss + wall_loss_rate

    return (
        coagulation_loss,
        coagulation_gain,
        dilution_loss,
        wall_loss_rate,
        total_rate,
    )


def coagulation_rates_cost_function(
    parameters: NDArray[np.float64],
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    dN_dt_concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    volume: float = 1,  # m^3
    input_flow_rate: float = 0.16e-6,  # m^3/s
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),  # m
) -> float:
    """Cost function for the optimization of the eddy diffusivity
    and alpha collision efficiency."""

    # Unpack the parameters
    wall_eddy_diffusivity = parameters[0]
    alpha_collision_efficiency = parameters[1]

    # Calculate the rates
    _, _, _, _, net_rate = calculate_pmf_rates(
        radius_bins=radius_bins,
        concentration_pmf=concentration_pmf,
        temperature=temperature,
        pressure=pressure,
        particle_density=particle_density,
        alpha_collision_efficiency=alpha_collision_efficiency,
        volume=volume,
        input_flow_rate=input_flow_rate,
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        chamber_dimensions=chamber_dimensions,
    )

    # Calculate the cost
    number_cost = mean_squared_error(dN_dt_concentration_pmf, net_rate)

    # total_volume comparison
    total_volume_cost = np.power(
        net_rate.sum() - dN_dt_concentration_pmf.sum(),
        2,
        dtype=np.float64,
    )

    if np.isnan(number_cost):
        return 1e34

    return number_cost + total_volume_cost


def optimize_chamber_parameters(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    dN_dt_concentration_pmf: NDArray[np.float64],
    guess_eddy_diffusivity: float = 0.1,
    guess_alpha_collision_efficiency: float = 0.5,
    bounds_eddy_diffusivity: Tuple[float, float] = (1e-6, 20),
    bounds_alpha_collision_efficiency: Tuple[float, float] = (1e-2, 2),
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    volume: float = 1,
    input_flow_rate_m3_sec: float = 1e-6,
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),
    minimize_method: str = "L-BFGS-B",
) -> Tuple[float, float]:
    """
    Optimize the eddy diffusivity and alpha collision efficiency parameters.

    Returns:
        wall_eddy_diffusivity_optimized: Optimized wall eddy diffusivity.
        alpha_collision_efficiency_optimized: Optimized alpha collision efficiency.
    """
    # Initial guess
    initial_guess = np.array(
        [guess_eddy_diffusivity, guess_alpha_collision_efficiency]
    )
    # Bounds
    bounds = [bounds_eddy_diffusivity, bounds_alpha_collision_efficiency]
    # Partial evaluation of the cost function
    partial_cost_function = partial(
        coagulation_rates_cost_function,
        radius_bins=radius_bins,
        concentration_pmf=concentration_pmf,
        dN_dt_concentration_pmf=dN_dt_concentration_pmf,
        temperature=temperature,
        pressure=pressure,
        particle_density=particle_density,
        volume=volume,
        input_flow_rate=input_flow_rate_m3_sec,
        chamber_dimensions=chamber_dimensions,
    )

    # Optimize the parameters
    result = minimize(
        fun=partial_cost_function,
        x0=initial_guess,
        method=minimize_method,
        bounds=bounds,
    )

    # Extract the optimized values
    wall_eddy_diffusivity_optimized = result.x[0]
    alpha_collision_efficiency_optimized = result.x[1]

    return (
        wall_eddy_diffusivity_optimized,
        alpha_collision_efficiency_optimized,
    )


def calculate_optimized_rates(
    radius_bins: np.ndarray,
    concentration_pmf: np.ndarray,
    wall_eddy_diffusivity: float,
    alpha_collision_efficiency: float,
    temperature: float = 293.15,
    pressure: float = 78000,
    particle_density: float = 1600,
    volume: float = 0.9,
    input_flow_rate: float = 1.2 * convert_units("L/min", "m^3/s"),
    chamber_dimensions: Tuple[float, float, float] = (0.739, 0.739, 1.663),
    dN_dt_concentration_pmf: np.ndarray = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate the coagulation rates using the optimized parameters and return the rates and R2 score.

    Returns:
        coagulation_loss: Loss rate due to coagulation.
        coagulation_gain: Gain rate due to coagulation.
        dilution_loss: Loss rate due to dilution.
        wall_loss_rate: Loss rate due to wall deposition.
        net_rate: Net rate considering all effects.
        r2_value: R2 score between the net rate and the observed rate.
    """
    # Calculate the rates
    (
        coagulation_loss,
        coagulation_gain,
        dilution_loss,
        wall_loss_rate,
        net_rate,
    ) = calculate_pmf_rates(
        radius_bins=radius_bins,
        concentration_pmf=concentration_pmf,
        temperature=temperature,
        pressure=pressure,
        particle_density=particle_density,
        alpha_collision_efficiency=alpha_collision_efficiency,
        volume=volume,
        input_flow_rate=input_flow_rate,
        wall_eddy_diffusivity=wall_eddy_diffusivity,
        chamber_dimensions=chamber_dimensions,
    )

    coagulation_net = coagulation_gain - coagulation_loss

    r2_value = (
        r2_score(dN_dt_concentration_pmf, net_rate)
        if dN_dt_concentration_pmf is not None
        else None
    )

    return (
        coagulation_loss,
        coagulation_gain,
        dilution_loss,
        wall_loss_rate,
        coagulation_net,
        r2_value,
    )
