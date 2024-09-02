# %% imports
"""
Create NN for lognormal PDF, fit for initial guess
"""
from functools import partial

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from particula.data.process import chamber_rate_fitting

from particula.next.particles.properties import lognormal_pdf_distribution, lognormal_pmf_distribution
from particula.data.process.ml_analysis import generate_and_train_2mode_sizer
from particula.data.stream import Stream
from particula.data import loader
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from scipy.optimize import minimize  # type: ignore

# all the imports

from particula.data import loader_interface, settings_generator
from particula.data.tests.example_data.get_example_data import get_data_folder
from particula.util.convert import distribution_convert_pdf_pms
from particula.next.dynamics import dilution, wall_loss, coagulation

from particula.data import stream_stats
from particula.util import convert, time_manage
from particula.util.input_handling import convert_units

# set the parent directory of the data folders
path = get_data_folder()
print("Path to data folder:")
print(path.rsplit("particula")[-1])

# %% Laod the data
# load the 1d data
smps_1d_stream_settings = settings_generator.load_settings_for_stream(
    path=path,
    subfolder="chamber_data",
    settings_suffix="_smps_1d",
)
stream_sizer_1d = loader_interface.load_files_interface(
    path=path, settings=smps_1d_stream_settings
)

# load the 2d data
smps_2d_stream_settings = settings_generator.load_settings_for_stream(
    path=path,
    subfolder="chamber_data",
    settings_suffix="_smps_2d",
)
stream_sizer_2d = loader_interface.load_files_interface(
    path=path, settings=smps_2d_stream_settings
)

print(stream_sizer_1d.header)

# 1 convert to dn/dDp
stream_sizer_2d.data = convert.convert_sizer_dn(
    diameter=np.array(stream_sizer_2d.header, dtype=float),
    dn_dlogdp=stream_sizer_2d.data,
)
# Dilution correction
dilution_correction = 2
# scale the concentrations
stream_sizer_2d.data *= dilution_correction
stream_sizer_1d["Total_Conc_(#/cc)"] *= dilution_correction

# select the time window
bad_window_start_epoch = time_manage.time_str_to_epoch(
    time="09-25-2023 19:00:00",
    time_format="%m-%d-%Y %H:%M:%S",
    timezone_identifier="UTC",
)
bad_window_end_epoch = time_manage.time_str_to_epoch(
    time="09-25-2023 19:45:00",
    time_format="%m-%d-%Y %H:%M:%S",
    timezone_identifier="UTC",
)
# remove the bad data
stream_sizer_1d = stream_stats.remove_time_window(
    stream=stream_sizer_1d,
    epoch_start=bad_window_start_epoch,
    epoch_end=bad_window_end_epoch,
)
stream_sizer_2d = stream_stats.remove_time_window(
    stream=stream_sizer_2d,
    epoch_start=bad_window_start_epoch,
    epoch_end=bad_window_end_epoch,
)
# remove the first few bins
stream_sizer_2d.data = stream_sizer_2d.data[:, 20:]
stream_sizer_2d.header = stream_sizer_2d.header[20:]

# crop start
experiment_start_epoch = time_manage.time_str_to_epoch(
    time="09-25-2023 15:25:00",
    time_format="%m-%d-%Y %H:%M:%S",
    timezone_identifier="UTC",
)
# crop the end
experiment_end_epoch = time_manage.time_str_to_epoch(
    time="09-25-2023 17:00:00",
    time_format="%m-%d-%Y %H:%M:%S",
    timezone_identifier="UTC",
) # time="09-26-2023 07:00:00",
# apply the time window
stream_sizer_1d = stream_stats.select_time_window(
    stream=stream_sizer_1d,
    epoch_start=experiment_start_epoch,
    epoch_end=experiment_end_epoch,
)
stream_sizer_2d = stream_stats.select_time_window(
    stream=stream_sizer_2d,
    epoch_start=experiment_start_epoch,
    epoch_end=experiment_end_epoch,
)
# save the cleaned data
loader.save_stream(
    stream=stream_sizer_2d,
    path=path,
    suffix_name="_sizer_2d_cleaned",
    folder="chamber_analysis",
)

experiment_time = time_manage.relative_time(
    epoch_array=stream_sizer_1d.time,
    units="hours",
)
# Plot the 2d data
fig, ax = plt.subplots(1, 1)
plt.contourf(
    experiment_time,
    stream_sizer_2d.header_float,
    stream_sizer_2d.data.T,
    cmap=plt.cm.PuBu_r,
    levels=50,
)
plt.yscale("log")
ax.set_xlabel("Experiment time (hours)")
ax.set_ylabel("Diameter (nm)")
ax.set_title("Concentration vs time")
plt.colorbar(
    label=r"Concentration $\dfrac{1}{cm^3}$", ax=ax
)
plt.show()


# %% convert the distribution base SI units (1/m^3) and m, then to pdf
radius_m = stream_sizer_2d.header_float / 2 * convert_units(old="nm", new="m")
concentration_m3_pmf = stream_sizer_2d.data * convert_units(old="1/cm^3", new="1/m^3")

concentration_m3_pdf = distribution_convert_pdf_pms(
    x_array=radius_m,
    distribution=concentration_m3_pmf,
    to_pdf=True,
)

# Plot the 2d data
fig, ax = plt.subplots(1, 1)
plt.contourf(
    experiment_time,
    radius_m,
    concentration_m3_pdf.T,
    cmap=plt.cm.PuBu_r,
    levels=50,
)
plt.yscale("log")
ax.set_xlabel("Experiment time (hours)")
ax.set_ylabel("Radius (m)")
ax.set_title("Concentration PDF")
plt.colorbar(label=r"Concentration $\dfrac{1}{m^3 \cdot m}$", ax=ax)
plt.show()


# %% fit the lognormal pdf distribution with 2 modes

stream_lognormal_prameters = chamber_rate_fitting.fit_lognormal_2mode_pdf(
    experiment_time=stream_sizer_2d.time,
    radius_m=radius_m,
    concentration_m3_pdf=concentration_m3_pdf,
)

# save the stream
loader.save_stream(
    stream=stream_lognormal_prameters,
    path=path,
    folder="chamber_analysis",
    suffix_name="_optimized_lognormal_2mode",
)
# # save to csv
# loader.save_stream_to_csv(
#     stream=fitted_stream,
#     path=path,
#     folder="chamber_analysis",
#     suffix_name="optimized_lognormal_2mode",
# )

# plot the r2 values
fig, ax = plt.subplots(1, 1)
plt.plot(experiment_time, stream_lognormal_prameters["R2"])
ax.set_xlabel("Time (hours)")
ax.set_ylabel("R2")
ax.set_title("R2 from Lognormal 2 mode fit")
plt.show()


# %% Defined the new scale and create the fitted pmf

# Create the fitted pmf
(
    fitted_pmf_stream, fitted_pmf_concentration
) = chamber_rate_fitting.create_lognormal_2mode_from_fit(
    parameters_stream=stream_lognormal_prameters,
    radius_min=1e-9,
    radius_max=1e-6,
    num_radius_bins=250,
)

# %% Plot the fitted pmf
fig, ax = plt.subplots(1, 1)
plt.contourf(
    experiment_time,
    fitted_pmf_stream.header_float,
    fitted_pmf_stream.data.T,
    cmap=plt.cm.PuBu_r,
    levels=50,
)
plt.yscale("log")
ax.set_xlabel("Experiment time (hours)")
ax.set_ylabel("Radius (m)")
ax.set_title("Fitted Concentration PMF")
plt.colorbar(label=r"Concentration $\dfrac{1}{m^3}$", ax=ax)
plt.show()

# % plot a slice of the data
fig, ax = plt.subplots(1, 1)
index = 100
plt.plot(fitted_pmf_stream.time,
         fitted_pmf_stream[index],
         label=f"Concentration at {fitted_pmf_stream.header_float[index]} m",
         marker="o")
plt.yscale("log")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Concentration (1/m^3)")
ax.set_title("Bin Concentration")
plt.legend()
plt.show()

# %% plot diameter vs concentration
fig, ax = plt.subplots(1, 1)
time_index = 5
ax.plot(fitted_pmf_stream.header_float,
        fitted_pmf_stream.data[time_index, :],
        label=f"Concentration at {experiment_time[time_index]} hours",)
ax.plot(fitted_pmf_stream.header_float,
        fitted_pmf_stream.data[time_index+5, :],
        label=f"Concentration at {experiment_time[time_index+5]} hours",)
ax.set_xscale("log")
ax.set_xlabel("Diameter (m)")
ax.set_ylabel("Concentration (1/m^3)")
ax.set_title("Bin Concentration")
plt.legend(loc='upper left')
plt.show()


# %% Linear window fit

pmf_derivative_stream = chamber_rate_fitting.time_derivative_of_pmf_fits(
    pmf_fitted_stream=fitted_pmf_stream,
    liner_slope_window_size=10,
)


# Plot the 2d data
fig, ax = plt.subplots(1, 1)
plt.contourf(
    experiment_time,
    pmf_derivative_stream.header_float,
    pmf_derivative_stream.data.T,
    cmap=plt.cm.PuBu_r,
    levels=50,
)
plt.yscale("log")
ax.set_xlabel("Experiment time (hours)")
ax.set_ylabel("Radius (m)")
ax.set_title("dN/dt")
plt.colorbar(label=r"Measured Rate $\dfrac{1}{m^3 s}$", ax=ax)
plt.show()

# plot a slice of the data
index = 100
fig, ax = plt.subplots(1, 1)
plt.plot(experiment_time,
         pmf_derivative_stream[index],
         label=f"dN/dt at {pmf_derivative_stream.header_float[index]} m",
         marker="o")
ax.set_xlabel("Time (hours)")
ax.set_ylabel(r"Measured Rate $\dfrac{1}{m^3 s}$")
ax.set_title("Bin dN/dt")
plt.legend()
plt.show()

# plot diameter vs dN/dt
time_index = 5
fig, ax = plt.subplots(1, 1)
ax.plot(pmf_derivative_stream.header_float,
        pmf_derivative_stream.data[time_index, :],
        label=f"dN/dt at {experiment_time[time_index]} hours",
        marker="o")
ax.set_xscale("log")
ax.set_xlabel("Diameter (m)")
ax.set_ylabel(r"Measured Rate $\dfrac{1}{m^3 s}$")
ax.set_title("Bin dN/dt")
plt.legend()
# %% optimize eddy diffusivity and alpha collision efficiency

chamber_properties = chamber_rate_fitting.ChamberParameters()
chamber_properties.temperature = 293.15
chamber_properties.pressure = 78000
chamber_properties.particle_density = 1600
chamber_properties.volume = 0.9
chamber_properties.input_flow_rate_m3_sec = 1.2 * convert_units("L/min", "m^3/s")
chamber_properties.chamber_dimensions = (0.739, 0.739, 1.663)

fit_guess, fit_bounds = chamber_rate_fitting.create_guess_and_bounds(
    guess_eddy_diffusivity=0.1,
    guess_alpha_collision_efficiency=0.05,
    bounds_eddy_diffusivity=(1e-2, 50),
    bounds_alpha_collision_efficiency=(0.01, 1),
)

# %% fit looped

(
    result_stream,
    coagulation_loss_stream,
    coagulation_gain_stream,
    coagulation_net_stream,
    dilution_loss_stream,
    wall_loss_rate_stream,
    total_rate_stream,
) = chamber_rate_fitting.optimize_and_calculate_rates_looped(
    pmf_stream=fitted_pmf_stream,
    pmf_derivative_stream=pmf_derivative_stream,
    chamber_parameters=chamber_properties,
    fit_guess=fit_guess,
    fit_bounds=fit_bounds,
)


# %% plot the optimized rates

fig, ax = plt.subplots(1, 1)
ax.plot(experiment_time, result_stream["r2_value"])
ax.set_xlabel("Time (hours)")
ax.set_ylabel("R2")
ax.set_title("R2 from optimization")
plt.show()

# plot the optimized values
fig, ax = plt.subplots(1, 1)
ax.plot(experiment_time, result_stream["wall_eddy_diffusivity_[1/s]"], label="Wall eddy diffusivity")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Wall eddy diffusivity (1/s)")
ax.set_title("Wall eddy diffusivity")
plt.legend()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(experiment_time, result_stream["alpha_collision_efficiency_[-]"],
        label="Alpha collision efficiency")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Alpha collision efficiency (-)")
ax.set_title("Alpha collision efficiency")
plt.legend()
plt.show()

# plot the rate fractions
fig, ax = plt.subplots(1, 1)
ax.plot(experiment_time, result_stream["coagulation_net_fraction"], label="Coagulation net")
ax.plot(experiment_time, result_stream["dilution_loss_fraction"], label="Dilution loss")
ax.plot(experiment_time, result_stream["wall_loss_rate_fraction"], label="Wall loss")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Rate fraction")
ax.set_title("Rate fractions")
plt.legend()
plt.show()

# %%
# % plot the optimized rates

time_index = 5
# print the optimized values

radius_m_values = fitted_pmf_stream.header_float
fig, ax = plt.subplots(1, 1)
plt.plot(radius_m_values,
         coagulation_net_stream.data[time_index, :],
         label="Coagulation net")
plt.plot(radius_m_values,
         dilution_loss_stream.data[time_index, :],
         , label="Dilution")
plt.plot(radius_m_values,
         wall_loss_rate_stream.data[time_index, :],
         label="Wall loss")
# plt.plot(radius_m_values, -concentration_m3_pmf_fits[11, :], label="Measured value",)
plt.plot(radius_m_values,
         coagulation_net_stream.data[time_index, :], label="Net rate")
plt.plot(
    radius_m_values,
    pmf_derivative_stream.data[time_index, :],
    label="Measured rate",
    linestyle="--",
)
plt.xscale("log")
plt.xlabel("Diameter (m)")
plt.ylabel(r"Rate $\dfrac{1}{m^3 s}$")
plt.title("Rate comparison")
plt.legend()
plt.show()


# %%
