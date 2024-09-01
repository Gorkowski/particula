# %% imports
"""
Create NN for lognormal PDF, fit for initial guess
"""
import numpy as np

import matplotlib.pyplot as plt

from particula.next.particles.properties import lognormal_pdf_distribution
from particula.data.process.ml_analysis import generate_and_train_2mode_sizer

# %% Generate data

x_values = np.logspace(1.5, 3, 150)

concentration_pdf = lognormal_pdf_distribution(
    x_values=x_values,
    mode=np.array([80, 150]),
    geometric_standard_deviation=np.array([1.2, 1.3]),
    number_of_particles=np.array([200, 500]),
)
# multiply by noise scale
concentration_pdf = concentration_pdf * np.random.uniform(
    low=0.8, high=1.2, size=concentration_pdf.shape
)

# Get the guess with the ML model
(
    mode_values_guess,
    geometric_standard_deviation_guess,
    number_of_particles_guess,
) = generate_and_train_2mode_sizer.lognormal_2mode_ml_guess(
    logspace_x=x_values,
    concentration_pdf=concentration_pdf,
)

# print the results
print(f"Mode: {mode_values_guess}")
print(f"GSD: {geometric_standard_deviation_guess}")
print(f"Number of particles: {number_of_particles_guess}")

# %% Optimize the lognormal fit
# Optimize the lognormal fit
(
    mode_values_optimized,
    gsd_optimized,
    number_of_particles_optimized,
    r2_optimized,
    optimization_results,
) = generate_and_train_2mode_sizer.optimize_lognormal_2mode(
    mode_guess=mode_values_guess,
    geometric_standard_deviation_guess=geometric_standard_deviation_guess,
    number_of_particles_in_mode_guess=number_of_particles_guess,
    logspace_x=x_values,
    concentration_pdf=concentration_pdf,
)

# Print the results
print(f"Optimized mode values: {mode_values_optimized}")
print(f"Optimized GSD: {gsd_optimized}")
print(f"Optimized number of particles: {number_of_particles_optimized}")
print(f"Optimized R²: {r2_optimized}")
print(f"Best method: {optimization_results['best_method']}")


# %% Plot
concentration_pdf_guess = lognormal_pdf_distribution(
    x_values=x_values,
    mode=mode_values_guess,
    geometric_standard_deviation=geometric_standard_deviation_guess,
    number_of_particles=number_of_particles_guess,
)
concentration_pdf_optimized = lognormal_pdf_distribution(
    x_values=x_values,
    mode=mode_values_optimized,
    geometric_standard_deviation=gsd_optimized,
    number_of_particles=number_of_particles_optimized,
)

fig, ax = plt.subplots()
ax.plot(x_values, concentration_pdf, label="Original")
ax.plot(x_values, concentration_pdf_guess, label="ML Guess")
ax.plot(
    x_values, concentration_pdf_optimized, label="Optimized", linestyle="--"
)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.legend()
plt.show()
# %%
