"""
Fit lognormal PDF, and test using sklearn for initial guess
"""

from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt

from scipy.optimize import minimize


from particula.next.particles.properties import lognormal_pdf_distribution

# %%

x_values = np.logspace(2, 4.5, 500)

guess_mode = np.array([100, 5000])
guess_geomertic_standard_deviation = np.array([1.4, 1.8])
guess_number_of_particles = np.array([1000, 5000])

predicted_pdf = lognormal_pdf_distribution(
    x_values=x_values,
    mode=guess_mode,
    geometric_standard_deviation=guess_geomertic_standard_deviation,
    number_of_particles=guess_number_of_particles,
)


# %% plot

fig, ax = plt.subplots()
ax.plot(x_values, predicted_pdf)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle Size (nm)")
ax.set_ylabel("Probability Density Function")
plt.show()
# %%