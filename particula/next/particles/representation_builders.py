"""
This module contains the builders for the particle representation classes.

Classes:
- MassParticleRepresentationBuilder: Builder class for
    HistogramParticleRepresentation objects.
- PDFParticleRepresentationBuilder: Builder class for PDFParticleRepresentation
    objects.
- DiscreteParticleRepresentationBuilder: Builder class for
    DiscreteParticleRepresentation objects.
"""

from typing import Optional
import logging
from numpy.typing import NDArray
import numpy as np

from particula.util.input_handling import convert_units

from particula.next.abc_builder import (
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderRadiusMixin,
    BuilderConcentrationMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderChargeMixin,
)
from particula.next.particles.distribution_strategies import (
    RadiiBasedMovingBin,
)
from particula.next.particles.activity_strategies import (
    IdealActivityMass,
)
from particula.next.particles.surface_strategies import (
    SurfaceStrategyVolume,
)
from particula.next.particles.representation import ParticleRepresentation
from particula.next.particles.properties.lognormal_size_distribution import (
    lognormal_pdf_distribution,
)

logger = logging.getLogger("particula")


class MassParticleRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderMassMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):  # pylint: disable=too-many-ancestors
    """General ParticleRepresentation objects with mass-based bins.

    Methods:
        set_distribution_strategy(strategy): Set the DistributionStrategy.
        set_activity_strategy(strategy): Set the ActivityStrategy.
        set_surface_strategy(strategy): Set the SurfaceStrategy.
        set_mass(mass, mass_units): Set the mass of the particles. Default
            units are 'kg'.
        set_density(density, density_units): Set the density of the particles.
            Default units are 'kg/m**3'.
        set_concentration(concentration, concentration_units): Set the
            concentration of the particles. Default units are '/m**3'.
        set_charge(charge, charge_units): Set the number of charges.
    """

    def __init__(self):
        required_parameters = [
            "distribution_strategy",
            "activity_strategy",
            "surface_strategy",
            "mass",
            "density",
            "concentration",
            "charge",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderMassMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="/m**3")
        BuilderChargeMixin.__init__(self)

    def build(self) -> ParticleRepresentation:
        """Validate and return the ParticleRepresentation object.

        Returns:
            The validated ParticleRepresentation object.
        """
        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=self.mass,  # type: ignore
            density=self.density,  # type: ignore
            concentration=self.concentration,  # type: ignore
            charge=self.charge,  # type: ignore
        )


class RadiusParticleRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):  # pylint: disable=too-many-ancestors
    """General ParticleRepresentation objects with radius-based bins.

    Methods:
        set_distribution_strategy(strategy): Set the DistributionStrategy.
        set_activity_strategy(strategy): Set the ActivityStrategy.
        set_surface_strategy(strategy): Set the SurfaceStrategy.
        set_radius(radius, radius_units): Set the radius of the particles.
            Default units are 'm'.
        set_density(density, density_units): Set the density of the particles.
            Default units are 'kg/m**3'.
        set_concentration(concentration, concentration_units): Set the
            concentration of the particles. Default units are '/m**3'.
        set_charge(charge, charge_units): Set the number of charges.
    """

    def __init__(self):
        required_parameters = [
            "distribution_strategy",
            "activity_strategy",
            "surface_strategy",
            "radius",
            "density",
            "concentration",
            "charge",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderRadiusMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="/m**3")
        BuilderChargeMixin.__init__(self)

    def build(self) -> ParticleRepresentation:
        """Validate and return the ParticleRepresentation object.

        Returns:
            The validated ParticleRepresentation object.
        """
        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=self.radius,  # type: ignore
            density=self.density,  # type: ignore
            concentration=self.concentration,  # type: ignore
            charge=self.charge,  # type: ignore
        )


class LimitedRadiusParticleBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):  # pylint: disable=too-many-ancestors
    """General ParticleRepresentation objects with radius-based bins.

    Methods:
        set_distribution_strategy(strategy): Set the DistributionStrategy.
        set_activity_strategy(strategy): Set the ActivityStrategy.
        set_surface_strategy(strategy): Set the SurfaceStrategy.
        set_modes(radius_limits): Set the limits for the mode.
    """

    def __init__(self):
        required_parameters = [
            "distribution_strategy",
            "activity_strategy",
            "surface_strategy",
            "radius",
            "density",
            "concentration",
            "charge",
        ]
        BuilderABC.__init__(self, required_parameters)
        BuilderDistributionStrategyMixin.__init__(self)
        BuilderActivityStrategyMixin.__init__(self)
        BuilderSurfaceStrategyMixin.__init__(self)
        BuilderRadiusMixin.__init__(self)
        BuilderDensityMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="/m**3")
        BuilderChargeMixin.__init__(self)
        self.default_parameters = {
            "distribution_strategy": RadiiBasedMovingBin(),
            "activity_strategy": IdealActivityMass(),
            "surface_strategy": SurfaceStrategyVolume(
                surface_tension=0.072, density=1000
            ),
            "mode": np.array([100e-9, 1e-6]),
            "geometric_standard_deviation": np.array([1.2, 1.4]),
            "number_concentration": np.array([1e4, 1e3]),
            "radius_bins": np.logspace(-9, -4, 250),
            "density": 1000,
            "charge": 0,
        }

    def set_mode(
        self,
        mode: NDArray[np.float_],
        mode_units: str = "m",
    ):
        """Set the modes for distribution

        Args:
            modes: The modes for the radius.
            modes_units: The units for the modes.
        """
        if np.any(mode < 0):
            message = "The mode must be positive."
            logger.error(message)
            raise ValueError(message)
        self.default_parameters["mode"] = mode * convert_units(mode_units, "m")

    def set_geometric_standard_deviation(
        self,
        geometric_standard_deviation: NDArray[np.float_],
        geometric_standard_deviation_units: Optional[str] = None,
    ):
        """Set the geometric standard deviation for the distribution

        Args:
            geometric_standard_deviation: The geometric standard deviation for
            the radius.
        """
        if np.any(geometric_standard_deviation < 0):
            message = "The geometric standard deviation must be positive."
            logger.error(message)
            raise ValueError(message)
        if geometric_standard_deviation_units is not None:
            logger.warning("Ignoring units for surface strategy parameter.")
        self.default_parameters["geometric_standard_deviation"] = (
            geometric_standard_deviation
        )

    def set_number_concentration(
        self,
        number_concentration: NDArray[np.float_],
        number_concentration_units: str = "/m**3",
    ):
        """Set the number concentration for the distribution

        Args:
            number_concentration: The number concentration for the radius.
        """
        if np.any(number_concentration < 0):
            message = "The number concentration must be positive."
            logger.error(message)
            raise ValueError(message)
        self.default_parameters["number_concentration"] = (
            number_concentration
            * convert_units(number_concentration_units, "/m**3")
        )

    def set_radius_bins(
        self,
        radius_bins: NDArray[np.float_],
        radius_bins_units: str = "m",
    ):
        """Set the radius bins for the distribution

        Args:
            radius_bins: The radius bins for the distribution.
        """
        if np.any(radius_bins < 0):
            message = "The radius bins must be positive."
            logger.error(message)
            raise ValueError(message)
        self.default_parameters["radius_bins"] = radius_bins * convert_units(
            radius_bins_units, "m"
        )

    def build(self) -> ParticleRepresentation:
        """Validate and return the ParticleRepresentation object.

        This will build a distribution of particles with a lognormal size
        distribution, before returning the ParticleRepresentation object.

        Returns:
            The validated ParticleRepresentation object.
        """
        self.default_parameters["radius"] = self.default_parameters[
            "radius_bins"
        ]
        number_concentration = lognormal_pdf_distribution(
            x_values=self.default_parameters["radius_bins"],
            mode=self.default_parameters["mode"],
            geometric_standard_deviation=self.default_parameters[
                "geometric_standard_deviation"
            ],
            number_of_particles=self.default_parameters[
                "number_concentration"
            ],
        )
        self.default_parameters["concentration"] = number_concentration
        self.default_parameters.pop("mode")
        self.default_parameters.pop("geometric_standard_deviation")
        self.default_parameters.pop("number_concentration")
        self.default_parameters.pop("radius_bins")

        self.set_parameters(self.default_parameters)
        self.pre_build_check()
        return ParticleRepresentation(
            strategy=self.distribution_strategy,  # type: ignore
            activity=self.activity_strategy,  # type: ignore
            surface=self.surface_strategy,  # type: ignore
            distribution=self.radius,  # type: ignore
            density=self.density,  # type: ignore
            concentration=self.concentration,  # type: ignore
            charge=self.charge,  # type: ignore
        )
