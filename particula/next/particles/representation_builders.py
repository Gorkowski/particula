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

import logging
from typing import Union, Optional

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
from particula.next.particles.representation import ParticleRepresentation
from particula.next.particles.properties.lognormal_size_distribution import (
    lognormal_pdf_distribution,
    lognormal_pmf_distribution,
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
    """Builder class for ParticleRepresentation objects with mass-based bins.

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
    """Builder class for ParticleRepresentation objects with radius-based bins.

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


class LogNormalParticleRepresentationBuilder(
    BuilderABC,
    BuilderDistributionStrategyMixin,
    BuilderActivityStrategyMixin,
    BuilderSurfaceStrategyMixin,
    BuilderRadiusMixin,
    BuilderDensityMixin,
    BuilderConcentrationMixin,
    BuilderChargeMixin,
):  # pylint: disable=too-many-ancestors
    """Builder class for ParticleRepresentation objects with log-normal
    distribution.

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
        BuilderDensityMixin.__init__(self)
        BuilderConcentrationMixin.__init__(self, default_units="/m**3")
        BuilderChargeMixin.__init__(self)

    def set_mean_radius(
        self,
        mean_radius: Union[float, NDArrya[np.float_]],
        mean_radius_units: Optional[str] = "m",
    ):
        """_summary_

        Args:
            mean_radius: Modes of the distribution.
            mean_radius_units: _description_. Defaults to "m".
        """


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