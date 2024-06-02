"""
This module contains the builders for the particle representation classes.

Classes:
- HistogramParticleRepresentationBuilder: Builder class for
    HistogramParticleRepresentation objects.
- PDFParticleRepresentationBuilder: Builder class for PDFParticleRepresentation
    objects.
- DiscreteParticleRepresentationBuilder: Builder class for
    DiscreteParticleRepresentation objects.
"""

from typing import Dict, Any, Union
import logging
from numpy.typing import NDArray

from particula.next.abc_builder import (
    BuilderABC,
    BuilderMolarMassMixin,
    BuilderDensityMixin,
    BuilderSurfaceTensionMixin,
)
from particula.next.particles.representation import ParticleRepresentation
