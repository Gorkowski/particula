"""a simple, fast, and powerful particle simulator.

particula is a simple, fast, and powerful particle simulator,
or at least two of the three, we hope. It is a simple particle
system that is designed to be easy to use and easy to extend.
The goal is to have a robust aerosol (gas phase + particle phase)
simulation system that can be used to answer scientific questions
that arise for experiments and research discussions.

The main features of particula are:
...

More details to follow.
"""

# pylint: disable=unused-import
# flake8: noqa
# pyright: basic

from importlib import import_module
from typing import TYPE_CHECKING

from particula.logger_setup import setup

__version__ = "0.2.11"

__all__ = [
    "gas",
    "particles",
    "util",
    "dynamics",
    "activity",
    "equilibria",
    "Aerosol",
    "AerosolBuilder",
    "RunnableSequence",
]

if TYPE_CHECKING:  # pragma: no cover - static import for type checkers
    from particula import gas, particles, util, dynamics, activity, equilibria
    from particula.aerosol import Aerosol
    from particula.aerosol_builder import AerosolBuilder
    from particula.runnable import RunnableSequence
else:
    gas = import_module("particula.gas")
    util = import_module("particula.util")
    dynamics = import_module("particula.dynamics")
    activity = import_module("particula.activity")
    equilibria = import_module("particula.equilibria")

    def _lazy_import_particles() -> None:
        global particles  # type: ignore[global-var-not-assigned]
        particles = import_module("particula.particles")

    try:
        _lazy_import_particles()
    except Exception:  # pragma: no cover - log and defer heavy import
        particles = None  # type: ignore[assignment]

    from particula.aerosol import Aerosol
    from particula.aerosol_builder import AerosolBuilder
    from particula.runnable import RunnableSequence

# setup the logger
logger = setup()
# log the version of particula upon loading
logger.info("particula version %s loaded.", __version__)
