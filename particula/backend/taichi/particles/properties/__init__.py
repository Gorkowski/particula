from .ti_vapor_correction_module import (
    ti_get_vapor_transition_correction,
    kget_vapor_transition_correction,
    fget_vapor_transition_correction,
)
from .ti_stokes_number_module import (
    ti_get_stokes_number,
    kget_stokes_number,
    fget_stokes_number,
)
from .ti_knudsen_number_module import (
    ti_get_knudsen_number,
    kget_knudsen_number,
    fget_knudsen_number,
)
from .ti_special_functions import (
    ti_get_debye_function,
    kget_debye_function,
    fget_debye_function,
)
from .ti_slip_correction_module import (
    ti_get_cunningham_slip_correction,
    kget_cunningham_slip_correction,
    fget_cunningham_slip_correction,
)
from .ti_settling_velocity import (
    ti_get_particle_settling_velocity,
    kget_particle_settling_velocity,
    fget_particle_settling_velocity,
)
from .ti_reynolds_number_module import (           # NEW – keep style parity
    ti_get_particle_reynolds_number,
    kget_particle_reynolds_number,
    fget_particle_reynolds_number,
)
