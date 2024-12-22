"""
This module contains functions to calculate the phase separation of organic
compounds in water. The functions are based on the BAT model by Gorkowski
et al. (2019).

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Union, Optional, List
import numpy as np
from numpy.typing import NDArray
from particula.activity.bat_coefficents import INTERPOLATE_WATER_FIT, LOWEST_ORGANIC_MOLE_FRACTION
from particula.activity.activity_coefficients import activity_coefficients
from particula.activity.species_density import organic_density_estimate
from particula.activity import phase_separation

from particula.util.machine_limit import safe_exp, safe_log

MIN_SPREAD_IN_AW = 10**-6
Q_ALPHA_AT_1PHASE_AW = 0.99


def organic_water_single_phase(
    molar_mass_ratio: Union[int, float, list, np.ndarray]
) -> np.ndarray:
    """
    Convert the given molar mass ratio (MW water / MW organic) to a
    and oxygen2carbon value were above is a single phase with water and below
    phase separation is possible.

    Args:
        - molar_mass_ratio: The molar mass ratio with respect to water.

    Returns:
        - The single phase cross point.

    References:
        - Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
          Relative-humidity-dependent organic aerosol thermodynamics
          Via an efficient reduced-complexity model.
          Atmospheric Chemistry and Physics
          https://doi.org/10.5194/acp-19-13383-2019
    """
    # check inputs
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)

    return (
        0.205
        / (1 + safe_exp(26.6 * (molar_mass_ratio - 0.12))) ** 0.843
        + 0.225
    )


# pylint: disable=too-many-locals
def find_phase_sep_index(
        activity_data: ArrayLike
) -> dict:
    """
    This function finds phase separation using activity>1 and
    inflections in the activity curve data.
    In physical systems activity can not be above one and
    curve should be monotonic. Or else there will be phase separation.

    Args:
    - activity_data: A array of activity data.

    Returns:
    dict: A dictionary containing the following keys:
        - 'phase_sep_activity': Phase separation via activity
            (1 if there is phase separation, 0 otherwise)
        - 'phase_sep_curve': Phase separation via activity curvature
            (1 if there is phase separation, 0 otherwise)
        - 'index_phase_sep_starts': Index where phase separation starts
        - 'index_phase_sep_end': Index where phase separation ends
    """
    # check inputs
    activity_data = np.asarray(activity_data, dtype=np.float64)

    # Compute difference between consecutive elements in the array
    activity_diff = np.diff(activity_data)
    data_length = len(activity_data)

    # Check if the data length is more than 3
    if data_length > 3:
        min_value = np.min(activity_diff)
        max_value = np.max(activity_diff)

        # Check if the min and max differences have the same sign
        if np.sign(min_value) == np.sign(max_value):
            # If so, no phase separation via activity curvature
            phase_sep_curve = 0
            # find the index where the activity is closest to 1
            index_phase_sep_starts = np.argmin(
                np.abs(activity_data - 1))
            index_phase_sep_end = index_phase_sep_starts
        else:
            # If signs differ, phase separation via activity curvature occurs
            phase_sep_curve = 1
            # Find indices where the sign of the second derivative changes
            sign_changes = np.diff(np.sign(activity_diff))

            # The first index where a sign change occurs
            inflection_index = np.where(sign_changes)[0]  # all indices
            index_start = inflection_index[0] \
                if len(inflection_index) > 0 else data_length
            # The last index where a sign change occurs
            back_index = inflection_index[-1] \
                if len(inflection_index) > 0 else data_length

            # Check if first section of activity data is greater than 1
            if np.any(activity_data[:index_start] > 1):
                index_phase_sep_starts = np.argmin(
                    np.abs(activity_data[:index_start] - 1))
            else:
                index_phase_sep_starts = index_start

            # Check if second section of activity data is greater than 1
            if np.any(activity_data[back_index:] > 1):
                index_phase_sep_end = np.argmin(
                    np.abs(activity_data[back_index:] - 1)) + back_index
            else:
                index_phase_sep_end = back_index
    else:
        phase_sep_activity = activity_data
        phase_sep_curve = 0
        index_phase_sep_starts = np.nan
        index_phase_sep_end = np.nan

    # Assign phase separation via activity based on data being greater than 1
    phase_sep_activity = 1 if sum(activity_data > 1) else 0
    return {'phase_sep_activity': phase_sep_activity,
            'phase_sep_curve': phase_sep_curve,
            'index_phase_sep_starts': index_phase_sep_starts,
            'index_phase_sep_end': index_phase_sep_end}


def find_phase_separation(
        activity_water: ArrayLike,
        activity_org: ArrayLike
) -> dict:
    """
    This function checks for phase separation in each activity curve.

    Args:
    - activity_water (np.array): A numpy array of water activity values.
    - activity_org (np.array): A numpy array of organic activity values.

    Returns:
    dict: A dictionary containing the following keys:
        - 'phase_sep_check': An integer indicating whether phase separation
                is present (1) or not (0).
        - 'lower_seperation_index': The index of the lower separation point
                in the activity curve.
        - 'upper_seperation_index': The index of the upper separation point in
                the activity curve.
        - 'matching_upper_seperation_index': The index where the difference
                between activity_water_beta and match_a_w is greater than 0.
        - 'lower_seperation': The value of water activity at the lower
                separation point.
        - 'upper_seperation': The value of water activity at the upper
                separation point.
        - 'matching_upper_seperation': The value of water activity at the
                matching upper separation point.
    """
    # check inputs
    activity_water = np.asarray(activity_water, dtype=np.float64)
    activity_org = np.asarray(activity_org, dtype=np.float64)

    # check for phase separation in each activity curve
    water_sep = find_phase_sep_index(activity_water)
    organic_sep = find_phase_sep_index(activity_org)

    # gather all the indexes into a list for easier access
    indexes = [water_sep['index_phase_sep_starts'],
               water_sep['index_phase_sep_end'],
               organic_sep['index_phase_sep_starts'],
               organic_sep['index_phase_sep_end']]

    # If there is a phase separation curve in the water activity data
    if water_sep['phase_sep_curve'] == 1:
        phase_sep_check = 1

        # Check for the direction of the curve (increasing or decreasing)
        if activity_water[0] < activity_water[-1]:  # increasing a_w with index
            # find the min and max indexes
            lower_seperation_index = min(indexes)
            upper_seperation_index = max(indexes)
            match_a_w = activity_water[upper_seperation_index]

            # start from the lower_seperation_index and find the index where
            # the difference between activity_water and match_a_w changes sign
            match_slice = np.sign(
                match_a_w - activity_water[lower_seperation_index:]
            )
            match_index_prime = np.where(match_slice == -1)
            if len(match_index_prime[0]) == 0:
                match_index_prime = lower_seperation_index
            else:
                match_index_prime = match_index_prime[0][0] + \
                    lower_seperation_index
        else:  # decreasing a_w with index
            # find the min and max indexes
            lower_seperation_index = max(indexes)
            upper_seperation_index = min(indexes)
            match_a_w = activity_water[upper_seperation_index]

            # start from the lower_seperation_index and find the index where
            # the difference between activity_water and match_a_w changes sign
            match_slice = np.sign(
                activity_water[:lower_seperation_index] - match_a_w
            )
            match_index_prime = np.where(match_slice == -1)
            if len(match_index_prime[0]) == 0:
                match_index_prime = lower_seperation_index
            else:
                match_index_prime = match_index_prime[0][0]

    else:
        upper_seperation_index = 2
        lower_seperation_index = 2
        match_index_prime = 2
        phase_sep_check = 0  # no phase seperation

    return {'phase_sep_check': phase_sep_check,
            'lower_seperation_index': lower_seperation_index,
            'upper_seperation_index': upper_seperation_index,
            'matching_upper_seperation_index': match_index_prime,
            'lower_seperation': activity_water[lower_seperation_index],
            'upper_seperation': activity_water[upper_seperation_index],
            'matching_upper_seperation': activity_water[match_index_prime]}


def q_alpha(
        seperation_activity: Union[int, float, np.ndarray],
        activities: ArrayLike,
) -> np.ndarray:
    """
    This function calculates the q_alpha value using a squeezed logistic
        function.

    Args:
        - seperation_activity (np.array): A numpy array of values representing
            the separation activity.
        - activities (np.array): A numpy array of activity values.

    Returns:
        np.array: The q_alpha value.

    Notes:
        - The q_alpha value represents the transfer from
            q_alpha ~0 to q_alpha ~1.
        - The function uses a sigmoid curve parameter to calculate the
            q_alpha value.
    """
    # check inputs
    activities = np.asarray(activities, dtype=np.float64)
    if seperation_activity == 0:
        return np.ones_like(activities)
    seperation_activity = np.asarray(seperation_activity, dtype=np.float64)

    # spread in transfer from 50/50 point
    delta_seperation = 1 - seperation_activity

    # check min value allowed
    above_min_delta_seperation_value = delta_seperation > MIN_SPREAD_IN_AW
    delta_seperation = delta_seperation * above_min_delta_seperation_value + \
        ~above_min_delta_seperation_value * MIN_SPREAD_IN_AW

    # calculate curve parameter of sigmoid
    sigmoid_curve_parameter = safe_log(
        1 / (1 - Q_ALPHA_AT_1PHASE_AW) - 1) / delta_seperation

    # calculate q_alpha return value
    return 1 - 1 / (
        1
        + safe_exp(
            sigmoid_curve_parameter *
            (activities - seperation_activity + delta_seperation)
        )
    )


def biphasic_water_activity_point(
    oxygen2carbon: Union[float, NDArray[np.float64]],
    hydrogen2carbon: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    functional_group: Optional[Union[list[str], str]] = None,
) -> np.ndarray:
    """
    This function computes the biphasic to single phase
    water activity (RH*100).

    Args:
        - oxygen2carbon : The oxygen to carbon ratio.
        - hydrogen2carbon : The hydrogen to carbon ratio.
        - molar_mass_ratio : The molar mass ratio of water to organic
          matter.
        - functional_group : Optional functional group(s) of the organic
          compound, if applicable.

    Returns:
        - The RH cross point array.
    """
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    hydrogen2carbon = np.asarray(hydrogen2carbon, dtype=np.float64)
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    if oxygen2carbon.ndim == 0:
        oxygen2carbon = np.expand_dims(oxygen2carbon, axis=0)
    if hydrogen2carbon.ndim == 0:
        hydrogen2carbon = np.expand_dims(hydrogen2carbon, axis=0)
    if molar_mass_ratio.ndim == 0:
        molar_mass_ratio = np.expand_dims(molar_mass_ratio, axis=0)

    water_activity_cross_point = np.zeros_like(oxygen2carbon)

    interpolate_step_numb = 200
    mole_frac = np.logspace(-6, 0, interpolate_step_numb + 1)

    for i, _ in enumerate(oxygen2carbon):
        density = organic_density_estimate(
            molar_mass_ratio[i],
            oxygen2carbon[i],
            hydrogen2carbon[i],
            mass_ratio_convert=True,
        )
        activities = activity_coefficients(
            molar_mass_ratio=molar_mass_ratio[i],
            organic_mole_fraction=mole_frac,
            oxygen2carbon=oxygen2carbon[i],
            density=density,
            functional_group=functional_group,
        )

        if np.isnan(activities[0]).any():
            raise ValueError("water activity is NaN, check inputs")

        phase_check = phase_separation.find_phase_separation(
            activities[0], activities[1]
        )

        if phase_check["phase_sep_check"] == 1:
            water_activity_cross_point[i] = phase_check["upper_seperation"]
        else:
            water_activity_cross_point[i] = 0

    water_activity_cross_point[water_activity_cross_point < 0] = 0
    water_activity_cross_point[water_activity_cross_point > 1] = 1

    return water_activity_cross_point


def fixed_water_activity(
    water_activity: Union[float, NDArray[np.float64]],
    molar_mass_ratio: Union[float, NDArray[np.float64]],
    oxygen2carbon: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
) -> Tuple[
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
    Union[float, NDArray[np.float64]],
]:
    """
    Calculate the activity coefficients of water and organic matter in
    organic-water mixtures.

    This function assumes a fixed water activity value (e.g., RH = 75%
    corresponds to 0.75 water activity in equilibrium).
    It calculates the activity coefficients for different phases and
    determines phase separations if they occur.

    Args:
        - water_activity : An array of water activity values.
        - molar_mass_ratio : Array of molar mass ratios of the components.
        - oxygen2carbon : Array of oxygen-to-carbon ratios.
        - density : Array of densities of the mixture, in kg/m^3.

    Returns:
        - A tuple containing the activity coefficients for alpha and beta
          phases, and the q_alpha (phase separation) value.
          If no phase separation occurs, the beta phase values are None.
    """
    water_activity = np.asarray(water_activity, dtype=np.float64)
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)

    if water_activity.size > 1 and water_activity[0] > water_activity[-1]:
        water_activity = np.flip(water_activity)
        flip = True
    else:
        flip = False

    organic_mole_fraction_array = np.linspace(
        1,
        LOWEST_ORGANIC_MOLE_FRACTION,
        INTERPOLATE_WATER_FIT,
        dtype=np.float64,
    )

    activities = activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction_array,
        oxygen2carbon=oxygen2carbon,
        density=density,
    )
    phase_check = phase_separation.find_phase_separation(
        activities[0], activities[1]
    )
    activities_water = np.asarray(activities[0], dtype=np.float64)
    if phase_check["phase_sep_check"] == 0:
        alpha_organic_mole_fraction = np.interp(
            xp=activities_water,
            fp=organic_mole_fraction_array,
