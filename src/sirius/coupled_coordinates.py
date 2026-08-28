from __future__ import annotations

from typing import Callable

from sirius.cooler_electrodes import (
    cooler_end_pair_from_common_difference,
)
from sirius.guidefield_model import (
    guidefield_pair_from_difference,
)
from sirius.parameters import PARAMETERS
from sirius.state import MachineState


GF1 = "guidefield1_voltage_v"
GF2 = "guidefield2_voltage_v"

HV1 = "deceleration_voltage_v"
HV4 = "acceleration_voltage_v"


def _command(
    state: MachineState,
    parameter_name: str,
) -> float:
    if parameter_name not in state.parameters:
        raise ValueError(
            f"State is missing {parameter_name}"
        )

    return float(
        state.parameters[
            parameter_name
        ]
    )


# ---------------------------------------------------------------------
# Guidefield coordinates
# ---------------------------------------------------------------------

def guidefield_difference_command(
    state: MachineState,
) -> float:
    return (
        _command(
            state,
            GF1,
        )
        - _command(
            state,
            GF2,
        )
    )


def guidefield_common_command(
    state: MachineState,
) -> float:
    return (
        _command(
            state,
            GF1,
        )
        + _command(
            state,
            GF2,
        )
    ) / 2.0


def guidefield_difference_builder(
    state: MachineState,
    difference_v: float,
) -> dict[str, float]:
    """
    Change GF1-GF2 while preserving the current command common mode.
    """

    pair = (
        guidefield_pair_from_difference(
            difference_v,
            common_mode_v=(
                guidefield_common_command(
                    state
                )
            ),
        )
    )

    return {
        GF1: (
            pair.gf1_voltage_v
        ),
        GF2: (
            pair.gf2_voltage_v
        ),
    }


def guidefield_common_builder(
    state: MachineState,
    common_mode_v: float,
) -> dict[str, float]:
    """
    Change guidefield common mode while preserving the current
    command-space difference.
    """

    pair = (
        guidefield_pair_from_difference(
            guidefield_difference_command(
                state
            ),
            common_mode_v=(
                common_mode_v
            ),
        )
    )

    return {
        GF1: (
            pair.gf1_voltage_v
        ),
        GF2: (
            pair.gf2_voltage_v
        ),
    }


def guidefield_difference_bounds(
    state: MachineState,
) -> tuple[
    float,
    float,
]:
    """
    Feasible delta=GF1-GF2 interval at the current common mode.
    """

    common = (
        guidefield_common_command(
            state
        )
    )

    gf1 = PARAMETERS[
        GF1
    ]

    gf2 = PARAMETERS[
        GF2
    ]

    minimum = max(
        2.0
        * (
            gf1.minimum
            - common
        ),
        2.0
        * (
            common
            - gf2.maximum
        ),
    )

    maximum = min(
        2.0
        * (
            gf1.maximum
            - common
        ),
        2.0
        * (
            common
            - gf2.minimum
        ),
    )

    if maximum <= minimum:
        raise ValueError(
            "No feasible guidefield-difference interval"
        )

    return (
        float(minimum),
        float(maximum),
    )


def guidefield_common_bounds(
    state: MachineState,
) -> tuple[
    float,
    float,
]:
    """
    Feasible common-mode interval at the current GF1-GF2 difference.
    """

    difference = (
        guidefield_difference_command(
            state
        )
    )

    gf1 = PARAMETERS[
        GF1
    ]

    gf2 = PARAMETERS[
        GF2
    ]

    minimum = max(
        gf1.minimum
        - difference / 2.0,
        gf2.minimum
        + difference / 2.0,
    )

    maximum = min(
        gf1.maximum
        - difference / 2.0,
        gf2.maximum
        + difference / 2.0,
    )

    if maximum <= minimum:
        raise ValueError(
            "No feasible guidefield common-mode interval"
        )

    return (
        float(minimum),
        float(maximum),
    )


# ---------------------------------------------------------------------
# Cooler entrance / exit coordinates
# ---------------------------------------------------------------------

def end_electrode_difference_command(
    state: MachineState,
) -> float:
    return (
        _command(
            state,
            HV1,
        )
        - _command(
            state,
            HV4,
        )
    )


def end_electrode_common_command(
    state: MachineState,
) -> float:
    return (
        _command(
            state,
            HV1,
        )
        + _command(
            state,
            HV4,
        )
    ) / 2.0


def end_electrode_difference_builder(
    state: MachineState,
    difference_v: float,
) -> dict[str, float]:
    pair = (
        cooler_end_pair_from_common_difference(
            common_bias_v=(
                end_electrode_common_command(
                    state
                )
            ),
            difference_v=(
                difference_v
            ),
        )
    )

    return {
        HV1: (
            pair.entrance_voltage_v
        ),
        HV4: (
            pair.exit_voltage_v
        ),
    }


def end_electrode_common_builder(
    state: MachineState,
    common_bias_v: float,
) -> dict[str, float]:
    pair = (
        cooler_end_pair_from_common_difference(
            common_bias_v=(
                common_bias_v
            ),
            difference_v=(
                end_electrode_difference_command(
                    state
                )
            ),
        )
    )

    return {
        HV1: (
            pair.entrance_voltage_v
        ),
        HV4: (
            pair.exit_voltage_v
        ),
    }


def _symmetric_pair_difference_bounds(
    state: MachineState,
    first_parameter: str,
    second_parameter: str,
    common_reader: Callable[
        [MachineState],
        float,
    ],
) -> tuple[
    float,
    float,
]:
    common = common_reader(
        state
    )

    first = PARAMETERS[
        first_parameter
    ]

    second = PARAMETERS[
        second_parameter
    ]

    minimum = max(
        2.0
        * (
            first.minimum
            - common
        ),
        2.0
        * (
            common
            - second.maximum
        ),
    )

    maximum = min(
        2.0
        * (
            first.maximum
            - common
        ),
        2.0
        * (
            common
            - second.minimum
        ),
    )

    if maximum <= minimum:
        raise ValueError(
            "No feasible differential-coordinate interval"
        )

    return (
        float(minimum),
        float(maximum),
    )


def _symmetric_pair_common_bounds(
    state: MachineState,
    first_parameter: str,
    second_parameter: str,
    difference_reader: Callable[
        [MachineState],
        float,
    ],
) -> tuple[
    float,
    float,
]:
    difference = (
        difference_reader(
            state
        )
    )

    first = PARAMETERS[
        first_parameter
    ]

    second = PARAMETERS[
        second_parameter
    ]

    minimum = max(
        first.minimum
        - difference / 2.0,
        second.minimum
        + difference / 2.0,
    )

    maximum = min(
        first.maximum
        - difference / 2.0,
        second.maximum
        + difference / 2.0,
    )

    if maximum <= minimum:
        raise ValueError(
            "No feasible common-mode interval"
        )

    return (
        float(minimum),
        float(maximum),
    )


def end_electrode_difference_bounds(
    state: MachineState,
) -> tuple[
    float,
    float,
]:
    return (
        _symmetric_pair_difference_bounds(
            state,
            HV1,
            HV4,
            end_electrode_common_command,
        )
    )


def end_electrode_common_bounds(
    state: MachineState,
) -> tuple[
    float,
    float,
]:
    return (
        _symmetric_pair_common_bounds(
            state,
            HV1,
            HV4,
            end_electrode_difference_command,
        )
    )