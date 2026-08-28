from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.parameters import PARAMETERS
from sirius.state import MachineState


QPT1_PARAMETER = "quadrupole1_voltage_v"
QPT2_PARAMETER = "quadrupole2_voltage_v"
QPT3_PARAMETER = "quadrupole3_voltage_v"

QPT_PARAMETERS = (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
)


@dataclass(frozen=True)
class QPTPolePairAssignment:
    """
    Assignment of one opposing pole pair to one QPT power supply.

    Each x/y entry represents two opposing physical poles, so the six
    assignments below describe all twelve QPT poles.
    """

    quadrupole: int
    axis: str
    parameter_name: str


QPT_POLE_PAIR_ASSIGNMENTS = (
    QPTPolePairAssignment(
        quadrupole=1,
        axis="x",
        parameter_name=QPT3_PARAMETER,
    ),
    QPTPolePairAssignment(
        quadrupole=1,
        axis="y",
        parameter_name=QPT2_PARAMETER,
    ),
    QPTPolePairAssignment(
        quadrupole=2,
        axis="x",
        parameter_name=QPT2_PARAMETER,
    ),
    QPTPolePairAssignment(
        quadrupole=2,
        axis="y",
        parameter_name=QPT1_PARAMETER,
    ),
    QPTPolePairAssignment(
        quadrupole=3,
        axis="x",
        parameter_name=QPT3_PARAMETER,
    ),
    QPTPolePairAssignment(
        quadrupole=3,
        axis="y",
        parameter_name=QPT2_PARAMETER,
    ),
)


@dataclass(frozen=True)
class QPTVoltageObservation:
    value_v: float
    source: str

    command_v: float
    readback_v: float | None


@dataclass(frozen=True)
class QPTCoordinates:
    """
    Reduced QPT coordinate system.

    V1 = QPT1
    V2 = QPT2
    V3 = QPT3

    common:
        C = V2

    outer strength coordinate:
        O = V2 - V3

    middle strength coordinate:
        M = V2 - V1

    global focus:
        F = (O + M) / 2

    asymmetry / balance:
        A = (M - O) / 2

    Therefore:

        M = F + A
        O = F - A

        V1 = C - F - A
        V2 = C
        V3 = C - F + A
    """

    common_v: float

    global_focus_v: float
    asymmetry_v: float

    outer_strength_v: float
    middle_strength_v: float


@dataclass(frozen=True)
class QPTCommandSet:
    qpt1_voltage_v: float
    qpt2_voltage_v: float
    qpt3_voltage_v: float

    coordinates: QPTCoordinates

    @property
    def parameters(
        self,
    ) -> dict[str, float]:
        return {
            QPT1_PARAMETER: (
                self.qpt1_voltage_v
            ),
            QPT2_PARAMETER: (
                self.qpt2_voltage_v
            ),
            QPT3_PARAMETER: (
                self.qpt3_voltage_v
            ),
        }


@dataclass(frozen=True)
class QPTState:
    qpt1: QPTVoltageObservation
    qpt2: QPTVoltageObservation
    qpt3: QPTVoltageObservation

    command_coordinates: QPTCoordinates
    best_available_coordinates: QPTCoordinates

    all_inputs_from_readback: bool


def _finite(
    name: str,
    value: float,
) -> float:
    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


def _finite_nonnegative(
    name: str,
    value: float,
) -> float:
    value = _finite(
        name,
        value,
    )

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return value


def _validate_command(
    parameter_name: str,
    value: float,
) -> float:
    value = _finite_nonnegative(
        parameter_name,
        value,
    )

    definition = PARAMETERS[
        parameter_name
    ]

    if not (
        definition.minimum
        <= value
        <= definition.maximum
    ):
        raise ValueError(
            f"{parameter_name}={value} outside hard bounds "
            f"{definition.minimum}..{definition.maximum}"
        )

    return value


def qpt_coordinates_from_voltages(
    qpt1_voltage_v: float,
    qpt2_voltage_v: float,
    qpt3_voltage_v: float,
) -> QPTCoordinates:
    """
    Convert raw QPT voltages into reduced optical coordinates.

    This function intentionally accepts any finite voltages. That makes
    it suitable for physical readbacks, which need not exactly equal the
    valid command range.
    """

    v1 = _finite(
        "QPT1 voltage",
        qpt1_voltage_v,
    )

    v2 = _finite(
        "QPT2 voltage",
        qpt2_voltage_v,
    )

    v3 = _finite(
        "QPT3 voltage",
        qpt3_voltage_v,
    )

    outer = (
        v2
        - v3
    )

    middle = (
        v2
        - v1
    )

    global_focus = (
        outer
        + middle
    ) / 2.0

    asymmetry = (
        middle
        - outer
    ) / 2.0

    return QPTCoordinates(
        common_v=v2,
        global_focus_v=(
            global_focus
        ),
        asymmetry_v=(
            asymmetry
        ),
        outer_strength_v=(
            outer
        ),
        middle_strength_v=(
            middle
        ),
    )


def qpt_commands_from_cfa(
    common_v: float,
    global_focus_v: float,
    asymmetry_v: float,
) -> QPTCommandSet:
    """
    Convert the reduced coordinates C/F/A into the three actual PSU
    commands and enforce the individual QPT hard limits.
    """

    common = _finite(
        "QPT common mode",
        common_v,
    )

    focus = _finite(
        "QPT global focus",
        global_focus_v,
    )

    asymmetry = _finite(
        "QPT asymmetry",
        asymmetry_v,
    )

    v1 = (
        common
        - focus
        - asymmetry
    )

    v2 = common

    v3 = (
        common
        - focus
        + asymmetry
    )

    v1 = _validate_command(
        QPT1_PARAMETER,
        v1,
    )

    v2 = _validate_command(
        QPT2_PARAMETER,
        v2,
    )

    v3 = _validate_command(
        QPT3_PARAMETER,
        v3,
    )

    coordinates = (
        qpt_coordinates_from_voltages(
            v1,
            v2,
            v3,
        )
    )

    return QPTCommandSet(
        qpt1_voltage_v=v1,
        qpt2_voltage_v=v2,
        qpt3_voltage_v=v3,
        coordinates=coordinates,
    )


def qpt_commands_from_om(
    common_v: float,
    outer_strength_v: float,
    middle_strength_v: float,
) -> QPTCommandSet:
    """
    Alternative O/M representation.

        O = V2 - V3
        M = V2 - V1
    """

    common = _finite(
        "QPT common mode",
        common_v,
    )

    outer = _finite(
        "QPT outer strength",
        outer_strength_v,
    )

    middle = _finite(
        "QPT middle strength",
        middle_strength_v,
    )

    focus = (
        outer
        + middle
    ) / 2.0

    asymmetry = (
        middle
        - outer
    ) / 2.0

    return qpt_commands_from_cfa(
        common,
        focus,
        asymmetry,
    )


def qpt_voltage_observation(
    state: MachineState,
    parameter_name: str,
) -> QPTVoltageObservation:
    state.validate()

    if parameter_name not in (
        QPT_PARAMETERS
    ):
        raise ValueError(
            f"{parameter_name} is not a QPT parameter"
        )

    if (
        parameter_name
        not in state.parameters
    ):
        raise ValueError(
            f"State does not contain {parameter_name}"
        )

    command = _validate_command(
        parameter_name,
        state.parameters[
            parameter_name
        ],
    )

    readback = state.readbacks.get(
        parameter_name
    )

    if readback is not None:
        readback = _finite_nonnegative(
            f"{parameter_name} readback",
            readback,
        )

        return QPTVoltageObservation(
            value_v=readback,
            source="readback",
            command_v=command,
            readback_v=readback,
        )

    return QPTVoltageObservation(
        value_v=command,
        source="command",
        command_v=command,
        readback_v=None,
    )


def evaluate_qpt(
    state: MachineState,
) -> QPTState:
    """
    Evaluate command-space and best-observed QPT coordinates.

    Readbacks are preferred individually. Mixed command/readback states
    remain explicit through each QPTVoltageObservation.
    """

    qpt1 = qpt_voltage_observation(
        state,
        QPT1_PARAMETER,
    )

    qpt2 = qpt_voltage_observation(
        state,
        QPT2_PARAMETER,
    )

    qpt3 = qpt_voltage_observation(
        state,
        QPT3_PARAMETER,
    )

    command_coordinates = (
        qpt_coordinates_from_voltages(
            qpt1.command_v,
            qpt2.command_v,
            qpt3.command_v,
        )
    )

    best_coordinates = (
        qpt_coordinates_from_voltages(
            qpt1.value_v,
            qpt2.value_v,
            qpt3.value_v,
        )
    )

    return QPTState(
        qpt1=qpt1,
        qpt2=qpt2,
        qpt3=qpt3,
        command_coordinates=(
            command_coordinates
        ),
        best_available_coordinates=(
            best_coordinates
        ),
        all_inputs_from_readback=(
            qpt1.source == "readback"
            and qpt2.source == "readback"
            and qpt3.source == "readback"
        ),
    )


def qpt_focus_bounds_for_common_asymmetry(
    common_v: float,
    asymmetry_v: float,
) -> tuple[
    float,
    float,
]:
    """
    Exact feasible F interval for fixed C and A.

    The interval is obtained analytically from the individual QPT1 and
    QPT3 voltage limits.
    """

    common = _validate_command(
        QPT2_PARAMETER,
        common_v,
    )

    asymmetry = _finite(
        "QPT asymmetry",
        asymmetry_v,
    )

    qpt1 = PARAMETERS[
        QPT1_PARAMETER
    ]

    qpt3 = PARAMETERS[
        QPT3_PARAMETER
    ]

    minimum = max(
        common
        - asymmetry
        - qpt1.maximum,

        common
        + asymmetry
        - qpt3.maximum,
    )

    maximum = min(
        common
        - asymmetry
        - qpt1.minimum,

        common
        + asymmetry
        - qpt3.minimum,
    )

    if maximum < minimum:
        raise ValueError(
            "No feasible QPT global-focus interval for this C/A point"
        )

    return (
        float(minimum),
        float(maximum),
    )


def qpt_asymmetry_bounds_for_common_focus(
    common_v: float,
    global_focus_v: float,
) -> tuple[
    float,
    float,
]:
    """
    Exact feasible A interval for fixed C and F.
    """

    common = _validate_command(
        QPT2_PARAMETER,
        common_v,
    )

    focus = _finite(
        "QPT global focus",
        global_focus_v,
    )

    qpt1 = PARAMETERS[
        QPT1_PARAMETER
    ]

    qpt3 = PARAMETERS[
        QPT3_PARAMETER
    ]

    base = (
        common
        - focus
    )

    minimum = max(
        base
        - qpt1.maximum,

        qpt3.minimum
        - base,
    )

    maximum = min(
        base
        - qpt1.minimum,

        qpt3.maximum
        - base,
    )

    if maximum < minimum:
        raise ValueError(
            "No feasible QPT asymmetry interval for this C/F point"
        )

    return (
        float(minimum),
        float(maximum),
    )


def qpt_cfa_is_feasible(
    common_v: float,
    global_focus_v: float,
    asymmetry_v: float,
) -> bool:
    try:
        qpt_commands_from_cfa(
            common_v,
            global_focus_v,
            asymmetry_v,
        )

    except ValueError:
        return False

    return True


def qpt_global_focus_builder(
    state: MachineState,
    global_focus_v: float,
) -> dict[str, float]:
    """
    Change F while preserving the current command-space C and A.
    """

    current = evaluate_qpt(
        state
    ).command_coordinates

    return qpt_commands_from_cfa(
        current.common_v,
        global_focus_v,
        current.asymmetry_v,
    ).parameters


def qpt_asymmetry_builder(
    state: MachineState,
    asymmetry_v: float,
) -> dict[str, float]:
    """
    Change A while preserving the current command-space C and F.
    """

    current = evaluate_qpt(
        state
    ).command_coordinates

    return qpt_commands_from_cfa(
        current.common_v,
        current.global_focus_v,
        asymmetry_v,
    ).parameters


def qpt_common_builder(
    state: MachineState,
    common_v: float,
) -> dict[str, float]:
    """
    Change C while preserving the current command-space F and A.
    """

    current = evaluate_qpt(
        state
    ).command_coordinates

    return qpt_commands_from_cfa(
        common_v,
        current.global_focus_v,
        current.asymmetry_v,
    ).parameters


def qpt_global_focus_command(
    state: MachineState,
) -> float:
    return float(
        evaluate_qpt(
            state
        ).command_coordinates.global_focus_v
    )


def qpt_asymmetry_command(
    state: MachineState,
) -> float:
    return float(
        evaluate_qpt(
            state
        ).command_coordinates.asymmetry_v
    )


def qpt_common_command(
    state: MachineState,
) -> float:
    return float(
        evaluate_qpt(
            state
        ).command_coordinates.common_v
    )