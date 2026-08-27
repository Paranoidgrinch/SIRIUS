from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.parameters import PARAMETERS
from sirius.state import MachineState


ENTRANCE_PARAMETER = "deceleration_voltage_v"
EXIT_PARAMETER = "acceleration_voltage_v"
COOLER_PARAMETER = "ion_cooler_voltage_v"


@dataclass(frozen=True)
class ElectrodeVoltageObservation:
    """
    One SIRIUS voltage observation.

    Commands are retained for reproducibility.
    Readbacks are preferred for physical interpretation when available.
    """

    value_v: float
    source: str

    command_v: float
    readback_v: float | None


@dataclass(frozen=True)
class CoolerEndElectrodeState:
    """
    Relative entrance/exit electrode state around the ion cooler.

    SIRIUS deliberately keeps these as controller-space bias magnitudes.

    No absolute laboratory-frame electrode potential is inferred here
    because that would require a validated wiring/reference convention.
    """

    cooler: ElectrodeVoltageObservation
    entrance: ElectrodeVoltageObservation
    exit: ElectrodeVoltageObservation

    command_end_difference_v: float
    best_available_end_difference_v: float

    command_common_bias_v: float
    best_available_common_bias_v: float

    all_inputs_from_readback: bool


@dataclass(frozen=True)
class CoolerEndElectrodePair:
    entrance_voltage_v: float
    exit_voltage_v: float

    difference_v: float
    common_bias_v: float


def _finite_nonnegative(
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

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return value


def _validated_command(
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


def electrode_voltage_observation(
    state: MachineState,
    parameter_name: str,
) -> ElectrodeVoltageObservation:
    state.validate()

    if parameter_name not in (
        COOLER_PARAMETER,
        ENTRANCE_PARAMETER,
        EXIT_PARAMETER,
    ):
        raise ValueError(
            f"{parameter_name} is not part of the ion-cooler electrode model"
        )

    if parameter_name not in state.parameters:
        raise ValueError(
            f"State does not contain {parameter_name}"
        )

    command = _validated_command(
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

        return ElectrodeVoltageObservation(
            value_v=readback,
            source="readback",
            command_v=command,
            readback_v=readback,
        )

    return ElectrodeVoltageObservation(
        value_v=command,
        source="command",
        command_v=command,
        readback_v=None,
    )


def evaluate_cooler_end_electrodes(
    state: MachineState,
) -> CoolerEndElectrodeState:
    """
    Evaluate HV1/HV4 around the current ion-cooler state.

    Coordinate definitions:

        delta_end = HV1 - HV4

        common = (HV1 + HV4) / 2

    The cooler voltage itself is retained because these electrode biases
    only have physical meaning in the context of the cooler operating
    point.
    """

    cooler = electrode_voltage_observation(
        state,
        COOLER_PARAMETER,
    )

    entrance = electrode_voltage_observation(
        state,
        ENTRANCE_PARAMETER,
    )

    exit_electrode = (
        electrode_voltage_observation(
            state,
            EXIT_PARAMETER,
        )
    )

    command_difference = (
        entrance.command_v
        - exit_electrode.command_v
    )

    observed_difference = (
        entrance.value_v
        - exit_electrode.value_v
    )

    command_common = (
        entrance.command_v
        + exit_electrode.command_v
    ) / 2.0

    observed_common = (
        entrance.value_v
        + exit_electrode.value_v
    ) / 2.0

    return CoolerEndElectrodeState(
        cooler=cooler,
        entrance=entrance,
        exit=exit_electrode,
        command_end_difference_v=(
            command_difference
        ),
        best_available_end_difference_v=(
            observed_difference
        ),
        command_common_bias_v=(
            command_common
        ),
        best_available_common_bias_v=(
            observed_common
        ),
        all_inputs_from_readback=(
            cooler.source == "readback"
            and entrance.source == "readback"
            and exit_electrode.source == "readback"
        ),
    )


def cooler_end_pair_from_commands(
    entrance_voltage_v: float,
    exit_voltage_v: float,
) -> CoolerEndElectrodePair:
    entrance = _validated_command(
        ENTRANCE_PARAMETER,
        entrance_voltage_v,
    )

    exit_voltage = _validated_command(
        EXIT_PARAMETER,
        exit_voltage_v,
    )

    return CoolerEndElectrodePair(
        entrance_voltage_v=(
            entrance
        ),
        exit_voltage_v=(
            exit_voltage
        ),
        difference_v=(
            entrance
            - exit_voltage
        ),
        common_bias_v=(
            (
                entrance
                + exit_voltage
            )
            / 2.0
        ),
    )


def cooler_end_pair_from_common_difference(
    *,
    common_bias_v: float,
    difference_v: float,
) -> CoolerEndElectrodePair:
    """
    Convert common/differential coordinates into HV1/HV4 commands.

        common = (HV1 + HV4) / 2
        delta  = HV1 - HV4

    therefore

        HV1 = common + delta / 2
        HV4 = common - delta / 2
    """

    common = float(
        common_bias_v
    )

    difference = float(
        difference_v
    )

    if not math.isfinite(
        common
    ):
        raise ValueError(
            "Common electrode bias must be finite"
        )

    if not math.isfinite(
        difference
    ):
        raise ValueError(
            "Electrode difference must be finite"
        )

    entrance = (
        common
        + difference / 2.0
    )

    exit_voltage = (
        common
        - difference / 2.0
    )

    return cooler_end_pair_from_commands(
        entrance,
        exit_voltage,
    )


def validate_cooler_bias_context(
    state: MachineState,
) -> None:
    """
    Safety/context check before HV1/HV4 optimization.

    The relative electrode supplies should not be optimized without an
    explicitly defined ion-cooler operating point.
    """

    state.validate()

    for parameter_name in (
        COOLER_PARAMETER,
        ENTRANCE_PARAMETER,
        EXIT_PARAMETER,
    ):
        if parameter_name not in state.parameters:
            raise ValueError(
                f"State is missing {parameter_name}"
            )

    _validated_command(
        COOLER_PARAMETER,
        state.parameters[
            COOLER_PARAMETER
        ],
    )

    _validated_command(
        ENTRANCE_PARAMETER,
        state.parameters[
            ENTRANCE_PARAMETER
        ],
    )

    _validated_command(
        EXIT_PARAMETER,
        state.parameters[
            EXIT_PARAMETER
        ],
    )