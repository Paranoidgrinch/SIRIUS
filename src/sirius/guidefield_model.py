from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.mass_profile import MassProfile
from sirius.parameters import PARAMETERS
from sirius.physics import (
    guidefield_command_difference_v,
)
from sirius.state import MachineState


GUIDEFIELD1_PARAMETER = "guidefield1_voltage_v"
GUIDEFIELD2_PARAMETER = "guidefield2_voltage_v"

GUIDEFIELD_LENGTH_M = 0.720
GUIDEFIELD_GEOMETRY_FACTOR = 0.65


@dataclass(frozen=True)
class GuidefieldVoltageObservation:
    value_v: float
    source: str

    command_v: float
    readback_v: float | None


@dataclass(frozen=True)
class GuidefieldState:
    """
    Command-domain and best-observed guidefield quantities.

    raw_difference:
        GF1 - GF2

    field_equivalent:
        Geometric estimate using E = 0.65 * deltaU / l.

    No claim about the physical forward direction is made unless an
    empirical guidefield_forward_sign has been learned for this ion mass.
    """

    gf1: GuidefieldVoltageObservation
    gf2: GuidefieldVoltageObservation

    command_difference_v: float
    best_available_difference_v: float

    command_field_equivalent_v_per_m: float
    best_available_field_equivalent_v_per_m: float

    all_inputs_from_readback: bool

    learned_forward_sign: int | None

    forward_drive_command_v: float | None
    forward_drive_best_available_v: float | None


@dataclass(frozen=True)
class GuidefieldPair:
    gf1_voltage_v: float
    gf2_voltage_v: float

    difference_v: float

    common_mode_v: float


def _finite_nonnegative(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return value


def _validate_parameter_voltage(
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


def guidefield_voltage_observation(
    state: MachineState,
    parameter_name: str,
) -> GuidefieldVoltageObservation:
    state.validate()

    if parameter_name not in (
        GUIDEFIELD1_PARAMETER,
        GUIDEFIELD2_PARAMETER,
    ):
        raise ValueError(
            f"{parameter_name} is not a guidefield parameter"
        )

    if parameter_name not in state.parameters:
        raise ValueError(
            f"State does not contain {parameter_name}"
        )

    command = _validate_parameter_voltage(
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

        return GuidefieldVoltageObservation(
            value_v=readback,
            source="readback",
            command_v=command,
            readback_v=readback,
        )

    return GuidefieldVoltageObservation(
        value_v=command,
        source="command",
        command_v=command,
        readback_v=None,
    )


def guidefield_field_equivalent_v_per_m(
    difference_v: float,
) -> float:
    """
    Geometric field-equivalent estimate.

        E = 0.65 * deltaU / l

    Sign follows the command-space definition GF1 - GF2.

    This is not automatically interpreted as the actual physical
    longitudinal direction because supply/electrode polarity has not yet
    been experimentally established in SIRIUS.
    """

    difference = float(
        difference_v
    )

    if not math.isfinite(
        difference
    ):
        raise ValueError(
            "Guidefield difference must be finite"
        )

    return (
        GUIDEFIELD_GEOMETRY_FACTOR
        * difference
        / GUIDEFIELD_LENGTH_M
    )


def evaluate_guidefield(
    state: MachineState,
    *,
    profile: MassProfile | None = None,
) -> GuidefieldState:
    gf1 = guidefield_voltage_observation(
        state,
        GUIDEFIELD1_PARAMETER,
    )

    gf2 = guidefield_voltage_observation(
        state,
        GUIDEFIELD2_PARAMETER,
    )

    command_difference = (
        guidefield_command_difference_v(
            gf1.command_v,
            gf2.command_v,
        )
    )

    best_difference = (
        guidefield_command_difference_v(
            gf1.value_v,
            gf2.value_v,
        )
    )

    forward_sign = None

    if profile is not None:
        profile.validate()

        if not math.isclose(
            profile.mass_u,
            state.mass_u,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Guidefield state and mass profile use different ion masses"
            )

        forward_sign = (
            profile.guidefield_forward_sign
        )

    if forward_sign is None:
        forward_command = None
        forward_best = None

    else:
        forward_command = (
            forward_sign
            * command_difference
        )

        forward_best = (
            forward_sign
            * best_difference
        )

    return GuidefieldState(
        gf1=gf1,
        gf2=gf2,
        command_difference_v=(
            command_difference
        ),
        best_available_difference_v=(
            best_difference
        ),
        command_field_equivalent_v_per_m=(
            guidefield_field_equivalent_v_per_m(
                command_difference
            )
        ),
        best_available_field_equivalent_v_per_m=(
            guidefield_field_equivalent_v_per_m(
                best_difference
            )
        ),
        all_inputs_from_readback=(
            gf1.source == "readback"
            and gf2.source == "readback"
        ),
        learned_forward_sign=(
            forward_sign
        ),
        forward_drive_command_v=(
            forward_command
        ),
        forward_drive_best_available_v=(
            forward_best
        ),
    )


def guidefield_pair_from_difference(
    difference_v: float,
    *,
    common_mode_v: float,
) -> GuidefieldPair:
    """
    Convert differential/common-mode coordinates into GF1/GF2 commands.

        common = (GF1 + GF2) / 2
        delta  = GF1 - GF2

    therefore

        GF1 = common + delta / 2
        GF2 = common - delta / 2

    The resulting pair must respect the individual 0..30 V and 0..75 V
    hardware ranges.
    """

    difference = float(
        difference_v
    )

    common = float(
        common_mode_v
    )

    if not math.isfinite(
        difference
    ):
        raise ValueError(
            "Guidefield difference must be finite"
        )

    if not math.isfinite(
        common
    ):
        raise ValueError(
            "Guidefield common mode must be finite"
        )

    gf1 = (
        common
        + difference / 2.0
    )

    gf2 = (
        common
        - difference / 2.0
    )

    gf1 = _validate_parameter_voltage(
        GUIDEFIELD1_PARAMETER,
        gf1,
    )

    gf2 = _validate_parameter_voltage(
        GUIDEFIELD2_PARAMETER,
        gf2,
    )

    return GuidefieldPair(
        gf1_voltage_v=gf1,
        gf2_voltage_v=gf2,
        difference_v=(
            gf1 - gf2
        ),
        common_mode_v=(
            (gf1 + gf2) / 2.0
        ),
    )


def guidefield_pair_from_commands(
    gf1_voltage_v: float,
    gf2_voltage_v: float,
) -> GuidefieldPair:
    gf1 = _validate_parameter_voltage(
        GUIDEFIELD1_PARAMETER,
        gf1_voltage_v,
    )

    gf2 = _validate_parameter_voltage(
        GUIDEFIELD2_PARAMETER,
        gf2_voltage_v,
    )

    return GuidefieldPair(
        gf1_voltage_v=gf1,
        gf2_voltage_v=gf2,
        difference_v=(
            gf1 - gf2
        ),
        common_mode_v=(
            (gf1 + gf2) / 2.0
        ),
    )


def infer_forward_sign(
    positive_difference_response: float,
    negative_difference_response: float,
    *,
    minimum_relative_advantage: float = 0.05,
) -> int | None:
    """
    Infer which command-space difference sign appears to give better
    transmission.

    This is deliberately a small pure helper and does not modify the
    MassProfile automatically.

    Returns:
        +1  if positive deltaU clearly wins
        -1  if negative deltaU clearly wins
        None if the evidence is not sufficiently asymmetric

    A statistically aware comparison will later be used by the actual
    Guidefield optimizer. This helper is mainly for profile semantics and
    unit testing.
    """

    positive = _finite_nonnegative(
        "positive-difference response",
        positive_difference_response,
    )

    negative = _finite_nonnegative(
        "negative-difference response",
        negative_difference_response,
    )

    minimum_advantage = _finite_nonnegative(
        "minimum relative advantage",
        minimum_relative_advantage,
    )

    best_scale = max(
        positive,
        negative,
    )

    if best_scale == 0:
        return None

    relative_difference = (
        abs(
            positive
            - negative
        )
        / best_scale
    )

    if (
        relative_difference
        < minimum_advantage
    ):
        return None

    if positive > negative:
        return 1

    if negative > positive:
        return -1

    return None