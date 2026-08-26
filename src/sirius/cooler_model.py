from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.parameters import PARAMETERS
from sirius.physics import beam_energy_ev
from sirius.state import MachineState


SOURCE_PARAMETERS = (
    "sputter_voltage_v",
    "extraction_voltage_v",
)

COOLER_PARAMETER = (
    "ion_cooler_voltage_v"
)


@dataclass(frozen=True)
class VoltageObservation:
    """
    Best available voltage magnitude for one physical quantity.

    SIRIUS uses positive voltage magnitudes internally even though the
    corresponding physical electrodes may be at negative high voltage.
    """

    value_v: float
    source: str

    command_v: float
    readback_v: float | None


@dataclass(frozen=True)
class IonCoolerEnergyState:
    """
    Command-domain and best-observed ion-cooler energy state.

    best_available values prefer physical readbacks and fall back to
    commands only when no readback is available.
    """

    sputter: VoltageObservation
    extraction: VoltageObservation
    cooler: VoltageObservation

    beam_energy_command_ev: float
    beam_energy_best_available_ev: float

    residual_energy_command_ev: float
    residual_energy_best_available_ev: float

    all_energy_inputs_from_readback: bool


@dataclass(frozen=True)
class CoolerVoltagePrediction:
    """
    Physics-derived nominal cooler command for a desired residual energy.

    This is deliberately only a nominal command estimate.

    SIRIUS does NOT compensate a command/readback offset here. The actual
    residual energy must be evaluated again from settled readbacks after
    applying the command.
    """

    desired_residual_energy_ev: float

    beam_energy_best_available_ev: float

    nominal_cooler_command_v: float

    within_hardware_range: bool


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


def voltage_observation(
    state: MachineState,
    parameter_name: str,
) -> VoltageObservation:
    """
    Return the best available voltage magnitude.

    Readback is preferred for physics.
    Command remains available for reproducibility.
    """

    state.validate()

    if parameter_name not in state.parameters:
        raise ValueError(
            f"State does not contain {parameter_name}"
        )

    command = _finite_nonnegative(
        f"{parameter_name} command",
        state.parameters[
            parameter_name
        ],
    )

    readback = (
        state.readbacks.get(
            parameter_name
        )
    )

    if readback is not None:
        readback = _finite_nonnegative(
            f"{parameter_name} readback",
            readback,
        )

        return VoltageObservation(
            value_v=readback,
            source="readback",
            command_v=command,
            readback_v=readback,
        )

    return VoltageObservation(
        value_v=command,
        source="command",
        command_v=command,
        readback_v=None,
    )


def ion_cooler_energy_state(
    state: MachineState,
) -> IonCoolerEnergyState:
    """
    Evaluate the current source/cooler energy state.

    Example with command values:

        sputter    = 8000 V
        extraction = 19600 V
        cooler     = 27558 V

        beam energy     = 27600 eV
        residual energy = 42 eV
    """

    sputter = voltage_observation(
        state,
        "sputter_voltage_v",
    )

    extraction = voltage_observation(
        state,
        "extraction_voltage_v",
    )

    cooler = voltage_observation(
        state,
        COOLER_PARAMETER,
    )

    command_beam_energy = beam_energy_ev(
        sputter.command_v,
        extraction.command_v,
    )

    best_beam_energy = beam_energy_ev(
        sputter.value_v,
        extraction.value_v,
    )

    command_residual = (
        command_beam_energy
        - cooler.command_v
    )

    best_residual = (
        best_beam_energy
        - cooler.value_v
    )

    return IonCoolerEnergyState(
        sputter=sputter,
        extraction=extraction,
        cooler=cooler,
        beam_energy_command_ev=(
            command_beam_energy
        ),
        beam_energy_best_available_ev=(
            best_beam_energy
        ),
        residual_energy_command_ev=(
            command_residual
        ),
        residual_energy_best_available_ev=(
            best_residual
        ),
        all_energy_inputs_from_readback=(
            sputter.source == "readback"
            and extraction.source == "readback"
            and cooler.source == "readback"
        ),
    )


def nominal_cooler_command_for_residual_energy(
    state: MachineState,
    desired_residual_energy_ev: float,
) -> CoolerVoltagePrediction:
    """
    Calculate a physics-derived nominal cooler command.

        U_cooler = E_beam - E_residual

    The best available source-voltage observations are used to estimate
    beam energy.

    No command/readback offset compensation is performed.
    """

    residual = _finite_nonnegative(
        "desired residual energy",
        desired_residual_energy_ev,
    )

    sputter = voltage_observation(
        state,
        "sputter_voltage_v",
    )

    extraction = voltage_observation(
        state,
        "extraction_voltage_v",
    )

    beam_energy = beam_energy_ev(
        sputter.value_v,
        extraction.value_v,
    )

    if residual > beam_energy:
        raise ValueError(
            "Desired residual energy exceeds incoming beam energy"
        )

    cooler_command = (
        beam_energy
        - residual
    )

    definition = PARAMETERS[
        COOLER_PARAMETER
    ]

    within_range = (
        definition.minimum
        <= cooler_command
        <= definition.maximum
    )

    return CoolerVoltagePrediction(
        desired_residual_energy_ev=(
            residual
        ),
        beam_energy_best_available_ev=(
            beam_energy
        ),
        nominal_cooler_command_v=(
            cooler_command
        ),
        within_hardware_range=(
            within_range
        ),
    )


def require_valid_cooler_prediction(
    prediction: CoolerVoltagePrediction,
) -> float:
    """
    Return a usable nominal cooler command or raise if it would exceed
    the SIRIUS ion-cooler hard limits.
    """

    if not prediction.within_hardware_range:
        definition = PARAMETERS[
            COOLER_PARAMETER
        ]

        raise ValueError(
            "Physics-derived ion-cooler command "
            f"{prediction.nominal_cooler_command_v} V "
            "is outside SIRIUS limits "
            f"{definition.minimum}..{definition.maximum} V"
        )

    return float(
        prediction.nominal_cooler_command_v
    )