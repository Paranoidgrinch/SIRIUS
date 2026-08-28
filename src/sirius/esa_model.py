from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.parameters import PARAMETERS
from sirius.physics import beam_energy_ev
from sirius.state import MachineState


ESA_PARAMETER = "esa_voltage_v"

SPUTTER_PARAMETER = "sputter_voltage_v"
EXTRACTION_PARAMETER = "extraction_voltage_v"

DEFAULT_ESA_ENERGY_PER_VOLT = 10.0


@dataclass(frozen=True)
class VoltageObservation:
    value_v: float
    source: str

    command_v: float
    readback_v: float | None


@dataclass(frozen=True)
class ESAState:
    """
    Command-domain and best-observed ESA operating state.

    energy_per_volt values are diagnostic ratios:

        E_beam / U_ESA

    They are not assumed to be universal physical constants.
    """

    sputter: VoltageObservation
    extraction: VoltageObservation
    esa: VoltageObservation

    beam_energy_command_ev: float
    beam_energy_best_available_ev: float

    energy_per_volt_command: float | None
    energy_per_volt_best_available: float | None

    all_inputs_from_readback: bool


@dataclass(frozen=True)
class ESAVoltagePrediction:
    """
    Physics-informed initial ESA voltage estimate.

        U_ESA = E_beam / calibration

    The best available source-voltage observations are used.

    No command/readback offset compensation is applied.
    """

    beam_energy_best_available_ev: float

    energy_per_volt: float

    nominal_esa_command_v: float

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


def _positive_finite(
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

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return value


def voltage_observation(
    state: MachineState,
    parameter_name: str,
) -> VoltageObservation:
    state.validate()

    if parameter_name not in (
        SPUTTER_PARAMETER,
        EXTRACTION_PARAMETER,
        ESA_PARAMETER,
    ):
        raise ValueError(
            f"{parameter_name} is not part of the ESA energy model"
        )

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

    readback = state.readbacks.get(
        parameter_name
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


def _energy_per_volt(
    beam_energy: float,
    esa_voltage: float,
) -> float | None:
    if esa_voltage <= 0:
        return None

    return (
        beam_energy
        / esa_voltage
    )


def evaluate_esa(
    state: MachineState,
) -> ESAState:
    """
    Evaluate source beam energy and ESA operating ratio.

    Source-energy convention:

        E_beam = U_sputter + U_extraction

    for singly charged negative ions using the positive voltage
    magnitudes stored by SIRIUS.
    """

    sputter = voltage_observation(
        state,
        SPUTTER_PARAMETER,
    )

    extraction = voltage_observation(
        state,
        EXTRACTION_PARAMETER,
    )

    esa = voltage_observation(
        state,
        ESA_PARAMETER,
    )

    command_energy = beam_energy_ev(
        sputter.command_v,
        extraction.command_v,
    )

    best_energy = beam_energy_ev(
        sputter.value_v,
        extraction.value_v,
    )

    return ESAState(
        sputter=sputter,
        extraction=extraction,
        esa=esa,
        beam_energy_command_ev=(
            command_energy
        ),
        beam_energy_best_available_ev=(
            best_energy
        ),
        energy_per_volt_command=(
            _energy_per_volt(
                command_energy,
                esa.command_v,
            )
        ),
        energy_per_volt_best_available=(
            _energy_per_volt(
                best_energy,
                esa.value_v,
            )
        ),
        all_inputs_from_readback=(
            sputter.source == "readback"
            and extraction.source == "readback"
            and esa.source == "readback"
        ),
    )


def predict_esa_voltage(
    state: MachineState,
    *,
    energy_per_volt: float = (
        DEFAULT_ESA_ENERGY_PER_VOLT
    ),
) -> ESAVoltagePrediction:
    """
    Predict an initial ESA command from the physical source energy.

        U_ESA = E_beam / k

    where k defaults to 10 eV/V for the current CologneAMS ESA
    starting model.

    This is only a search seed. SIRIUS does not force the ESA readback
    or transmission to obey this ratio.
    """

    calibration = _positive_finite(
        "ESA energy-per-volt calibration",
        energy_per_volt,
    )

    sputter = voltage_observation(
        state,
        SPUTTER_PARAMETER,
    )

    extraction = voltage_observation(
        state,
        EXTRACTION_PARAMETER,
    )

    beam_energy = beam_energy_ev(
        sputter.value_v,
        extraction.value_v,
    )

    nominal_command = (
        beam_energy
        / calibration
    )

    definition = PARAMETERS[
        ESA_PARAMETER
    ]

    within_range = (
        definition.minimum
        <= nominal_command
        <= definition.maximum
    )

    return ESAVoltagePrediction(
        beam_energy_best_available_ev=(
            beam_energy
        ),
        energy_per_volt=(
            calibration
        ),
        nominal_esa_command_v=(
            nominal_command
        ),
        within_hardware_range=(
            within_range
        ),
    )


def require_valid_esa_prediction(
    prediction: ESAVoltagePrediction,
) -> float:
    """
    Return the nominal ESA command or raise when the physics seed lies
    outside the configured ESA hardware range.

    SIRIUS deliberately does not silently clip the seed.
    """

    if not prediction.within_hardware_range:
        definition = PARAMETERS[
            ESA_PARAMETER
        ]

        raise ValueError(
            "Physics-derived ESA command "
            f"{prediction.nominal_esa_command_v} V "
            "is outside SIRIUS limits "
            f"{definition.minimum}..{definition.maximum} V"
        )

    return float(
        prediction.nominal_esa_command_v
    )


def infer_energy_per_volt(
    state: MachineState,
) -> float:
    """
    Infer the effective E_beam/U_ESA ratio from the best available
    observations.

    This is intended for diagnostics and later empirical learning.
    """

    evaluated = evaluate_esa(
        state
    )

    ratio = (
        evaluated.energy_per_volt_best_available
    )

    if ratio is None:
        raise ValueError(
            "Cannot infer ESA calibration from zero ESA voltage"
        )

    return float(
        ratio
    )