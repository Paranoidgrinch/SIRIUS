from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
)
from sirius.cooler_model import (
    IonCoolerEnergyState,
    ion_cooler_energy_state,
    nominal_cooler_command_for_residual_energy,
    require_valid_cooler_prediction,
)
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.objective import (
    ScalarComparison,
    ScalarEstimate,
    compare_estimates,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
    TransmissionResult,
    transmission_from_reference,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState
from sirius.safe_transition import apply_state


COOLER_PARAMETER = "ion_cooler_voltage_v"


@dataclass(frozen=True)
class ResidualEnergyScanPolicy:
    """
    Coarse-to-fine scan expressed in residual ion energy, not cooler voltage.

    Example:
        minimum_ev = 10
        maximum_ev = 120
        steps_ev   = (10, 1)

    First level:
        full 10..120 eV range in 10 eV increments.

    Second level:
        local refinement around the best target using 1 eV increments.
    """

    minimum_ev: float
    maximum_ev: float

    steps_ev: tuple[float, ...]

    refinement_half_width_factor: float = 1.0

    max_points_per_level: int = 500

    def __post_init__(self) -> None:
        if not math.isfinite(
            float(self.minimum_ev)
        ):
            raise ValueError(
                "minimum_ev must be finite"
            )

        if not math.isfinite(
            float(self.maximum_ev)
        ):
            raise ValueError(
                "maximum_ev must be finite"
            )

        if self.minimum_ev < 0:
            raise ValueError(
                "minimum_ev must be non-negative"
            )

        if self.maximum_ev <= self.minimum_ev:
            raise ValueError(
                "maximum_ev must be greater than minimum_ev"
            )

        if not self.steps_ev:
            raise ValueError(
                "At least one residual-energy scan step is required"
            )

        for step in self.steps_ev:
            if not math.isfinite(
                float(step)
            ):
                raise ValueError(
                    "Residual-energy scan steps must be finite"
                )

            if step <= 0:
                raise ValueError(
                    "Residual-energy scan steps must be greater than zero"
                )

        for previous, current in zip(
            self.steps_ev,
            self.steps_ev[1:],
        ):
            if current >= previous:
                raise ValueError(
                    "Residual-energy scan steps must strictly decrease"
                )

        if self.refinement_half_width_factor <= 0:
            raise ValueError(
                "refinement_half_width_factor must be greater than zero"
            )

        if self.max_points_per_level < 2:
            raise ValueError(
                "max_points_per_level must be at least 2"
            )


@dataclass(frozen=True)
class ResidualEnergyScanPoint:
    level: int
    step_ev: float

    target_residual_energy_ev: float

    nominal_cooler_command_v: float

    state: MachineState

    energy_state: IonCoolerEnergyState

    measurement: BeamMeasurement

    reference: SourceReference

    transmission: TransmissionResult

    comparison: ScalarComparison

    accepted_as_best: bool


@dataclass(frozen=True)
class ResidualEnergyScanResult:
    initial_state: MachineState

    initial_energy_state: IonCoolerEnergyState
    initial_measurement: BeamMeasurement
    initial_reference: SourceReference
    initial_transmission: TransmissionResult

    best_state: MachineState
    best_target_residual_energy_ev: float
    best_energy_state: IonCoolerEnergyState
    best_measurement: BeamMeasurement
    best_reference: SourceReference
    best_transmission: TransmissionResult

    final_state: MachineState

    points: tuple[
        ResidualEnergyScanPoint,
        ...
    ]


def _energy_grid(
    minimum_ev: float,
    maximum_ev: float,
    step_ev: float,
    *,
    max_points: int,
) -> tuple[float, ...]:
    if maximum_ev < minimum_ev:
        raise ValueError(
            "Energy maximum must not be below minimum"
        )

    if step_ev <= 0:
        raise ValueError(
            "Energy step must be greater than zero"
        )

    values: list[float] = []

    value = float(
        minimum_ev
    )

    tolerance = (
        abs(step_ev)
        * 1e-9
        + 1e-12
    )

    while value <= maximum_ev + tolerance:
        values.append(
            float(value)
        )

        if len(values) > max_points:
            raise ValueError(
                "Residual-energy scan exceeds max_points_per_level"
            )

        value += step_ev

    if not math.isclose(
        values[-1],
        maximum_ev,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        values.append(
            float(maximum_ev)
        )

    if len(values) > max_points:
        raise ValueError(
            "Residual-energy scan exceeds max_points_per_level"
        )

    return tuple(
        values
    )


def _candidate_state(
    base: MachineState,
    cooler_command_v: float,
    target_residual_energy_ev: float,
) -> MachineState:
    parameters = dict(
        base.parameters
    )

    parameters[
        COOLER_PARAMETER
    ] = float(
        cooler_command_v
    )

    readbacks = dict(
        base.readbacks
    )

    # Old cooler readback does not describe the new command.
    readbacks.pop(
        COOLER_PARAMETER,
        None,
    )

    candidate = MachineState(
        mass_u=base.mass_u,
        parameters=parameters,
        readbacks=readbacks,
        cup=base.cup,
        stage=base.stage,
        role="scan_candidate",
        rfq=deepcopy(
            base.rfq
        ),
        fixed_conditions=deepcopy(
            base.fixed_conditions
        ),
        metadata={
            **deepcopy(base.metadata),
            "objective": (
                "cup1_normalized_transmission"
            ),
            "scan_parameter": (
                "residual_energy_ev"
            ),
            "target_residual_energy_ev": (
                float(
                    target_residual_energy_ev
                )
            ),
            "nominal_cooler_command_v": (
                float(
                    cooler_command_v
                )
            ),
        },
    )

    candidate.validate()

    return candidate


def _latest_reference(
    tracker: SourceReferenceTracker,
    mass_u: float,
) -> SourceReference:
    reference = tracker.latest

    if reference is None:
        raise ValueError(
            "Residual-energy optimization requires a Cup-1 reference"
        )

    if not math.isclose(
        reference.mass_u,
        mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Cup-1 reference and machine state use different ion masses"
        )

    return reference


def _transmission_estimate(
    transmission: TransmissionResult,
    measurement: BeamMeasurement,
) -> ScalarEstimate:
    """
    Preserve the downstream beam-measurement noise-floor classification
    while optimizing the normalized transmission.
    """

    estimate = ScalarEstimate(
        value=float(
            transmission.transmission
        ),
        sem=float(
            transmission.transmission_sem
        ),
        below_noise_floor=(
            measurement.below_noise_floor
        ),
    )

    estimate.validate()

    return estimate


def _validate_inputs(
    current_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> None:
    current_state.validate()
    profile.validate()

    if current_state.cup != 3:
        raise ValueError(
            "Residual-energy ion-cooler scan requires Cup 3"
        )

    if current_state.stage not in (
        None,
        3,
    ):
        raise ValueError(
            "Residual-energy scan requires stage 3 or no stage assignment"
        )

    if current_state.mass_u != profile.mass_u:
        raise ValueError(
            "Machine state and mass profile must use the same ion mass"
        )

    for parameter_name in (
        "sputter_voltage_v",
        "extraction_voltage_v",
        COOLER_PARAMETER,
    ):
        if parameter_name not in current_state.parameters:
            raise ValueError(
                f"State is missing {parameter_name}"
            )

    if COOLER_PARAMETER not in settling_policies:
        raise KeyError(
            "No settling policy configured for ion_cooler_voltage_v"
        )

    _latest_reference(
        tracker,
        current_state.mass_u,
    )


def scan_residual_energy(
    adapter,
    current_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    scan_policy: ResidualEnergyScanPolicy,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    *,
    noise_floor_a: float | None = None,
    logger=None,
    maintenance_hook: Callable[
        [MachineState],
        MachineState,
    ] | None = None,
) -> ResidualEnergyScanResult:
    """
    Optimize Cup-3 transmission as a function of ion-cooler residual energy.

    Critical distinction:

        target residual energy
            -> physics-derived nominal cooler command

        nominal cooler command
            -> FLAVIA hardware command

        settled cooler readback
            -> observed residual energy

    No automatic command/readback offset correction is performed.
    """

    _validate_inputs(
        current_state,
        profile,
        tracker,
        settling_policies,
    )

    physical_state = current_state
    initial_state = current_state

    initial_energy_state = (
        ion_cooler_energy_state(
            physical_state
        )
    )

    initial_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
        )
    )

    initial_reference = (
        _latest_reference(
            tracker,
            physical_state.mass_u,
        )
    )

    initial_transmission = (
        transmission_from_reference(
            3,
            initial_measurement,
            initial_reference,
        )
    )

    if logger is not None:
        logger.log_event(
            "residual_energy_scan_started",
            {
                "state_id": (
                    initial_state.state_id
                ),
                "minimum_ev": (
                    scan_policy.minimum_ev
                ),
                "maximum_ev": (
                    scan_policy.maximum_ev
                ),
                "steps_ev": (
                    scan_policy.steps_ev
                ),
                "initial_energy_state": (
                    initial_energy_state
                ),
                "reference_state_id": (
                    initial_reference.state_id
                ),
            },
        )

        logger.log_measurement(
            initial_measurement,
            cup=3,
            state_id=(
                initial_state.state_id
            ),
            purpose=(
                "residual_energy_scan_baseline"
            ),
        )

        logger.log_transmission(
            initial_transmission
        )

    best_state = (
        physical_state
    )

    best_energy_state = (
        initial_energy_state
    )

    best_measurement = (
        initial_measurement
    )

    best_reference = (
        initial_reference
    )

    best_transmission = (
        initial_transmission
    )

    # For refinement we need a target coordinate.
    # The initial observed residual energy is the most meaningful
    # starting coordinate available.
    best_target_energy = max(
        scan_policy.minimum_ev,
        min(
            scan_policy.maximum_ev,
            initial_energy_state.residual_energy_best_available_ev,
        ),
    )

    points: list[
        ResidualEnergyScanPoint
    ] = []

    evaluated_targets: list[
        float
    ] = []

    previous_step: float | None = None

    for level_index, step_ev in enumerate(
        scan_policy.steps_ev,
        start=1,
    ):
        if previous_step is None:
            level_minimum = (
                scan_policy.minimum_ev
            )

            level_maximum = (
                scan_policy.maximum_ev
            )

        else:
            half_width = (
                previous_step
                * scan_policy.refinement_half_width_factor
            )

            level_minimum = max(
                scan_policy.minimum_ev,
                best_target_energy
                - half_width,
            )

            level_maximum = min(
                scan_policy.maximum_ev,
                best_target_energy
                + half_width,
            )

        grid = _energy_grid(
            level_minimum,
            level_maximum,
            step_ev,
            max_points=(
                scan_policy.max_points_per_level
            ),
        )

        if logger is not None:
            logger.log_event(
                "residual_energy_scan_level_started",
                {
                    "level": (
                        level_index
                    ),
                    "step_ev": (
                        step_ev
                    ),
                    "minimum_ev": (
                        level_minimum
                    ),
                    "maximum_ev": (
                        level_maximum
                    ),
                    "points": (
                        len(grid)
                    ),
                },
            )

        for target_energy in grid:
            if any(
                math.isclose(
                    target_energy,
                    previous,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for previous
                in evaluated_targets
            ):
                continue

            if maintenance_hook is not None:
                physical_state = (
                    maintenance_hook(
                        physical_state
                    )
                )

                physical_state.validate()

                if physical_state.cup != 3:
                    raise ValueError(
                        "Maintenance hook did not restore Cup 3"
                    )

                if (
                    physical_state.mass_u
                    != current_state.mass_u
                ):
                    raise ValueError(
                        "Maintenance hook changed ion mass"
                    )

            prediction = (
                nominal_cooler_command_for_residual_energy(
                    physical_state,
                    target_energy,
                )
            )

            cooler_command = (
                require_valid_cooler_prediction(
                    prediction
                )
            )

            candidate = (
                _candidate_state(
                    physical_state,
                    cooler_command,
                    target_energy,
                )
            )

            transition = apply_state(
                adapter,
                current=physical_state,
                target=candidate,
                settling_policies=(
                    settling_policies
                ),
                select_target_cup=False,
            )

            physical_state = (
                transition.observed_state
            )

            # This is calculated AFTER settling and therefore uses the new
            # stable cooler readback when one is available.
            energy_state = (
                ion_cooler_energy_state(
                    physical_state
                )
            )

            candidate_measurement = (
                measure_beam_current(
                    adapter,
                    measurement_policy,
                    noise_floor_a=(
                        noise_floor_a
                    ),
                )
            )

            candidate_reference = (
                _latest_reference(
                    tracker,
                    physical_state.mass_u,
                )
            )

            candidate_transmission = (
                transmission_from_reference(
                    3,
                    candidate_measurement,
                    candidate_reference,
                )
            )

            comparison = (
                compare_estimates(
                    _transmission_estimate(
                        best_transmission,
                        best_measurement,
                    ),
                    _transmission_estimate(
                        candidate_transmission,
                        candidate_measurement,
                    ),
                    comparison_policy,
                )
            )

            accepted = (
                comparison.decision
                == ComparisonDecision.BETTER
            )

            baseline_state_id = (
                best_state.state_id
            )

            if accepted:
                best_state = (
                    physical_state
                )

                best_target_energy = (
                    float(
                        target_energy
                    )
                )

                best_energy_state = (
                    energy_state
                )

                best_measurement = (
                    candidate_measurement
                )

                best_reference = (
                    candidate_reference
                )

                best_transmission = (
                    candidate_transmission
                )

            if logger is not None:
                logger.log_state_transition(
                    transition
                )

                logger.log_measurement(
                    candidate_measurement,
                    cup=3,
                    state_id=(
                        physical_state.state_id
                    ),
                    purpose=(
                        "residual_energy_scan_candidate"
                    ),
                )

                logger.log_transmission(
                    candidate_transmission
                )

                logger.log_event(
                    "residual_energy_observation",
                    {
                        "state_id": (
                            physical_state.state_id
                        ),
                        "level": (
                            level_index
                        ),
                        "step_ev": (
                            step_ev
                        ),
                        "target_residual_energy_ev": (
                            target_energy
                        ),
                        "nominal_cooler_command_v": (
                            cooler_command
                        ),
                        "cooler_readback_v": (
                            energy_state.cooler.readback_v
                        ),
                        "observed_residual_energy_ev": (
                            energy_state.residual_energy_best_available_ev
                        ),
                        "command_residual_energy_ev": (
                            energy_state.residual_energy_command_ev
                        ),
                        "beam_energy_best_available_ev": (
                            energy_state.beam_energy_best_available_ev
                        ),
                        "transmission": (
                            candidate_transmission.transmission
                        ),
                        "transmission_sem": (
                            candidate_transmission.transmission_sem
                        ),
                        "reference_state_id": (
                            candidate_reference.state_id
                        ),
                    },
                )

                logger.log_optimizer_decision(
                    stage=3,
                    cup=3,
                    parameter=(
                        "residual_energy_ev"
                    ),
                    decision=(
                        comparison.decision.value
                    ),
                    baseline_state_id=(
                        baseline_state_id
                    ),
                    candidate_state_id=(
                        physical_state.state_id
                    ),
                    details={
                        "target_residual_energy_ev": (
                            target_energy
                        ),
                        "observed_residual_energy_ev": (
                            energy_state.residual_energy_best_available_ev
                        ),
                        "nominal_cooler_command_v": (
                            cooler_command
                        ),
                        "delta": (
                            comparison.delta
                        ),
                        "required_margin": (
                            comparison.required_margin
                        ),
                        "accepted_as_best": (
                            accepted
                        ),
                    },
                )

            points.append(
                ResidualEnergyScanPoint(
                    level=level_index,
                    step_ev=float(
                        step_ev
                    ),
                    target_residual_energy_ev=float(
                        target_energy
                    ),
                    nominal_cooler_command_v=float(
                        cooler_command
                    ),
                    state=(
                        physical_state
                    ),
                    energy_state=(
                        energy_state
                    ),
                    measurement=(
                        candidate_measurement
                    ),
                    reference=(
                        candidate_reference
                    ),
                    transmission=(
                        candidate_transmission
                    ),
                    comparison=(
                        comparison
                    ),
                    accepted_as_best=(
                        accepted
                    ),
                )
            )

            evaluated_targets.append(
                float(
                    target_energy
                )
            )

        previous_step = float(
            step_ev
        )

    # Return the physical machine to the best candidate.
    final_transition = apply_state(
        adapter,
        current=physical_state,
        target=best_state,
        settling_policies=(
            settling_policies
        ),
        select_target_cup=False,
    )

    final_state = (
        final_transition.observed_state
    )

    if logger is not None:
        logger.log_state_transition(
            final_transition
        )

        logger.log_event(
            "residual_energy_scan_completed",
            {
                "best_state_id": (
                    final_state.state_id
                ),
                "best_target_residual_energy_ev": (
                    best_target_energy
                ),
                "best_observed_residual_energy_ev": (
                    best_energy_state.residual_energy_best_available_ev
                ),
                "best_cooler_command_v": (
                    best_state.parameters[
                        COOLER_PARAMETER
                    ]
                ),
                "best_cooler_readback_v": (
                    best_energy_state.cooler.readback_v
                ),
                "best_transmission": (
                    best_transmission.transmission
                ),
                "evaluated_points": (
                    len(points)
                ),
            },
        )

    return ResidualEnergyScanResult(
        initial_state=(
            initial_state
        ),
        initial_energy_state=(
            initial_energy_state
        ),
        initial_measurement=(
            initial_measurement
        ),
        initial_reference=(
            initial_reference
        ),
        initial_transmission=(
            initial_transmission
        ),
        best_state=(
            best_state
        ),
        best_target_residual_energy_ev=(
            best_target_energy
        ),
        best_energy_state=(
            best_energy_state
        ),
        best_measurement=(
            best_measurement
        ),
        best_reference=(
            best_reference
        ),
        best_transmission=(
            best_transmission
        ),
        final_state=(
            final_state
        ),
        points=tuple(
            points
        ),
    )