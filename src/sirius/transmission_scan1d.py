from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
)
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.objective import (
    ScalarComparison,
    compare_estimates,
    estimate_from_transmission,
)
from sirius.parameters import PARAMETERS
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
    TransmissionResult,
    transmission_from_reference,
)
from sirius.scan1d import (
    ScanPolicy,
    _generate_grid,
    _resolve_effective_bounds,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState
from sirius.safe_transition import apply_state


@dataclass(frozen=True)
class TransmissionScanPointResult:
    level: int
    step: float

    command_value: float

    state: MachineState

    measurement: BeamMeasurement
    reference: SourceReference
    transmission: TransmissionResult

    comparison: ScalarComparison

    accepted_as_best: bool


@dataclass(frozen=True)
class TransmissionScanResult:
    parameter_name: str
    cup: int

    initial_state: MachineState
    final_state: MachineState

    baseline_measurement: BeamMeasurement
    baseline_reference: SourceReference
    baseline_transmission: TransmissionResult

    best_measurement: BeamMeasurement
    best_reference: SourceReference
    best_transmission: TransmissionResult

    initial_command: float
    best_command: float

    effective_minimum: float
    effective_maximum: float

    points: tuple[
        TransmissionScanPointResult,
        ...
    ]

    @property
    def improved(self) -> bool:
        return not math.isclose(
            self.initial_command,
            self.best_command,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def _commands_equal(
    first: float,
    second: float,
) -> bool:
    return math.isclose(
        float(first),
        float(second),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _candidate_state(
    base: MachineState,
    parameter_name: str,
    command_value: float,
) -> MachineState:
    parameters = dict(
        base.parameters
    )

    parameters[
        parameter_name
    ] = float(
        command_value
    )

    readbacks = dict(
        base.readbacks
    )

    readbacks.pop(
        parameter_name,
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
            "scan_parameter": (
                parameter_name
            ),
            "scan_command": float(
                command_value
            ),
            "objective": (
                "cup1_normalized_transmission"
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
            "Transmission optimization requires an existing Cup-1 reference"
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


def scan_parameter_transmission_1d(
    adapter,
    current_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    parameter_name: str,
    scan_policy: ScanPolicy,
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
) -> TransmissionScanResult:
    """
    Coarse-to-fine scan using Cup-1-normalized transmission as objective.

    Every candidate is normalized against the latest SourceReference in
    the tracker at the time of its measurement.

    This means a maintenance hook may perform a periodic Cup-1 source
    check and update the tracker between scan points without invalidating
    comparisons made later in the scan.
    """

    current_state.validate()
    profile.validate()

    if current_state.cup is None:
        raise ValueError(
            "Current state must define the active cup"
        )

    if not 2 <= current_state.cup <= 6:
        raise ValueError(
            "Transmission scan is intended for downstream Cups 2..6"
        )

    if current_state.mass_u != profile.mass_u:
        raise ValueError(
            "Machine state and mass profile must use the same ion mass"
        )

    if parameter_name not in PARAMETERS:
        raise KeyError(
            f"Unknown SIRIUS parameter: {parameter_name}"
        )

    definition = PARAMETERS[
        parameter_name
    ]

    if not definition.enabled:
        raise ValueError(
            f"{parameter_name} is currently disabled"
        )

    if not definition.optimizable:
        raise ValueError(
            f"{parameter_name} is not currently optimizable"
        )

    if parameter_name not in current_state.parameters:
        raise ValueError(
            f"Current state does not contain {parameter_name}"
        )

    if parameter_name not in settling_policies:
        raise KeyError(
            f"No settling policy configured for {parameter_name}"
        )

    initial_reference = _latest_reference(
        tracker,
        current_state.mass_u,
    )

    initial_command = float(
        current_state.parameters[
            parameter_name
        ]
    )

    effective_minimum, effective_maximum = (
        _resolve_effective_bounds(
            profile,
            parameter_name,
            initial_command,
        )
    )

    initial_state = current_state
    physical_state = current_state

    baseline_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
        )
    )

    baseline_reference = (
        initial_reference
    )

    baseline_transmission = (
        transmission_from_reference(
            current_state.cup,
            baseline_measurement,
            baseline_reference,
        )
    )

    if logger is not None:
        logger.log_event(
            "transmission_scan_started",
            {
                "parameter": (
                    parameter_name
                ),
                "cup": current_state.cup,
                "stage": (
                    current_state.stage
                ),
                "initial_command": (
                    initial_command
                ),
                "effective_minimum": (
                    effective_minimum
                ),
                "effective_maximum": (
                    effective_maximum
                ),
                "steps": (
                    scan_policy.steps
                ),
                "reference_state_id": (
                    baseline_reference.state_id
                ),
            },
        )

        logger.log_measurement(
            baseline_measurement,
            cup=current_state.cup,
            state_id=(
                current_state.state_id
            ),
            purpose=(
                f"transmission_scan_baseline:{parameter_name}"
            ),
        )

        logger.log_transmission(
            baseline_transmission
        )

    best_state = current_state
    best_measurement = (
        baseline_measurement
    )
    best_reference = (
        baseline_reference
    )
    best_transmission = (
        baseline_transmission
    )
    best_command = initial_command

    evaluated_commands = {
        initial_command
    }

    results: list[
        TransmissionScanPointResult
    ] = []

    previous_step: float | None = None

    for level_index, step in enumerate(
        scan_policy.steps,
        start=1,
    ):
        if previous_step is None:
            level_minimum = (
                effective_minimum
            )
            level_maximum = (
                effective_maximum
            )

        else:
            half_width = (
                previous_step
                * scan_policy.refinement_half_width_factor
            )

            level_minimum = max(
                effective_minimum,
                best_command
                - half_width,
            )

            level_maximum = min(
                effective_maximum,
                best_command
                + half_width,
            )

        grid = _generate_grid(
            level_minimum,
            level_maximum,
            step,
            max_points=(
                scan_policy.max_points_per_level
            ),
        )

        for command_value in grid:
            if any(
                _commands_equal(
                    command_value,
                    previous,
                )
                for previous
                in evaluated_commands
            ):
                continue

            if maintenance_hook is not None:
                physical_state = (
                    maintenance_hook(
                        physical_state
                    )
                )

                physical_state.validate()

                if (
                    physical_state.mass_u
                    != current_state.mass_u
                ):
                    raise ValueError(
                        "Maintenance hook changed ion mass"
                    )

                if (
                    physical_state.cup
                    != current_state.cup
                ):
                    raise ValueError(
                        "Maintenance hook did not restore the active cup"
                    )

            candidate = (
                _candidate_state(
                    physical_state,
                    parameter_name,
                    command_value,
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

            if logger is not None:
                logger.log_state_transition(
                    transition
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
                    physical_state.cup,
                    candidate_measurement,
                    candidate_reference,
                )
            )

            comparison = (
                compare_estimates(
                    estimate_from_transmission(
                        best_transmission
                    ),
                    estimate_from_transmission(
                        candidate_transmission
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

                best_measurement = (
                    candidate_measurement
                )

                best_reference = (
                    candidate_reference
                )

                best_transmission = (
                    candidate_transmission
                )

                best_command = float(
                    command_value
                )

            if logger is not None:
                logger.log_measurement(
                    candidate_measurement,
                    cup=(
                        physical_state.cup
                    ),
                    state_id=(
                        physical_state.state_id
                    ),
                    purpose=(
                        f"transmission_scan_candidate:{parameter_name}"
                    ),
                )

                logger.log_transmission(
                    candidate_transmission
                )

                logger.log_optimizer_decision(
                    stage=(
                        physical_state.stage
                        if physical_state.stage
                        is not None
                        else 0
                    ),
                    cup=(
                        physical_state.cup
                    ),
                    parameter=(
                        parameter_name
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
                        "level": (
                            level_index
                        ),
                        "step": (
                            step
                        ),
                        "command_value": (
                            command_value
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

            results.append(
                TransmissionScanPointResult(
                    level=level_index,
                    step=float(step),
                    command_value=float(
                        command_value
                    ),
                    state=physical_state,
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

            evaluated_commands.add(
                float(command_value)
            )

        previous_step = float(
            step
        )

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
            "transmission_scan_completed",
            {
                "parameter": (
                    parameter_name
                ),
                "cup": (
                    final_state.cup
                ),
                "initial_command": (
                    initial_command
                ),
                "best_command": (
                    best_command
                ),
                "baseline_transmission": (
                    baseline_transmission.transmission
                ),
                "best_transmission": (
                    best_transmission.transmission
                ),
                "evaluated_points": (
                    len(results)
                ),
                "best_reference_state_id": (
                    best_reference.state_id
                ),
            },
        )

    return TransmissionScanResult(
        parameter_name=parameter_name,
        cup=final_state.cup,
        initial_state=initial_state,
        final_state=final_state,
        baseline_measurement=(
            baseline_measurement
        ),
        baseline_reference=(
            baseline_reference
        ),
        baseline_transmission=(
            baseline_transmission
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
        initial_command=(
            initial_command
        ),
        best_command=(
            best_command
        ),
        effective_minimum=(
            effective_minimum
        ),
        effective_maximum=(
            effective_maximum
        ),
        points=tuple(
            results
        ),
    )