from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
    MeasurementComparison,
    compare_measurements,
)
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.parameters import PARAMETERS
from sirius.settling import SettlingPolicy
from sirius.state import MachineState
from sirius.transition import (
    AppliedStateResult,
    apply_state,
)


@dataclass(frozen=True)
class ScanPolicy:
    """
    Generic coarse-to-fine one-dimensional scan policy.

    Example:
        steps=(1000.0, 200.0, 50.0)

    The first level scans the full effective range. Each following level
    scans a smaller window around the best command found so far.
    """

    steps: tuple[float, ...]

    refinement_half_width_factor: float = 1.0

    max_points_per_level: int = 500

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(
                "At least one scan step is required"
            )

        for step in self.steps:
            if not math.isfinite(
                float(step)
            ):
                raise ValueError(
                    "Scan steps must be finite"
                )

            if step <= 0:
                raise ValueError(
                    "Scan steps must be greater than zero"
                )

        for previous, current in zip(
            self.steps,
            self.steps[1:],
        ):
            if current >= previous:
                raise ValueError(
                    "Scan steps must strictly decrease "
                    "from coarse to fine"
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
class ScanPointResult:
    level: int
    step: float

    command_value: float

    state: MachineState
    measurement: BeamMeasurement

    comparison: MeasurementComparison

    accepted_as_best: bool


@dataclass(frozen=True)
class OneDimensionalScanResult:
    parameter_name: str

    initial_state: MachineState
    final_state: MachineState

    baseline_measurement: BeamMeasurement
    best_measurement: BeamMeasurement

    initial_command: float
    best_command: float

    effective_minimum: float
    effective_maximum: float

    points: tuple[
        ScanPointResult,
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

    # A previous readback for a changed command is stale until FLAVIA
    # provides a new physical observation.
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
        rfq=deepcopy(base.rfq),
        fixed_conditions=deepcopy(
            base.fixed_conditions
        ),
        metadata={
            **deepcopy(base.metadata),
            "scan_parameter": parameter_name,
            "scan_command": float(
                command_value
            ),
        },
    )

    candidate.validate()

    return candidate


def _generate_grid(
    minimum: float,
    maximum: float,
    step: float,
    *,
    max_points: int,
) -> tuple[float, ...]:
    if maximum < minimum:
        raise ValueError(
            "Scan maximum must not be below minimum"
        )

    if step <= 0:
        raise ValueError(
            "Scan step must be greater than zero"
        )

    values: list[float] = []

    value = float(
        minimum
    )

    tolerance = (
        abs(step)
        * 1e-9
        + 1e-12
    )

    while value <= maximum + tolerance:
        values.append(
            float(value)
        )

        if len(values) > max_points:
            raise ValueError(
                "Scan would exceed max_points_per_level"
            )

        value += step

    if not values:
        values.append(
            float(minimum)
        )

    if not math.isclose(
        values[-1],
        maximum,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        values.append(
            float(maximum)
        )

    if len(values) > max_points:
        raise ValueError(
            "Scan would exceed max_points_per_level"
        )

    return tuple(
        values
    )


def _resolve_effective_bounds(
    profile: MassProfile,
    parameter_name: str,
    current_command: float,
) -> tuple[float, float]:
    """
    Use learned bounds but never silently exclude the currently known
    machine command.

    If a previously learned range has become stale, the current command
    remains part of the scan rather than being discarded.
    """

    minimum, maximum = (
        profile.effective_bounds(
            parameter_name
        )
    )

    hard = PARAMETERS[
        parameter_name
    ]

    minimum = max(
        hard.minimum,
        min(
            minimum,
            current_command,
        ),
    )

    maximum = min(
        hard.maximum,
        max(
            maximum,
            current_command,
        ),
    )

    return (
        float(minimum),
        float(maximum),
    )


def scan_parameter_1d(
    adapter,
    current_state: MachineState,
    profile: MassProfile,
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
) -> OneDimensionalScanResult:
    """
    Perform a generic coarse-to-fine optimization of one SIRIUS parameter.

    The objective is the locally measured beam-current magnitude.

    The physical machine may temporarily visit worse candidates. At the
    end of the scan the best known command state is always restored.

    maintenance_hook can later be used by higher-level optimization code
    to perform periodic Cup-1 source-reference checks without coupling the
    generic scanner directly to reference-tracking logic.
    """

    current_state.validate()
    profile.validate()

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

    if current_state.cup is None:
        raise ValueError(
            "Current state must define the active measurement cup"
        )

    if current_state.mass_u != profile.mass_u:
        raise ValueError(
            "Machine state and mass profile must use the same ion mass"
        )

    if parameter_name not in settling_policies:
        raise KeyError(
            f"No settling policy configured for {parameter_name}"
        )

    initial_state = current_state

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

    if logger is not None:
        logger.log_event(
            "parameter_scan_started",
            {
                "parameter": parameter_name,
                "cup": current_state.cup,
                "stage": current_state.stage,
                "initial_state_id": (
                    current_state.state_id
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
                "steps": scan_policy.steps,
            },
        )

    baseline_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
        )
    )

    if logger is not None:
        logger.log_measurement(
            baseline_measurement,
            cup=current_state.cup,
            state_id=current_state.state_id,
            purpose=(
                f"scan_baseline:{parameter_name}"
            ),
        )

    best_state = current_state
    best_measurement = (
        baseline_measurement
    )
    best_command = initial_command

    physical_state = current_state

    results: list[
        ScanPointResult
    ] = []

    evaluated_commands = {
        initial_command
    }

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

        if logger is not None:
            logger.log_event(
                "parameter_scan_level_started",
                {
                    "parameter": parameter_name,
                    "level": level_index,
                    "step": step,
                    "minimum": level_minimum,
                    "maximum": level_maximum,
                    "points": len(grid),
                },
            )

        for command_value in grid:
            if any(
                _commands_equal(
                    command_value,
                    previous,
                )
                for previous in evaluated_commands
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

            candidate = _candidate_state(
                physical_state,
                parameter_name,
                command_value,
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

                logger.log_event(
                    "scan_candidate",
                    {
                        "parameter": parameter_name,
                        "level": level_index,
                        "step": step,
                        "command_value": (
                            command_value
                        ),
                        "state": physical_state,
                    },
                )

            candidate_measurement = (
                measure_beam_current(
                    adapter,
                    measurement_policy,
                    noise_floor_a=noise_floor_a,
                )
            )

            if logger is not None:
                logger.log_measurement(
                    candidate_measurement,
                    cup=physical_state.cup,
                    state_id=(
                        physical_state.state_id
                    ),
                    purpose=(
                        f"scan_candidate:{parameter_name}"
                    ),
                )

            comparison = (
                compare_measurements(
                    best_measurement,
                    candidate_measurement,
                    comparison_policy,
                )
            )

            accepted = (
                comparison.decision
                == ComparisonDecision.BETTER
            )

            if accepted:
                best_state = (
                    physical_state
                )

                best_measurement = (
                    candidate_measurement
                )

                best_command = float(
                    command_value
                )

            if logger is not None:
                logger.log_optimizer_decision(
                    stage=(
                        physical_state.stage
                        if physical_state.stage
                        is not None
                        else 0
                    ),
                    cup=physical_state.cup,
                    parameter=parameter_name,
                    decision=(
                        comparison.decision.value
                    ),
                    baseline_state_id=(
                        best_state.state_id
                        if accepted
                        else best_state.state_id
                    ),
                    candidate_state_id=(
                        physical_state.state_id
                    ),
                    details={
                        "level": level_index,
                        "step": step,
                        "command_value": (
                            command_value
                        ),
                        "delta_a": (
                            comparison.delta_a
                        ),
                        "required_margin_a": (
                            comparison.required_margin_a
                        ),
                        "uncertainty_score": (
                            comparison.uncertainty_score
                        ),
                        "accepted_as_best": (
                            accepted
                        ),
                    },
                )

            results.append(
                ScanPointResult(
                    level=level_index,
                    step=float(step),
                    command_value=float(
                        command_value
                    ),
                    state=physical_state,
                    measurement=(
                        candidate_measurement
                    ),
                    comparison=comparison,
                    accepted_as_best=accepted,
                )
            )

            evaluated_commands.add(
                float(command_value)
            )

        previous_step = float(
            step
        )

    # Always leave the physical machine at the best state found.
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
            "parameter_scan_completed",
            {
                "parameter": parameter_name,
                "initial_command": (
                    initial_command
                ),
                "best_command": (
                    best_command
                ),
                "initial_mean_a": (
                    baseline_measurement.mean_a
                ),
                "best_mean_a": (
                    best_measurement.mean_a
                ),
                "evaluated_points": (
                    len(results)
                ),
                "best_state_id": (
                    final_state.state_id
                ),
            },
        )

    return OneDimensionalScanResult(
        parameter_name=parameter_name,
        initial_state=initial_state,
        final_state=final_state,
        baseline_measurement=(
            baseline_measurement
        ),
        best_measurement=(
            best_measurement
        ),
        initial_command=(
            initial_command
        ),
        best_command=best_command,
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