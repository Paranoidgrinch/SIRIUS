from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

from sirius.coupled_transition import (
    CoupledTransitionPolicy,
    apply_coupled_transition,
)

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
)
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
from sirius.qpt_model import (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
    QPTCoordinates,
    evaluate_qpt,
    qpt_cfa_is_feasible,
    qpt_commands_from_cfa,
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


QPT_PARAMETERS = (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
)


@dataclass(frozen=True)
class QPTScanLevel:
    focus_step_v: float
    asymmetry_step_v: float

    def __post_init__(self) -> None:
        for name, value in (
            (
                "focus_step_v",
                self.focus_step_v,
            ),
            (
                "asymmetry_step_v",
                self.asymmetry_step_v,
            ),
        ):
            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero"
                )


@dataclass(frozen=True)
class QPT2DScanPolicy:
    """
    Coarse-to-fine two-dimensional scan in reduced QPT coordinates.

    C = QPT2 remains fixed.

    Only F and A are optimized.
    """

    initial_focus_half_width_v: float = 2000.0
    initial_asymmetry_half_width_v: float = 1500.0

    levels: tuple[
        QPTScanLevel,
        ...
    ] = field(
        default_factory=lambda: (
            QPTScanLevel(
                focus_step_v=500.0,
                asymmetry_step_v=500.0,
            ),
            QPTScanLevel(
                focus_step_v=100.0,
                asymmetry_step_v=100.0,
            ),
            QPTScanLevel(
                focus_step_v=25.0,
                asymmetry_step_v=25.0,
            ),
        )
    )

    refinement_half_width_factor: float = 2.0

    max_points_per_level: int = 500

    # None keeps the scanner usable in offline/unit-test contexts.
    # Real hardware orchestration should provide an explicit bounded
    # QPT coupled-transition policy.
    transition_policy: CoupledTransitionPolicy | None = None

    def __post_init__(self) -> None:
        for name, value in (
            (
                "initial_focus_half_width_v",
                self.initial_focus_half_width_v,
            ),
            (
                "initial_asymmetry_half_width_v",
                self.initial_asymmetry_half_width_v,
            ),
            (
                "refinement_half_width_factor",
                self.refinement_half_width_factor,
            ),
        ):
            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero"
                )

        if not self.levels:
            raise ValueError(
                "At least one QPT scan level is required"
            )

        for previous, current in zip(
            self.levels,
            self.levels[1:],
        ):
            if (
                current.focus_step_v
                >= previous.focus_step_v
            ):
                raise ValueError(
                    "QPT focus step must decrease between levels"
                )

            if (
                current.asymmetry_step_v
                >= previous.asymmetry_step_v
            ):
                raise ValueError(
                    "QPT asymmetry step must decrease between levels"
                )

        if self.max_points_per_level < 1:
            raise ValueError(
                "max_points_per_level must be at least 1"
            )

        if self.transition_policy is not None:
            if set(
                self.transition_policy.parameter_order
            ) != set(
                QPT_PARAMETERS
            ):
                raise ValueError(
                    "QPT transition policy must contain exactly "
                    "QPT1, QPT2, and QPT3"
                )


@dataclass(frozen=True)
class QPTScanPoint:
    level: int

    target_focus_v: float
    target_asymmetry_v: float
    common_v: float

    commands: dict[
        str,
        float,
    ]

    state: MachineState

    observed_coordinates: QPTCoordinates

    measurement: BeamMeasurement
    reference: SourceReference
    transmission: TransmissionResult

    comparison: ScalarComparison

    accepted_as_best: bool


@dataclass(frozen=True)
class QPT2DScanResult:
    initial_state: MachineState

    common_v: float

    initial_coordinates: QPTCoordinates
    best_target_focus_v: float
    best_target_asymmetry_v: float

    best_state: MachineState
    best_observed_coordinates: QPTCoordinates

    baseline_measurement: BeamMeasurement
    baseline_reference: SourceReference
    baseline_transmission: TransmissionResult

    best_measurement: BeamMeasurement
    best_reference: SourceReference
    best_transmission: TransmissionResult

    final_state: MachineState

    points: tuple[
        QPTScanPoint,
        ...
    ]


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


def _axis_grid(
    minimum: float,
    maximum: float,
    step: float,
) -> tuple[
    float,
    ...
]:
    minimum = _finite(
        "grid minimum",
        minimum,
    )

    maximum = _finite(
        "grid maximum",
        maximum,
    )

    step = _finite(
        "grid step",
        step,
    )

    if maximum < minimum:
        raise ValueError(
            "Grid maximum must not be below minimum"
        )

    if step <= 0:
        raise ValueError(
            "Grid step must be greater than zero"
        )

    values: list[
        float
    ] = []

    value = minimum

    tolerance = (
        step
        * 1e-9
        + 1e-9
    )

    while value <= maximum + tolerance:
        values.append(
            float(value)
        )

        value += step

        if len(values) > 10000:
            raise ValueError(
                "QPT grid axis is unreasonably large"
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

    return tuple(
        values
    )


def _latest_reference(
    tracker: SourceReferenceTracker,
    mass_u: float,
) -> SourceReference:
    reference = tracker.latest

    if reference is None:
        raise ValueError(
            "QPT scan requires an existing Cup-1 reference"
        )

    if not math.isclose(
        reference.mass_u,
        mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Cup-1 reference and QPT state use different ion masses"
        )

    return reference


def _estimate(
    transmission: TransmissionResult,
    measurement: BeamMeasurement,
) -> ScalarEstimate:
    result = ScalarEstimate(
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

    result.validate()

    return result


def _candidate_state(
    base: MachineState,
    *,
    common_v: float,
    focus_v: float,
    asymmetry_v: float,
) -> tuple[
    MachineState,
    dict[str, float],
]:
    command_set = qpt_commands_from_cfa(
        common_v,
        focus_v,
        asymmetry_v,
    )

    commands = command_set.parameters

    parameters = dict(
        base.parameters
    )

    readbacks = dict(
        base.readbacks
    )

    for (
        parameter_name,
        value,
    ) in commands.items():
        parameters[
            parameter_name
        ] = value

        # The previous readback no longer describes this target state.
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
            "objective": (
                "cup1_normalized_transmission"
            ),
            "scan_coordinate": (
                "qpt_focus_asymmetry_2d"
            ),
            "qpt_common_v": (
                float(
                    common_v
                )
            ),
            "qpt_target_focus_v": (
                float(
                    focus_v
                )
            ),
            "qpt_target_asymmetry_v": (
                float(
                    asymmetry_v
                )
            ),
        },
    )

    candidate.validate()

    return (
        candidate,
        commands,
    )


def _ordered_nearest_neighbor(
    points: tuple[
        tuple[
            float,
            float,
        ],
        ...
    ],
    *,
    start_focus: float,
    start_asymmetry: float,
    focus_scale: float,
    asymmetry_scale: float,
) -> tuple[
    tuple[
        float,
        float,
    ],
    ...
]:
    """
    Deterministic greedy ordering that prefers the nearest remaining
    F/A point.

    This does not change the tested grid; it only reduces unnecessary
    jumps between consecutive QPT command states.
    """

    remaining = list(
        points
    )

    ordered: list[
        tuple[
            float,
            float,
        ]
    ] = []

    current_focus = float(
        start_focus
    )

    current_asymmetry = float(
        start_asymmetry
    )

    focus_scale = max(
        abs(
            float(
                focus_scale
            )
        ),
        1e-12,
    )

    asymmetry_scale = max(
        abs(
            float(
                asymmetry_scale
            )
        ),
        1e-12,
    )

    while remaining:
        best_index = min(
            range(
                len(
                    remaining
                )
            ),
            key=lambda index: (
                (
                    (
                        remaining[index][0]
                        - current_focus
                    )
                    / focus_scale
                ) ** 2
                +
                (
                    (
                        remaining[index][1]
                        - current_asymmetry
                    )
                    / asymmetry_scale
                ) ** 2,
                remaining[index][0],
                remaining[index][1],
            ),
        )

        point = remaining.pop(
            best_index
        )

        ordered.append(
            point
        )

        (
            current_focus,
            current_asymmetry,
        ) = point

    return tuple(
        ordered
    )


def _feasible_grid(
    *,
    common_v: float,
    focus_minimum: float,
    focus_maximum: float,
    asymmetry_minimum: float,
    asymmetry_maximum: float,
    focus_step: float,
    asymmetry_step: float,
    max_points: int,
    start_focus: float,
    start_asymmetry: float,
) -> tuple[
    tuple[
        float,
        float,
    ],
    ...
]:
    focus_values = _axis_grid(
        focus_minimum,
        focus_maximum,
        focus_step,
    )

    asymmetry_values = _axis_grid(
        asymmetry_minimum,
        asymmetry_maximum,
        asymmetry_step,
    )

    feasible: list[
        tuple[
            float,
            float,
        ]
    ] = []

    for focus in focus_values:
        for asymmetry in asymmetry_values:
            if not qpt_cfa_is_feasible(
                common_v,
                focus,
                asymmetry,
            ):
                continue

            feasible.append(
                (
                    float(
                        focus
                    ),
                    float(
                        asymmetry
                    ),
                )
            )

            if len(
                feasible
            ) > max_points:
                raise ValueError(
                    "QPT scan level exceeds max_points_per_level"
                )

    if not feasible:
        raise ValueError(
            "QPT scan level contains no hardware-feasible points"
        )

    return _ordered_nearest_neighbor(
        tuple(
            feasible
        ),
        start_focus=(
            start_focus
        ),
        start_asymmetry=(
            start_asymmetry
        ),
        focus_scale=(
            focus_step
        ),
        asymmetry_scale=(
            asymmetry_step
        ),
    )


def _validate_inputs(
    current_state: MachineState,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> None:
    current_state.validate()

    if current_state.cup != 4:
        raise ValueError(
            "QPT F/A optimization requires Cup 4"
        )

    if current_state.stage not in (
        None,
        4,
    ):
        raise ValueError(
            "QPT F/A optimization requires stage 4 or no stage assignment"
        )

    for parameter_name in (
        QPT_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"QPT state is missing {parameter_name}"
            )

        if (
            parameter_name
            not in settling_policies
        ):
            raise KeyError(
                f"No settling policy configured for {parameter_name}"
            )

    _latest_reference(
        tracker,
        current_state.mass_u,
    )


def _apply_qpt_target(
    adapter,
    current: MachineState,
    target: MachineState,
    scan_policy: QPT2DScanPolicy,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    *,
    logger=None,
) -> MachineState:
    """
    Apply one logical QPT target.

    Offline/backward-compatible mode:
        normal apply_state()

    Hardware-safe mode:
        bounded coupled transition with sequential settled microsteps.
    """

    if scan_policy.transition_policy is None:
        transition = apply_state(
            adapter,
            current=current,
            target=target,
            settling_policies=(
                settling_policies
            ),
            select_target_cup=False,
        )

        if logger is not None:
            logger.log_state_transition(
                transition
            )

        return transition.observed_state

    result = apply_coupled_transition(
        adapter,
        current,
        target,
        settling_policies,
        scan_policy.transition_policy,
        logger=logger,
    )

    return result.final_state


def scan_qpt_focus_asymmetry_2d(
    adapter,
    current_state: MachineState,
    tracker: SourceReferenceTracker,
    scan_policy: QPT2DScanPolicy,
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
) -> QPT2DScanResult:
    """
    Optimize the QPT with only two optical degrees of freedom:

        F = global triplet focus
        A = middle-vs-outer balance

    QPT2 / common mode C remains frozen to its initial command value.

    All invalid C/F/A combinations are discarded before hardware access.
    """

    _validate_inputs(
        current_state,
        tracker,
        settling_policies,
    )

    physical_state = (
        current_state
    )

    if maintenance_hook is not None:
        physical_state = (
            maintenance_hook(
                physical_state
            )
        )

        physical_state.validate()

        if physical_state.cup != 4:
            raise ValueError(
                "Maintenance hook did not restore Cup 4"
            )

        if not math.isclose(
            physical_state.mass_u,
            current_state.mass_u,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Maintenance hook changed ion mass"
            )

    initial_state = (
        physical_state
    )

    initial_qpt = evaluate_qpt(
        initial_state
    )

    initial_coordinates = (
        initial_qpt.command_coordinates
    )

    # This is deliberately command-space C. It is the reproducible common
    # PSU setting to remain frozen throughout the F/A optimization.
    common_v = float(
        initial_coordinates.common_v
    )

    baseline_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=(
                noise_floor_a
            ),
        )
    )

    baseline_reference = (
        _latest_reference(
            tracker,
            initial_state.mass_u,
        )
    )

    baseline_transmission = (
        transmission_from_reference(
            4,
            baseline_measurement,
            baseline_reference,
        )
    )

    best_state = (
        initial_state
    )

    best_focus = float(
        initial_coordinates.global_focus_v
    )

    best_asymmetry = float(
        initial_coordinates.asymmetry_v
    )

    best_observed_coordinates = (
        initial_qpt.best_available_coordinates
    )

    best_measurement = (
        baseline_measurement
    )

    best_reference = (
        baseline_reference
    )

    best_transmission = (
        baseline_transmission
    )

    points: list[
        QPTScanPoint
    ] = []

    evaluated: list[
        tuple[
            float,
            float,
        ]
    ] = [
        (
            best_focus,
            best_asymmetry,
        )
    ]

    if logger is not None:
        logger.log_event(
            "qpt_2d_scan_started",
            {
                "state_id": (
                    initial_state.state_id
                ),
                "cup": 4,
                "mass_u": (
                    initial_state.mass_u
                ),
                "common_v": (
                    common_v
                ),
                "initial_focus_v": (
                    best_focus
                ),
                "initial_asymmetry_v": (
                    best_asymmetry
                ),
                "levels": [
                    {
                        "focus_step_v": (
                            level.focus_step_v
                        ),
                        "asymmetry_step_v": (
                            level.asymmetry_step_v
                        ),
                    }
                    for level
                    in scan_policy.levels
                ],
            },
        )

        logger.log_measurement(
            baseline_measurement,
            cup=4,
            state_id=(
                initial_state.state_id
            ),
            purpose=(
                "qpt_2d_scan_baseline"
            ),
        )

        logger.log_transmission(
            baseline_transmission
        )

    previous_level: (
        QPTScanLevel | None
    ) = None

    for level_index, level in enumerate(
        scan_policy.levels,
        start=1,
    ):
        if previous_level is None:
            focus_half_width = (
                scan_policy.initial_focus_half_width_v
            )

            asymmetry_half_width = (
                scan_policy.initial_asymmetry_half_width_v
            )

        else:
            focus_half_width = (
                previous_level.focus_step_v
                * scan_policy.refinement_half_width_factor
            )

            asymmetry_half_width = (
                previous_level.asymmetry_step_v
                * scan_policy.refinement_half_width_factor
            )

        focus_minimum = (
            best_focus
            - focus_half_width
        )

        focus_maximum = (
            best_focus
            + focus_half_width
        )

        asymmetry_minimum = (
            best_asymmetry
            - asymmetry_half_width
        )

        asymmetry_maximum = (
            best_asymmetry
            + asymmetry_half_width
        )

        grid = _feasible_grid(
            common_v=common_v,
            focus_minimum=(
                focus_minimum
            ),
            focus_maximum=(
                focus_maximum
            ),
            asymmetry_minimum=(
                asymmetry_minimum
            ),
            asymmetry_maximum=(
                asymmetry_maximum
            ),
            focus_step=(
                level.focus_step_v
            ),
            asymmetry_step=(
                level.asymmetry_step_v
            ),
            max_points=(
                scan_policy.max_points_per_level
            ),
            start_focus=(
                evaluate_qpt(
                    physical_state
                ).command_coordinates.global_focus_v
            ),
            start_asymmetry=(
                evaluate_qpt(
                    physical_state
                ).command_coordinates.asymmetry_v
            ),
        )

        if logger is not None:
            logger.log_event(
                "qpt_2d_scan_level_started",
                {
                    "level": (
                        level_index
                    ),
                    "focus_step_v": (
                        level.focus_step_v
                    ),
                    "asymmetry_step_v": (
                        level.asymmetry_step_v
                    ),
                    "focus_minimum_v": (
                        focus_minimum
                    ),
                    "focus_maximum_v": (
                        focus_maximum
                    ),
                    "asymmetry_minimum_v": (
                        asymmetry_minimum
                    ),
                    "asymmetry_maximum_v": (
                        asymmetry_maximum
                    ),
                    "feasible_points": (
                        len(
                            grid
                        )
                    ),
                },
            )

        for (
            target_focus,
            target_asymmetry,
        ) in grid:
            if any(
                math.isclose(
                    target_focus,
                    old_focus,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    target_asymmetry,
                    old_asymmetry,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for (
                    old_focus,
                    old_asymmetry,
                )
                in evaluated
            ):
                continue

            if maintenance_hook is not None:
                physical_state = (
                    maintenance_hook(
                        physical_state
                    )
                )

                physical_state.validate()

                if physical_state.cup != 4:
                    raise ValueError(
                        "Maintenance hook did not restore Cup 4"
                    )

                after_maintenance = (
                    evaluate_qpt(
                        physical_state
                    ).command_coordinates
                )

                if not math.isclose(
                    after_maintenance.common_v,
                    common_v,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Maintenance hook changed frozen QPT common mode"
                    )

            (
                candidate,
                commands,
            ) = _candidate_state(
                physical_state,
                common_v=(
                    common_v
                ),
                focus_v=(
                    target_focus
                ),
                asymmetry_v=(
                    target_asymmetry
                ),
            )

            physical_state = _apply_qpt_target(
                adapter,
                physical_state,
                candidate,
                scan_policy,
                settling_policies,
                logger=logger,
            )

            observed_qpt = evaluate_qpt(
                physical_state
            )

            observed_coordinates = (
                observed_qpt.best_available_coordinates
            )

            # Common command must remain exactly frozen even if physical
            # readback differs from it.
            command_coordinates = (
                observed_qpt.command_coordinates
            )

            if not math.isclose(
                command_coordinates.common_v,
                common_v,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "QPT common mode changed during F/A scan"
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
                    4,
                    candidate_measurement,
                    candidate_reference,
                )
            )

            comparison = (
                compare_estimates(
                    _estimate(
                        best_transmission,
                        best_measurement,
                    ),
                    _estimate(
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

                best_focus = float(
                    target_focus
                )

                best_asymmetry = float(
                    target_asymmetry
                )

                best_observed_coordinates = (
                    observed_coordinates
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
                logger.log_measurement(
                    candidate_measurement,
                    cup=4,
                    state_id=(
                        physical_state.state_id
                    ),
                    purpose=(
                        "qpt_2d_scan_candidate"
                    ),
                )

                logger.log_transmission(
                    candidate_transmission
                )

                logger.log_optimizer_decision(
                    stage=4,
                    cup=4,
                    parameter=(
                        "qpt_focus_asymmetry_2d"
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
                        "common_v": (
                            common_v
                        ),
                        "target_focus_v": (
                            target_focus
                        ),
                        "target_asymmetry_v": (
                            target_asymmetry
                        ),
                        "commands": (
                            commands
                        ),
                        "observed_focus_v": (
                            observed_coordinates.global_focus_v
                        ),
                        "observed_asymmetry_v": (
                            observed_coordinates.asymmetry_v
                        ),
                        "observed_common_v": (
                            observed_coordinates.common_v
                        ),
                        "transmission": (
                            candidate_transmission.transmission
                        ),
                        "transmission_sem": (
                            candidate_transmission.transmission_sem
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
                QPTScanPoint(
                    level=(
                        level_index
                    ),
                    target_focus_v=(
                        float(
                            target_focus
                        )
                    ),
                    target_asymmetry_v=(
                        float(
                            target_asymmetry
                        )
                    ),
                    common_v=(
                        common_v
                    ),
                    commands=(
                        dict(
                            commands
                        )
                    ),
                    state=(
                        physical_state
                    ),
                    observed_coordinates=(
                        observed_coordinates
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

            evaluated.append(
                (
                    float(
                        target_focus
                    ),
                    float(
                        target_asymmetry
                    ),
                )
            )

        previous_level = (
            level
        )

    # Restore the physically best tested state.
    final_state = _apply_qpt_target(
        adapter,
        physical_state,
        best_state,
        scan_policy,
        settling_policies,
        logger=logger,
    )

    final_qpt = evaluate_qpt(
        final_state
    )

    if not math.isclose(
        final_qpt.command_coordinates.common_v,
        common_v,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Restored QPT state no longer has frozen common mode"
        )

    if logger is not None:
        logger.log_event(
            "qpt_2d_scan_completed",
            {
                "final_state_id": (
                    final_state.state_id
                ),
                "common_v": (
                    common_v
                ),
                "best_target_focus_v": (
                    best_focus
                ),
                "best_target_asymmetry_v": (
                    best_asymmetry
                ),
                "best_observed_focus_v": (
                    best_observed_coordinates.global_focus_v
                ),
                "best_observed_asymmetry_v": (
                    best_observed_coordinates.asymmetry_v
                ),
                "best_transmission": (
                    best_transmission.transmission
                ),
                "evaluated_points": (
                    len(
                        points
                    )
                ),
            },
        )

    return QPT2DScanResult(
        initial_state=(
            initial_state
        ),
        common_v=(
            common_v
        ),
        initial_coordinates=(
            initial_coordinates
        ),
        best_target_focus_v=(
            best_focus
        ),
        best_target_asymmetry_v=(
            best_asymmetry
        ),
        best_state=(
            best_state
        ),
        best_observed_coordinates=(
            best_observed_coordinates
        ),
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
        final_state=(
            final_state
        ),
        points=tuple(
            points
        ),
    )