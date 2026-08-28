from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
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
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState
from sirius.transition import apply_state


CommandBuilder = Callable[
    [MachineState, float],
    Mapping[str, float],
]

CoordinateReader = Callable[
    [MachineState],
    float,
]


@dataclass(frozen=True)
class DerivedScanPointResult:
    level: int
    step: float

    coordinate_value: float

    commands: dict[
        str,
        float,
    ]

    state: MachineState

    measurement: BeamMeasurement
    reference: SourceReference
    transmission: TransmissionResult

    comparison: ScalarComparison

    accepted_as_best: bool


@dataclass(frozen=True)
class DerivedScanResult:
    coordinate_name: str

    affected_parameters: tuple[
        str,
        ...
    ]

    cup: int

    minimum: float
    maximum: float

    initial_coordinate: float
    best_coordinate: float

    initial_state: MachineState
    final_state: MachineState

    baseline_measurement: BeamMeasurement
    baseline_reference: SourceReference
    baseline_transmission: TransmissionResult

    best_measurement: BeamMeasurement
    best_reference: SourceReference
    best_transmission: TransmissionResult

    points: tuple[
        DerivedScanPointResult,
        ...
    ]

    @property
    def improved(self) -> bool:
        return not math.isclose(
            self.initial_coordinate,
            self.best_coordinate,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


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


def _latest_reference(
    tracker: SourceReferenceTracker,
    mass_u: float,
) -> SourceReference:
    reference = tracker.latest

    if reference is None:
        raise ValueError(
            "Derived-coordinate transmission scan requires "
            "an existing Cup-1 reference"
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


def _candidate_state(
    base: MachineState,
    *,
    coordinate_name: str,
    coordinate_value: float,
    affected_parameters: tuple[
        str,
        ...
    ],
    command_builder: CommandBuilder,
) -> tuple[
    MachineState,
    dict[str, float],
]:
    updates = dict(
        command_builder(
            base,
            float(
                coordinate_value
            ),
        )
    )

    expected = set(
        affected_parameters
    )

    returned = set(
        updates
    )

    if returned != expected:
        raise ValueError(
            "Derived-coordinate command builder must return exactly "
            f"{sorted(expected)}, got {sorted(returned)}"
        )

    parameters = dict(
        base.parameters
    )

    readbacks = dict(
        base.readbacks
    )

    normalized_updates: dict[
        str,
        float,
    ] = {}

    for (
        parameter_name,
        command_value,
    ) in updates.items():
        if parameter_name not in PARAMETERS:
            raise ValueError(
                f"Unknown SIRIUS parameter: {parameter_name}"
            )

        definition = PARAMETERS[
            parameter_name
        ]

        if not definition.enabled:
            raise ValueError(
                f"{parameter_name} is disabled"
            )

        if not definition.optimizable:
            raise ValueError(
                f"{parameter_name} is not optimizable"
            )

        value = _finite(
            parameter_name,
            command_value,
        )

        if not (
            definition.minimum
            <= value
            <= definition.maximum
        ):
            raise ValueError(
                f"{parameter_name}={value} outside hard bounds "
                f"{definition.minimum}..{definition.maximum}"
            )

        parameters[
            parameter_name
        ] = value

        readbacks.pop(
            parameter_name,
            None,
        )

        normalized_updates[
            parameter_name
        ] = value

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
            "scan_coordinate": (
                coordinate_name
            ),
            "scan_coordinate_value": (
                float(
                    coordinate_value
                )
            ),
            "scan_affected_parameters": (
                list(
                    affected_parameters
                )
            ),
            "objective": (
                "cup1_normalized_transmission"
            ),
        },
    )

    candidate.validate()

    return (
        candidate,
        normalized_updates,
    )


def _apply_derived_target(
    adapter,
    current: MachineState,
    target: MachineState,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    *,
    coupled_transition_policy: (
        CoupledTransitionPolicy | None
    ),
    logger=None,
) -> MachineState:
    """
    Apply one logical derived-coordinate target.

    Without a coupled policy:
        use the normal single transition path.

    With a coupled policy:
        route the affected physical commands through bounded sequential
        microsteps, settling every physical channel before continuing.
    """

    if coupled_transition_policy is None:
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
        coupled_transition_policy,
        logger=logger,
    )

    return result.final_state


def scan_derived_coordinate_transmission_1d(
    adapter,
    current_state: MachineState,
    tracker: SourceReferenceTracker,
    *,
    coordinate_name: str,
    minimum: float,
    maximum: float,
    coordinate_reader: CoordinateReader,
    command_builder: CommandBuilder,
    affected_parameters: tuple[
        str,
        ...
    ],
    scan_policy: ScanPolicy,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    coupled_transition_policy: (
        CoupledTransitionPolicy | None
    ) = None,
    noise_floor_a: float | None = None,
    logger=None,
    maintenance_hook: Callable[
        [MachineState],
        MachineState,
    ] | None = None,
) -> DerivedScanResult:
    """
    Scan one derived coordinate that maps to multiple physical commands.

    Example:

        coordinate:
            guidefield difference

        affected commands:
            guidefield1_voltage_v
            guidefield2_voltage_v

    The command builder creates one coherent target MachineState for
    every coordinate value.

    A normal derived coordinate uses apply_state(). If an explicit
    CoupledTransitionPolicy is supplied, the logical multi-channel target
    is instead reached through bounded sequential physical microsteps.

    Objective:
        source-normalized transmission T_1->cup
    """

    current_state.validate()

    if not coordinate_name:
        raise ValueError(
            "coordinate_name must not be empty"
        )

    if current_state.cup is None:
        raise ValueError(
            "Current state must define a cup"
        )

    if current_state.cup == 1:
        raise ValueError(
            "Derived transmission scan requires a downstream cup"
        )

    if not affected_parameters:
        raise ValueError(
            "At least one affected parameter is required"
        )

    if len(
        set(
            affected_parameters
        )
    ) != len(
        affected_parameters
    ):
        raise ValueError(
            "affected_parameters contains duplicates"
        )

    for parameter_name in (
        affected_parameters
    ):
        if parameter_name not in PARAMETERS:
            raise ValueError(
                f"Unknown SIRIUS parameter: {parameter_name}"
            )

        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"Current state is missing {parameter_name}"
            )

        if (
            parameter_name
            not in settling_policies
        ):
            raise KeyError(
                f"No settling policy configured for {parameter_name}"
            )

    if coupled_transition_policy is not None:
        expected_parameters = set(
            affected_parameters
        )

        transition_parameters = set(
            coupled_transition_policy.parameter_order
        )

        if (
            transition_parameters
            != expected_parameters
        ):
            raise ValueError(
                "Coupled transition policy must contain exactly the "
                "derived scan affected parameters: "
                f"expected={sorted(expected_parameters)}, "
                f"got={sorted(transition_parameters)}"
            )

    minimum = _finite(
        "coordinate minimum",
        minimum,
    )

    maximum = _finite(
        "coordinate maximum",
        maximum,
    )

    if maximum <= minimum:
        raise ValueError(
            "Coordinate maximum must be greater than minimum"
        )

    physical_state = current_state

    # Unlike the older generic scanner, perform maintenance BEFORE the
    # baseline measurement as well. This prevents beginning a new scan
    # with an already expired source reference.
    if maintenance_hook is not None:
        physical_state = (
            maintenance_hook(
                physical_state
            )
        )

        physical_state.validate()

        if (
            physical_state.cup
            != current_state.cup
        ):
            raise ValueError(
                "Maintenance hook did not restore the same cup"
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

    initial_coordinate = _finite(
        "initial coordinate",
        coordinate_reader(
            initial_state
        ),
    )

    if not (
        minimum
        <= initial_coordinate
        <= maximum
    ):
        raise ValueError(
            f"Initial {coordinate_name}={initial_coordinate} "
            f"is outside scan window {minimum}..{maximum}"
        )

    baseline_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
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
            initial_state.cup,
            baseline_measurement,
            baseline_reference,
        )
    )

    if logger is not None:
        logger.log_event(
            "derived_scan_started",
            {
                "coordinate": (
                    coordinate_name
                ),
                "affected_parameters": (
                    affected_parameters
                ),
                "cup": (
                    initial_state.cup
                ),
                "stage": (
                    initial_state.stage
                ),
                "minimum": (
                    minimum
                ),
                "maximum": (
                    maximum
                ),
                "initial_coordinate": (
                    initial_coordinate
                ),
                "steps": (
                    scan_policy.steps
                ),
                "reference_state_id": (
                    baseline_reference.state_id
                ),
                "coupled_transition_enabled": (
                    coupled_transition_policy
                    is not None
                ),
                "transition_parameter_order": (
                    None
                    if coupled_transition_policy
                    is None
                    else list(
                        coupled_transition_policy.parameter_order
                    )
                ),
                "transition_max_steps": (
                    None
                    if coupled_transition_policy
                    is None
                    else {
                        name: float(
                            coupled_transition_policy.max_step_by_parameter[
                                name
                            ]
                        )
                        for name
                        in coupled_transition_policy.parameter_order
                    }
                ),
            },
        )

        logger.log_measurement(
            baseline_measurement,
            cup=initial_state.cup,
            state_id=(
                initial_state.state_id
            ),
            purpose=(
                f"derived_scan_baseline:{coordinate_name}"
            ),
        )

        logger.log_transmission(
            baseline_transmission
        )

    best_state = (
        initial_state
    )

    best_coordinate = (
        initial_coordinate
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
        DerivedScanPointResult
    ] = []

    evaluated_coordinates = {
        float(
            initial_coordinate
        )
    }

    previous_step: float | None = None

    for level_index, step in enumerate(
        scan_policy.steps,
        start=1,
    ):
        if previous_step is None:
            level_minimum = (
                minimum
            )

            level_maximum = (
                maximum
            )

        else:
            half_width = (
                previous_step
                * scan_policy.refinement_half_width_factor
            )

            level_minimum = max(
                minimum,
                best_coordinate
                - half_width,
            )

            level_maximum = min(
                maximum,
                best_coordinate
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
                "derived_scan_level_started",
                {
                    "coordinate": (
                        coordinate_name
                    ),
                    "level": (
                        level_index
                    ),
                    "step": (
                        step
                    ),
                    "minimum": (
                        level_minimum
                    ),
                    "maximum": (
                        level_maximum
                    ),
                },
            )

        for coordinate_value in grid:
            if any(
                math.isclose(
                    coordinate_value,
                    previous,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for previous
                in evaluated_coordinates
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
                    physical_state.cup
                    != current_state.cup
                ):
                    raise ValueError(
                        "Maintenance hook did not restore the same cup"
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

            (
                candidate,
                commands,
            ) = _candidate_state(
                physical_state,
                coordinate_name=(
                    coordinate_name
                ),
                coordinate_value=(
                    coordinate_value
                ),
                affected_parameters=(
                    affected_parameters
                ),
                command_builder=(
                    command_builder
                ),
            )

            physical_state = _apply_derived_target(
                adapter,
                physical_state,
                candidate,
                settling_policies,
                coupled_transition_policy=(
                    coupled_transition_policy
                ),
                logger=logger,
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

                best_coordinate = float(
                    coordinate_value
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
                    cup=(
                        physical_state.cup
                    ),
                    state_id=(
                        physical_state.state_id
                    ),
                    purpose=(
                        f"derived_scan_candidate:{coordinate_name}"
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
                        coordinate_name
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
                        "coordinate_value": (
                            coordinate_value
                        ),
                        "commands": (
                            commands
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

            points.append(
                DerivedScanPointResult(
                    level=(
                        level_index
                    ),
                    step=float(
                        step
                    ),
                    coordinate_value=float(
                        coordinate_value
                    ),
                    commands=commands,
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

            evaluated_coordinates.add(
                float(
                    coordinate_value
                )
            )

        previous_step = float(
            step
        )

    final_state = _apply_derived_target(
        adapter,
        physical_state,
        best_state,
        settling_policies,
        coupled_transition_policy=(
            coupled_transition_policy
        ),
        logger=logger,
    )

    if logger is not None:
        logger.log_event(
            "derived_scan_completed",
            {
                "coordinate": (
                    coordinate_name
                ),
                "initial_coordinate": (
                    initial_coordinate
                ),
                "best_coordinate": (
                    best_coordinate
                ),
                "best_transmission": (
                    best_transmission.transmission
                ),
                "evaluated_points": (
                    len(points)
                ),
                "final_state_id": (
                    final_state.state_id
                ),
            },
        )

    return DerivedScanResult(
        coordinate_name=(
            coordinate_name
        ),
        affected_parameters=(
            affected_parameters
        ),
        cup=final_state.cup,
        minimum=minimum,
        maximum=maximum,
        initial_coordinate=(
            initial_coordinate
        ),
        best_coordinate=(
            best_coordinate
        ),
        initial_state=(
            initial_state
        ),
        final_state=(
            final_state
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
        points=tuple(
            points
        ),
    )