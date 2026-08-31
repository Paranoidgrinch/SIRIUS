from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

from sirius.comparison import ComparisonPolicy
from sirius.mass_profile import MassProfile
from sirius.optimizer_api import (
    ObjectiveEvaluation,
    OptimizationAxis,
    OptimizationProblem,
    comparison_policy_comparator,
    OptimizationResult,
)
from sirius.rcds_optimizer import (
    RCDSPolicy,
    RobustConjugateDirectionOptimizer,
)
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.parameters import PARAMETERS
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
    TransmissionResult,
    transmission_from_reference,
)
from sirius.reference_orchestrator import (
    SourceReferenceCheckResult,
    perform_source_reference_check,
)
from sirius.scan1d import ScanPolicy
from sirius.settling import SettlingPolicy
from sirius.safe_transition import apply_state
from sirius.state import (
    MachineState,
    utc_now_iso,
)
from sirius.transition import (
    capture_readbacks,
)
from sirius.transmission_scan1d import (
    TransmissionScanResult,
    scan_parameter_transmission_1d,
)


CUP2_PRIMARY_PARAMETERS = (
    "lens2_voltage_v",
    "steerer_x1_v",
    "steerer_y1_v",
)

CUP2_UPSTREAM_RETUNE_PARAMETERS = (
    "einzel_lens_voltage_v",
)

CUP2_FROZEN_CUP1_PARAMETERS = (
    "sputter_voltage_v",
    "extraction_voltage_v",
    "magnet_current_a",
)

CUP2_REQUIRED_PARAMETERS = (
    *CUP2_PRIMARY_PARAMETERS,
    *CUP2_UPSTREAM_RETUNE_PARAMETERS,
    *CUP2_FROZEN_CUP1_PARAMETERS,
)


CUP2_PRIMARY_RCDS_MAX_CUP2_MEASUREMENTS = 74


class Cup2OptimizationError(RuntimeError):
    pass


class Cup2OptimizationNoBeamError(
    Cup2OptimizationError
):
    pass


@dataclass(frozen=True)
class Cup2OptimizationPolicy:
    """
    Initial Cup-2 coordinate-descent policy.

    Lens2 and X1/Y1 are primary Cup-2 controls.
    The einzel lens gets only a narrower upstream retuning window.
    """

    lens2_half_width_v: float = 3000.0

    lens2_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                500.0,
                100.0,
                25.0,
            )
        )
    )

    steerer_half_width_v: float = 150.0

    steerer_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                50.0,
                10.0,
                2.0,
            )
        )
    )

    einzel_half_width_v: float = 800.0

    einzel_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                200.0,
                50.0,
            )
        )
    )

    coordinate_passes: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            (
                "lens2_half_width_v",
                self.lens2_half_width_v,
            ),
            (
                "steerer_half_width_v",
                self.steerer_half_width_v,
            ),
            (
                "einzel_half_width_v",
                self.einzel_half_width_v,
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

        if self.coordinate_passes < 1:
            raise ValueError(
                "coordinate_passes must be at least 1"
            )


def cup2_primary_rcds_production_policy(
) -> RCDSPolicy:
    """Return the bounded live-machine Cup-2 RCDS policy."""

    return RCDSPolicy(
        max_iterations=2,
        max_evaluations=73,
        line_samples=7,
        line_half_width=0.35,
        minimum_direction_norm=1e-6,
        stall_iterations=2,
        parabolic_refinement=True,
        reuse_cached_evaluations=False,
    )


@dataclass(frozen=True)
class Cup2OptimizationResult:
    initial_state: MachineState

    scans: tuple[
        TransmissionScanResult,
        ...
    ]

    reference_checks: tuple[
        SourceReferenceCheckResult,
        ...
    ]

    final_state: MachineState

    final_measurement: BeamMeasurement
    final_reference: SourceReference
    final_transmission: TransmissionResult

    primary_optimization: (
        OptimizationResult | None
    ) = None

    primary_confirmation: (
        ObjectiveEvaluation | None
    ) = None


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


def _local_profile(
    profile: MassProfile,
    parameter_name: str,
    center: float,
    half_width: float,
) -> MassProfile:
    """
    Build an ephemeral local search window without modifying persistent
    learned bounds.
    """

    local = deepcopy(
        profile
    )

    hard = PARAMETERS[
        parameter_name
    ]

    learned_minimum, learned_maximum = (
        profile.effective_bounds(
            parameter_name
        )
    )

    allowed_minimum = max(
        float(hard.minimum),
        min(
            float(center),
            float(learned_minimum),
        ),
    )

    allowed_maximum = min(
        float(hard.maximum),
        max(
            float(center),
            float(learned_maximum),
        ),
    )

    local_minimum = max(
        allowed_minimum,
        float(center)
        - float(half_width),
    )

    local_maximum = min(
        allowed_maximum,
        float(center)
        + float(half_width),
    )

    local.set_learned_range(
        parameter_name,
        local_minimum,
        local_maximum,
        source="cup2_local_window",
    )

    return local


def _build_primary_rcds_problem(
    current_state: MachineState,
    profile: MassProfile,
    comparison_policy: ComparisonPolicy,
    optimization_policy: Cup2OptimizationPolicy,
) -> OptimizationProblem:
    """Build the local 3-D Cup-2 RCDS search geometry."""

    current_state.validate()
    profile.validate()

    if current_state.mass_u != profile.mass_u:
        raise ValueError(
            "Machine state and mass profile must use the same ion mass"
        )

    if current_state.cup != 2:
        raise ValueError(
            "Cup-2 RCDS problem requires cup 2"
        )

    if current_state.stage not in (
        None,
        2,
    ):
        raise ValueError(
            "Cup-2 RCDS problem requires stage 2 or no stage assignment"
        )

    half_widths = {
        "lens2_voltage_v": (
            optimization_policy.lens2_half_width_v
        ),
        "steerer_x1_v": (
            optimization_policy.steerer_half_width_v
        ),
        "steerer_y1_v": (
            optimization_policy.steerer_half_width_v
        ),
    }

    axes: list[
        OptimizationAxis
    ] = []

    initial_point: list[
        float
    ] = []

    for parameter_name in (
        CUP2_PRIMARY_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"Cup-2 state is missing {parameter_name}"
            )

        center = float(
            current_state.parameters[
                parameter_name
            ]
        )

        local_profile = _local_profile(
            profile,
            parameter_name,
            center,
            half_widths[
                parameter_name
            ],
        )

        minimum, maximum = (
            local_profile.effective_bounds(
                parameter_name
            )
        )

        minimum = float(
            minimum
        )

        maximum = float(
            maximum
        )

        if maximum <= minimum:
            raise Cup2OptimizationError(
                "Local RCDS bounds collapse for "
                f"{parameter_name}: "
                f"{minimum}..{maximum}"
            )

        axes.append(
            OptimizationAxis(
                name=parameter_name,
                minimum=minimum,
                maximum=maximum,
            )
        )

        initial_point.append(
            center
        )

    return OptimizationProblem(
        axes=tuple(
            axes
        ),
        initial_point=tuple(
            initial_point
        ),
        maximize=True,
        comparison=(
            comparison_policy_comparator(
                policy=comparison_policy
            )
        ),
    )


def _retag_final_state(
    state: MachineState,
) -> MachineState:
    result = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            state.parameters
        ),
        readbacks=dict(
            state.readbacks
        ),
        cup=2,
        stage=2,
        role="stage_best",
        rfq=deepcopy(
            state.rfq
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata={
            **deepcopy(state.metadata),
            "optimized_stage": 2,
            "objective": (
                "cup1_normalized_transmission"
            ),
        },
    )

    result.validate()

    return result


def _validate_inputs(
    current_state: MachineState,
    cup1_reference_state: MachineState,
    profile: MassProfile,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> None:
    current_state.validate()
    cup1_reference_state.validate()
    profile.validate()

    if current_state.mass_u != profile.mass_u:
        raise ValueError(
            "Machine state and mass profile must use the same ion mass"
        )

    if (
        cup1_reference_state.mass_u
        != current_state.mass_u
    ):
        raise ValueError(
            "Cup-1 reference and Cup-2 state must use the same ion mass"
        )

    if current_state.cup != 2:
        raise ValueError(
            "Cup-2 optimization requires cup 2"
        )

    if current_state.stage not in (
        None,
        2,
    ):
        raise ValueError(
            "Cup-2 optimization requires stage 2 or no stage assignment"
        )

    if cup1_reference_state.cup != 1:
        raise ValueError(
            "Saved Cup-1 reference state must select cup 1"
        )

    for parameter_name in (
        CUP2_REQUIRED_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"Cup-2 state is missing {parameter_name}"
            )

    for parameter_name in (
        CUP2_FROZEN_CUP1_PARAMETERS
    ):
        if (
            parameter_name
            not in cup1_reference_state.parameters
        ):
            raise ValueError(
                f"Cup-1 reference state is missing {parameter_name}"
            )

        current_value = (
            current_state.parameters[
                parameter_name
            ]
        )

        reference_value = (
            cup1_reference_state.parameters[
                parameter_name
            ]
        )

        if not _commands_equal(
            current_value,
            reference_value,
        ):
            raise ValueError(
                f"{parameter_name} must still match the "
                "Cup-1 reference command before Cup-2 optimization"
            )

    optimizable_here = (
        *CUP2_PRIMARY_PARAMETERS,
        *CUP2_UPSTREAM_RETUNE_PARAMETERS,
    )

    for parameter_name in (
        optimizable_here
    ):
        if (
            parameter_name
            not in settling_policies
        ):
            raise KeyError(
                f"No settling policy configured for {parameter_name}"
            )


def _assert_frozen_cup1_parameters(
    state: MachineState,
    cup1_reference_state: MachineState,
) -> None:
    for parameter_name in (
        CUP2_FROZEN_CUP1_PARAMETERS
    ):
        actual = state.parameters[
            parameter_name
        ]

        expected = (
            cup1_reference_state.parameters[
                parameter_name
            ]
        )

        if not _commands_equal(
            actual,
            expected,
        ):
            raise Cup2OptimizationError(
                f"{parameter_name} changed during Cup-2 optimization"
            )
@dataclass
class _Cup2PrimaryRCDSEvaluator:
    """
    Stateful bridge from a 3-D RCDS point to one real Cup-2
    transmission evaluation.

    Hardware execution remains exclusively behind
    sirius.safe_transition.apply_state().
    """

    adapter: object
    working_state: MachineState
    cup1_reference_state: MachineState
    tracker: SourceReferenceTracker

    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ]

    measurement_policy: MeasurementPolicy

    noise_floor_a: float | None = None
    logger: object | None = None

    maintenance_hook: (
        Callable[
            [MachineState],
            MachineState,
        ]
        | None
    ) = None

    def __post_init__(
        self,
    ) -> None:
        self.working_state.validate()
        self.cup1_reference_state.validate()

        if self.working_state.cup != 2:
            raise ValueError(
                "Cup-2 RCDS evaluator requires cup 2"
            )

        if self.working_state.stage not in (
            None,
            2,
        ):
            raise ValueError(
                "Cup-2 RCDS evaluator requires stage 2 "
                "or no stage assignment"
            )

        if (
            self.working_state.mass_u
            != self.cup1_reference_state.mass_u
        ):
            raise ValueError(
                "Cup-1 reference state and Cup-2 working state "
                "must use the same ion mass"
            )

        _assert_frozen_cup1_parameters(
            self.working_state,
            self.cup1_reference_state,
        )

        for parameter_name in (
            CUP2_PRIMARY_PARAMETERS
        ):
            if (
                parameter_name
                not in self.working_state.parameters
            ):
                raise ValueError(
                    f"Cup-2 state is missing {parameter_name}"
                )

            if (
                parameter_name
                not in self.settling_policies
            ):
                raise KeyError(
                    "No settling policy configured for "
                    f"{parameter_name}"
                )

    def __call__(
        self,
        point: tuple[
            float,
            ...
        ],
    ) -> ObjectiveEvaluation:
        if len(
            point
        ) != len(
            CUP2_PRIMARY_PARAMETERS
        ):
            raise ValueError(
                "Cup-2 primary RCDS point must contain "
                "lens2, X1 and Y1"
            )

        requested_point = tuple(
            float(value)
            for value
            in point
        )

        if not all(
            math.isfinite(
                value
            )
            for value
            in requested_point
        ):
            raise ValueError(
                "Cup-2 RCDS point must contain only finite values"
            )

        if self.maintenance_hook is not None:
            refreshed = (
                self.maintenance_hook(
                    self.working_state
                )
            )

            refreshed.validate()

            if (
                refreshed.mass_u
                != self.working_state.mass_u
            ):
                raise Cup2OptimizationError(
                    "RCDS maintenance hook changed ion mass"
                )

            if refreshed.cup != 2:
                raise Cup2OptimizationError(
                    "RCDS maintenance hook did not restore Cup 2"
                )

            if refreshed.stage not in (
                None,
                2,
            ):
                raise Cup2OptimizationError(
                    "RCDS maintenance hook changed optimization stage"
                )

            self.working_state = (
                refreshed
            )

        _assert_frozen_cup1_parameters(
            self.working_state,
            self.cup1_reference_state,
        )

        parameters = dict(
            self.working_state.parameters
        )

        readbacks = dict(
            self.working_state.readbacks
        )

        for (
            parameter_name,
            command_value,
        ) in zip(
            CUP2_PRIMARY_PARAMETERS,
            requested_point,
        ):
            parameters[
                parameter_name
            ] = float(
                command_value
            )

            # A readback captured before the new command must
            # never masquerade as verification of this candidate.
            readbacks.pop(
                parameter_name,
                None,
            )

        candidate = MachineState(
            mass_u=(
                self.working_state.mass_u
            ),
            parameters=parameters,
            readbacks=readbacks,
            cup=2,
            stage=(
                self.working_state.stage
            ),
            role="optimizer_candidate",
            rfq=deepcopy(
                self.working_state.rfq
            ),
            fixed_conditions=deepcopy(
                self.working_state.fixed_conditions
            ),
            metadata={
                **deepcopy(
                    self.working_state.metadata
                ),
                "optimizer": "rcds",
                "optimizer_axes": (
                    CUP2_PRIMARY_PARAMETERS
                ),
                "objective": (
                    "cup1_normalized_transmission"
                ),
            },
        )

        candidate.validate()

        transition = apply_state(
            self.adapter,
            current=(
                self.working_state
            ),
            target=candidate,
            settling_policies=(
                self.settling_policies
            ),
            select_target_cup=False,
        )

        observed = (
            transition.observed_state
        )

        observed.validate()

        if observed.cup != 2:
            raise Cup2OptimizationError(
                "Safe transition did not preserve Cup 2"
            )

        if observed.stage not in (
            None,
            2,
        ):
            raise Cup2OptimizationError(
                "Safe transition changed Cup-2 optimization stage"
            )

        if (
            observed.mass_u
            != self.working_state.mass_u
        ):
            raise Cup2OptimizationError(
                "Safe transition changed ion mass"
            )

        self.working_state = (
            observed
        )

        _assert_frozen_cup1_parameters(
            self.working_state,
            self.cup1_reference_state,
        )

        if self.logger is not None:
            self.logger.log_state_transition(
                transition
            )

        candidate_measurement = (
            measure_beam_current(
                self.adapter,
                self.measurement_policy,
                noise_floor_a=(
                    self.noise_floor_a
                ),
            )
        )

        candidate_reference = (
            self.tracker.latest
        )

        if candidate_reference is None:
            raise Cup2OptimizationError(
                "Cup-2 RCDS evaluation requires "
                "a current Cup-1 source reference"
            )

        if not math.isclose(
            float(
                candidate_reference.mass_u
            ),
            float(
                self.working_state.mass_u
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Cup2OptimizationError(
                "Cup-1 source reference and Cup-2 state "
                "use different ion masses"
            )

        candidate_transmission = (
            transmission_from_reference(
                2,
                candidate_measurement,
                candidate_reference,
            )
        )

        if self.logger is not None:
            self.logger.log_measurement(
                candidate_measurement,
                cup=2,
                state_id=(
                    self.working_state.state_id
                ),
                purpose="cup2_rcds_candidate",
            )

            self.logger.log_transmission(
                candidate_transmission
            )

        return ObjectiveEvaluation(
            # RCDS explicitly requires this to be the requested
            # optimizer point. The observed MachineState is kept
            # separately in evaluator state and metadata.
            point=requested_point,
            value=float(
                candidate_transmission.transmission
            ),
            sem=float(
                candidate_transmission.transmission_sem
            ),
            safe=True,
            below_noise_floor=bool(
                candidate_measurement.below_noise_floor
            ),
            metadata={
                "requested_state_id": (
                    candidate.state_id
                ),
                "observed_state_id": (
                    self.working_state.state_id
                ),
                "reference_state_id": (
                    candidate_reference.state_id
                ),
                "current_a": float(
                    candidate_measurement.mean_a
                ),
                "current_sem_a": float(
                    candidate_measurement.sem_a
                ),
                "transmission": float(
                    candidate_transmission.transmission
                ),
                "transmission_sem": float(
                    candidate_transmission.transmission_sem
                ),
            },
        )


def _confirm_primary_rcds_best(
    evaluator: _Cup2PrimaryRCDSEvaluator,
    result: OptimizationResult,
) -> ObjectiveEvaluation:
    """Safely return to and freshly measure the RCDS best point."""

    if result.optimizer_name != "rcds":
        raise ValueError(
            "Cup-2 primary best confirmation requires an RCDS result"
        )

    axis_names = tuple(
        result.metadata.get(
            "axis_names",
            (),
        )
    )

    if axis_names != (
        CUP2_PRIMARY_PARAMETERS
    ):
        raise ValueError(
            "RCDS result axes do not match the Cup-2 primary axes"
        )

    if not result.best_evaluation.safe:
        raise Cup2OptimizationError(
            "RCDS best evaluation is not marked safe"
        )

    best_point = tuple(
        float(value)
        for value
        in result.best_evaluation.point
    )

    if len(
        best_point
    ) != len(
        CUP2_PRIMARY_PARAMETERS
    ):
        raise ValueError(
            "RCDS best point dimension does not match Cup-2"
        )

    confirmation = evaluator(
        best_point
    )

    if not confirmation.safe:
        raise Cup2OptimizationError(
            "Fresh RCDS best-point confirmation was not safe"
        )

    if (
        len(
            confirmation.point
        )
        != len(
            best_point
        )
        or any(
            not _commands_equal(
                actual,
                expected,
            )
            for actual, expected
            in zip(
                confirmation.point,
                best_point,
            )
        )
    ):
        raise Cup2OptimizationError(
            "Best-point confirmation returned the wrong optimizer point"
        )

    physical_point = tuple(
        float(
            evaluator.working_state.parameters[
                parameter_name
            ]
        )
        for parameter_name
        in CUP2_PRIMARY_PARAMETERS
    )

    if any(
        not _commands_equal(
            actual,
            expected,
        )
        for actual, expected
        in zip(
            physical_point,
            best_point,
        )
    ):
        raise Cup2OptimizationError(
            "Physical Cup-2 state did not return to the RCDS best point"
        )

    return confirmation


def _run_primary_rcds(
    adapter,
    working_state: MachineState,
    cup1_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[str, SettlingPolicy],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    optimization_policy: Cup2OptimizationPolicy,
    *,
    rcds_policy: RCDSPolicy | None = None,
    noise_floor_a: float | None = None,
    logger=None,
    maintenance_hook: Callable[
        [MachineState],
        MachineState,
    ] | None = None,
) -> tuple[
    OptimizationResult,
    ObjectiveEvaluation,
    MachineState,
]:
    """Run the isolated primary Cup-2 RCDS optimization."""

    problem = _build_primary_rcds_problem(
        working_state,
        profile,
        comparison_policy,
        optimization_policy,
    )

    evaluator = _Cup2PrimaryRCDSEvaluator(
        adapter=adapter,
        working_state=working_state,
        cup1_reference_state=cup1_reference_state,
        tracker=tracker,
        settling_policies=settling_policies,
        measurement_policy=measurement_policy,
        noise_floor_a=noise_floor_a,
        logger=logger,
        maintenance_hook=maintenance_hook,
    )

    optimizer = RobustConjugateDirectionOptimizer(
        policy=rcds_policy
    )

    result = optimizer.optimize(
        problem,
        evaluator,
    )

    if logger is not None:
        logger.log_optimizer_trace(
            result,
            stage=2,
            cup=2,
        )

    confirmation = _confirm_primary_rcds_best(
        evaluator,
        result,
    )

    return (
        result,
        confirmation,
        evaluator.working_state,
    )


def _log_reference_check(
    logger,
    result: SourceReferenceCheckResult,
) -> None:
    if logger is None:
        return

    logger.log_state_transition(
        result.reference_application
    )

    logger.log_measurement(
        result.measurement,
        cup=1,
        state_id=(
            result.reference.state_id
        ),
        purpose=(
            "periodic_cup1_source_reference"
        ),
    )

    logger.log_reference(
        result.reference
    )

    logger.log_state_transition(
        result.restoration
    )


def _refresh_reference_if_due(
    adapter,
    working_state: MachineState,
    cup1_reference_state: MachineState,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    *,
    noise_floor_a: float | None,
    logger,
    monotonic: Callable[
        [],
        float,
    ],
    utc_now: Callable[
        [],
        str,
    ],
    results: list[
        SourceReferenceCheckResult
    ],
) -> MachineState:
    now = monotonic()

    if not tracker.is_due(
        now
    ):
        return working_state

    check = perform_source_reference_check(
        adapter,
        working_state,
        cup1_reference_state,
        tracker,
        settling_policies,
        measurement_policy,
        noise_floor_a=noise_floor_a,
        monotonic=monotonic,
        utc_now=utc_now,
    )

    results.append(
        check
    )

    _log_reference_check(
        logger,
        check,
    )

    return check.working_state_after


def _update_profile(
    profile: MassProfile,
    state: MachineState,
) -> None:
    """
    Only Cup-2-primary commands are written into the simple global
    best_commands mapping.

    The locally retuned einzel value belongs to the complete Cup-2 state,
    so it is preserved through cup2_best rather than overwriting the
    Cup-1 einzel starting value.
    """

    for parameter_name in (
        CUP2_PRIMARY_PARAMETERS
    ):
        profile.set_best_command(
            parameter_name,
            state.parameters[
                parameter_name
            ],
        )

    profile.set_best_state(
        "cup2_best",
        state.state_id,
    )


def optimize_cup2(
    adapter,
    current_state: MachineState,
    cup1_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    *,
    optimization_policy: (
        Cup2OptimizationPolicy | None
    ) = None,
    primary_rcds_policy: (
        RCDSPolicy | None
    ) = None,
    noise_floor_a: float | None = None,
    logger=None,
    monotonic: Callable[
        [],
        float,
    ] = time.monotonic,
    utc_now: Callable[
        [],
        str,
    ] = utc_now_iso,
) -> Cup2OptimizationResult:
    """
    Optimize transport from the saved Cup-1 reference state to Cup 2.

    Objective:
        T_1->2 = I_2 / I_1,ref

    Frozen throughout Cup 2:
        sputter
        extraction
        analyzing magnet

    Primary controls:
        lens2
        steerer X1
        steerer Y1

    Local upstream correction:
        einzel lens

    Periodic Cup-1 source checks are inserted whenever the reference
    tracker says the reference is due.
    """

    policy = (
        optimization_policy
        if optimization_policy is not None
        else Cup2OptimizationPolicy()
    )

    _validate_inputs(
        current_state,
        cup1_reference_state,
        profile,
        settling_policies,
    )

    working_state = capture_readbacks(
        adapter,
        current_state,
    )

    initial_state = (
        working_state
    )

    reference_checks: list[
        SourceReferenceCheckResult
    ] = []

    working_state = (
        _refresh_reference_if_due(
            adapter,
            working_state,
            cup1_reference_state,
            tracker,
            settling_policies,
            measurement_policy,
            noise_floor_a=(
                noise_floor_a
            ),
            logger=logger,
            monotonic=monotonic,
            utc_now=utc_now,
            results=reference_checks,
        )
    )

    _assert_frozen_cup1_parameters(
        working_state,
        cup1_reference_state,
    )

    if tracker.latest is None:
        raise Cup2OptimizationError(
            "Cup-2 optimization requires a valid Cup-1 source reference"
        )

    if logger is not None:
        logger.log_event(
            "cup2_optimization_started",
            {
                "state_id": (
                    working_state.state_id
                ),
                "cup1_reference_state_id": (
                    cup1_reference_state.state_id
                ),
                "source_reference_state_id": (
                    tracker.latest.state_id
                ),
                "commands": (
                    working_state.parameters
                ),
                "readbacks": (
                    working_state.readbacks
                ),
            },
        )

    scans: list[
        TransmissionScanResult
    ] = []

    scan_definitions = (
        (
            "lens2_voltage_v",
            policy.lens2_half_width_v,
            policy.lens2_scan,
        ),
        (
            "steerer_x1_v",
            policy.steerer_half_width_v,
            policy.steerer_scan,
        ),
        (
            "steerer_y1_v",
            policy.steerer_half_width_v,
            policy.steerer_scan,
        ),
        (
            "einzel_lens_voltage_v",
            policy.einzel_half_width_v,
            policy.einzel_scan,
        ),
    )

    def maintenance_hook(
        physical_state: MachineState,
    ) -> MachineState:
        refreshed = (
            _refresh_reference_if_due(
                adapter,
                physical_state,
                cup1_reference_state,
                tracker,
                settling_policies,
                measurement_policy,
                noise_floor_a=(
                    noise_floor_a
                ),
                logger=logger,
                monotonic=monotonic,
                utc_now=utc_now,
                results=(
                    reference_checks
                ),
            )
        )

        _assert_frozen_cup1_parameters(
            refreshed,
            cup1_reference_state,
        )

        return refreshed

    primary_optimization = None
    primary_confirmation = None

    coordinate_passes = (
        policy.coordinate_passes
    )

    if primary_rcds_policy is not None:
        (
            primary_optimization,
            primary_confirmation,
            working_state,
        ) = _run_primary_rcds(
            adapter,
            working_state,
            cup1_reference_state,
            profile,
            tracker,
            settling_policies,
            measurement_policy,
            comparison_policy,
            policy,
            rcds_policy=(
                primary_rcds_policy
            ),
            noise_floor_a=(
                noise_floor_a
            ),
            logger=logger,
            maintenance_hook=(
                maintenance_hook
            ),
        )

        _assert_frozen_cup1_parameters(
            working_state,
            cup1_reference_state,
        )

        # The three primary controls have already been handled
        # jointly by RCDS. Keep only the existing narrow local
        # upstream einzel correction.
        scan_definitions = (
            (
                "einzel_lens_voltage_v",
                policy.einzel_half_width_v,
                policy.einzel_scan,
            ),
        )

        coordinate_passes = 1

    for pass_index in range(
        1,
        coordinate_passes + 1,
    ):
        if logger is not None:
            logger.log_event(
                "cup2_coordinate_pass_started",
                {
                    "pass": (
                        pass_index
                    ),
                    "state_id": (
                        working_state.state_id
                    ),
                    "source_reference_state_id": (
                        tracker.latest.state_id
                    ),
                },
            )

        for (
            parameter_name,
            half_width,
            scan_policy,
        ) in scan_definitions:
            # A long preceding scan might have made the reference due
            # before the next scan's baseline measurement.
            working_state = (
                maintenance_hook(
                    working_state
                )
            )

            center = float(
                working_state.parameters[
                    parameter_name
                ]
            )

            local_profile = (
                _local_profile(
                    profile,
                    parameter_name,
                    center,
                    half_width,
                )
            )

            scan = (
                scan_parameter_transmission_1d(
                    adapter,
                    working_state,
                    local_profile,
                    tracker,
                    parameter_name,
                    scan_policy,
                    settling_policies,
                    measurement_policy,
                    comparison_policy,
                    noise_floor_a=(
                        noise_floor_a
                    ),
                    logger=logger,
                    maintenance_hook=(
                        maintenance_hook
                    ),
                )
            )

            working_state = (
                scan.final_state
            )

            scans.append(
                scan
            )

            _assert_frozen_cup1_parameters(
                working_state,
                cup1_reference_state,
            )

    # Ensure the final characterization does not use an expired source
    # reference.
    working_state = (
        maintenance_hook(
            working_state
        )
    )

    working_state = (
        capture_readbacks(
            adapter,
            working_state,
        )
    )

    _assert_frozen_cup1_parameters(
        working_state,
        cup1_reference_state,
    )

    final_state = (
        _retag_final_state(
            working_state
        )
    )

    final_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
        )
    )

    if (
        final_measurement.below_noise_floor
        or final_measurement.mean_a <= 0
    ):
        raise Cup2OptimizationNoBeamError(
            "Final Cup-2 beam current is not a valid transport signal"
        )

    final_reference = (
        tracker.latest
    )

    if final_reference is None:
        raise Cup2OptimizationError(
            "Cup-1 reference disappeared during Cup-2 optimization"
        )

    final_transmission = (
        transmission_from_reference(
            2,
            final_measurement,
            final_reference,
        )
    )

    _update_profile(
        profile,
        final_state,
    )

    if logger is not None:
        logger.save_state(
            final_state,
            "cup2_best",
        )

        logger.log_measurement(
            final_measurement,
            cup=2,
            state_id=(
                final_state.state_id
            ),
            purpose=(
                "cup2_final"
            ),
        )

        logger.log_transmission(
            final_transmission
        )

        logger.log_event(
            "cup2_optimization_completed",
            {
                "state_id": (
                    final_state.state_id
                ),
                "current_a": (
                    final_measurement.mean_a
                ),
                "transmission": (
                    final_transmission.transmission
                ),
                "transmission_percent": (
                    final_transmission.transmission_percent
                ),
                "reference_state_id": (
                    final_reference.state_id
                ),
                "reference_checks": (
                    len(reference_checks)
                ),
                "scan_count": (
                    len(scans)
                ),
                "commands": (
                    final_state.parameters
                ),
                "readbacks": (
                    final_state.readbacks
                ),
            },
        )

    return Cup2OptimizationResult(
        initial_state=initial_state,
        scans=tuple(
            scans
        ),
        reference_checks=tuple(
            reference_checks
        ),
        final_state=final_state,
        final_measurement=(
            final_measurement
        ),
        final_reference=(
            final_reference
        ),
        final_transmission=(
            final_transmission
        ),
        primary_optimization=(
            primary_optimization
        ),
        primary_confirmation=(
            primary_confirmation
        ),
    )
