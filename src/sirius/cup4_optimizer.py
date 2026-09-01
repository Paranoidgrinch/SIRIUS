from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

from sirius.comparison import ComparisonPolicy
from sirius.optimizer_api import (
    ObjectiveEvaluation,
    OptimizationAxis,
    OptimizationProblem,
    OptimizationResult,
    comparison_policy_comparator,
)
from sirius.rcds_optimizer import (
    RCDSPolicy,
    RobustConjugateDirectionOptimizer,
)
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.parameters import PARAMETERS
from sirius.qpt_model import (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
    evaluate_qpt,
    qpt_cfa_is_feasible,
    qpt_commands_from_cfa,
)
from sirius.qpt_scan2d import (
    QPT2DScanPolicy,
    QPT2DScanResult,
    QPTScanLevel,
    scan_qpt_focus_asymmetry_2d,
)
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
from sirius.run_logging import RunLogger
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


CUP4_QPT_PARAMETERS = (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
)

CUP4_STEERER_PARAMETERS = (
    "steerer_x2_v",
    "steerer_y2_v",
)

CUP4_PRIMARY_PARAMETERS = (
    *CUP4_QPT_PARAMETERS,
    *CUP4_STEERER_PARAMETERS,
)


CUP4_RCDS_AXIS_NAMES = (
    "qpt_global_focus_v",
    "qpt_asymmetry_v",
    *CUP4_STEERER_PARAMETERS,
)

# Everything established upstream of the QPT is frozen during Cup 4.
CUP4_FROZEN_UPSTREAM_PARAMETERS = (
    "sputter_voltage_v",
    "extraction_voltage_v",
    "einzel_lens_voltage_v",
    "magnet_current_a",

    "lens2_voltage_v",
    "steerer_x1_v",
    "steerer_y1_v",

    "ion_cooler_voltage_v",
    "deceleration_voltage_v",
    "acceleration_voltage_v",

    "guidefield1_voltage_v",
    "guidefield2_voltage_v",
)


class Cup4OptimizationError(RuntimeError):
    pass


class Cup4OptimizationNoBeamError(
    Cup4OptimizationError
):
    pass


@dataclass(frozen=True)
class Cup4OptimizationPolicy:
    """
    Cup-4 optimization policy.

    The QPT optical degrees of freedom are F and A.

    QPT2 defines the common mode C and is frozen throughout the complete
    Cup-4 optimization.
    """

    qpt_scan: QPT2DScanPolicy = field(
        default_factory=QPT2DScanPolicy
    )

    steerer_half_width_v: float = 100.0

    steerer_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                25.0,
                5.0,
            )
        )
    )

    steerer_passes: int = 2

    final_qpt_scan: QPT2DScanPolicy = field(
        default_factory=lambda: QPT2DScanPolicy(
            initial_focus_half_width_v=150.0,
            initial_asymmetry_half_width_v=150.0,
            levels=(
                QPTScanLevel(
                    focus_step_v=50.0,
                    asymmetry_step_v=50.0,
                ),
                QPTScanLevel(
                    focus_step_v=10.0,
                    asymmetry_step_v=10.0,
                ),
            ),
            refinement_half_width_factor=2.0,
            max_points_per_level=500,
        )
    )

    def __post_init__(self) -> None:
        if not math.isfinite(
            float(
                self.steerer_half_width_v
            )
        ):
            raise ValueError(
                "steerer_half_width_v must be finite"
            )

        if self.steerer_half_width_v <= 0:
            raise ValueError(
                "steerer_half_width_v must be greater than zero"
            )

        if self.steerer_passes < 1:
            raise ValueError(
                "steerer_passes must be at least 1"
            )


@dataclass(frozen=True)
class Cup4OptimizationResult:
    initial_state: MachineState

    initial_qpt_scan: (
        QPT2DScanResult | None
    )

    steerer_scans: tuple[
        TransmissionScanResult,
        ...
    ]

    final_qpt_scan: (
        QPT2DScanResult | None
    )

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


def _validate_inputs(
    current_state: MachineState,
    cup3_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> None:
    current_state.validate()
    cup3_reference_state.validate()
    profile.validate()

    if current_state.cup != 4:
        raise ValueError(
            "Cup-4 optimization requires Cup 4"
        )

    if current_state.stage not in (
        None,
        4,
    ):
        raise ValueError(
            "Cup-4 optimization requires stage 4 or no stage assignment"
        )

    if cup3_reference_state.cup != 3:
        raise ValueError(
            "Saved Cup-3 reference state must select Cup 3"
        )

    if not math.isclose(
        current_state.mass_u,
        cup3_reference_state.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Cup-3 and Cup-4 states use different ion masses"
        )

    if not math.isclose(
        current_state.mass_u,
        profile.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Machine state and mass profile use different ion masses"
        )

    for parameter_name in (
        CUP4_PRIMARY_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"Cup-4 state is missing {parameter_name}"
            )

    for parameter_name in (
        CUP4_FROZEN_UPSTREAM_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"Cup-4 state is missing upstream parameter "
                f"{parameter_name}"
            )

        if (
            parameter_name
            not in cup3_reference_state.parameters
        ):
            raise ValueError(
                f"Cup-3 reference is missing {parameter_name}"
            )

        if not _commands_equal(
            current_state.parameters[
                parameter_name
            ],
            cup3_reference_state.parameters[
                parameter_name
            ],
        ):
            raise ValueError(
                f"{parameter_name} must match the Cup-3 reference"
            )

    # RFQ configuration is part of the frozen Cup-3 transport solution.
    if (
        current_state.rfq
        != cup3_reference_state.rfq
    ):
        raise ValueError(
            "Cup-4 RFQ state must match the Cup-3 reference"
        )

    for parameter_name in (
        CUP4_PRIMARY_PARAMETERS
    ):
        if (
            parameter_name
            not in settling_policies
        ):
            raise KeyError(
                f"No settling policy configured for {parameter_name}"
            )

    if tracker.latest is None:
        raise ValueError(
            "Cup-4 optimization requires an existing Cup-1 reference"
        )


def _assert_upstream_frozen(
    state: MachineState,
    cup3_reference_state: MachineState,
) -> None:
    for parameter_name in (
        CUP4_FROZEN_UPSTREAM_PARAMETERS
    ):
        actual = state.parameters[
            parameter_name
        ]

        expected = (
            cup3_reference_state.parameters[
                parameter_name
            ]
        )

        if not _commands_equal(
            actual,
            expected,
        ):
            raise Cup4OptimizationError(
                f"{parameter_name} changed during Cup-4 optimization"
            )

    if state.rfq != cup3_reference_state.rfq:
        raise Cup4OptimizationError(
            "RFQ configuration changed during Cup-4 optimization"
        )


def _assert_qpt_common_frozen(
    state: MachineState,
    common_v: float,
) -> None:
    current_common = (
        evaluate_qpt(
            state
        ).command_coordinates.common_v
    )

    if not _commands_equal(
        current_common,
        common_v,
    ):
        raise Cup4OptimizationError(
            "QPT common mode changed during Cup-4 optimization"
        )


def _local_profile(
    profile: MassProfile,
    parameter_name: str,
    center: float,
    half_width: float,
) -> MassProfile:
    """
    Build a temporary learned-range view for a local steerer scan without
    mutating the persistent mass profile.
    """

    local = deepcopy(
        profile
    )

    definition = PARAMETERS[
        parameter_name
    ]

    learned_minimum, learned_maximum = (
        profile.effective_bounds(
            parameter_name
        )
    )

    allowed_minimum = max(
        float(
            definition.minimum
        ),
        min(
            float(center),
            float(
                learned_minimum
            ),
        ),
    )

    allowed_maximum = min(
        float(
            definition.maximum
        ),
        max(
            float(center),
            float(
                learned_maximum
            ),
        ),
    )

    minimum = max(
        allowed_minimum,
        float(center)
        - float(half_width),
    )

    maximum = min(
        allowed_maximum,
        float(center)
        + float(half_width),
    )

    if maximum <= minimum:
        raise ValueError(
            f"Local scan window collapsed for {parameter_name}"
        )

    local.set_learned_range(
        parameter_name,
        minimum,
        maximum,
        source="cup4_local_window",
    )

    return local


def _build_primary_rcds_problem(
    current_state: MachineState,
    profile: MassProfile,
    comparison_policy: ComparisonPolicy,
    optimization_policy: Cup4OptimizationPolicy,
) -> OptimizationProblem:
    """
    Build the local four-dimensional Cup-4 RCDS geometry.

    Optimizer coordinates:
        F, A, X2, Y2

    QPT common mode C is frozen to the current command value.
    """

    current_state.validate()
    profile.validate()

    if not math.isclose(
        current_state.mass_u,
        profile.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Machine state and mass profile use different ion masses"
        )

    if current_state.cup != 4:
        raise ValueError(
            "Cup-4 RCDS problem requires Cup 4"
        )

    if current_state.stage not in (
        None,
        4,
    ):
        raise ValueError(
            "Cup-4 RCDS problem requires stage 4 "
            "or no stage assignment"
        )

    for parameter_name in (
        CUP4_STEERER_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                "Cup-4 state is missing "
                f"{parameter_name}"
            )

    coordinates = (
        evaluate_qpt(
            current_state
        ).command_coordinates
    )

    frozen_common_v = float(
        coordinates.common_v
    )

    initial_focus_v = float(
        coordinates.global_focus_v
    )

    initial_asymmetry_v = float(
        coordinates.asymmetry_v
    )

    qpt_policy = (
        optimization_policy.qpt_scan
    )

    axes: list[
        OptimizationAxis
    ] = [
        OptimizationAxis(
            name=(
                CUP4_RCDS_AXIS_NAMES[
                    0
                ]
            ),
            minimum=(
                initial_focus_v
                - float(
                    qpt_policy
                    .initial_focus_half_width_v
                )
            ),
            maximum=(
                initial_focus_v
                + float(
                    qpt_policy
                    .initial_focus_half_width_v
                )
            ),
        ),
        OptimizationAxis(
            name=(
                CUP4_RCDS_AXIS_NAMES[
                    1
                ]
            ),
            minimum=(
                initial_asymmetry_v
                - float(
                    qpt_policy
                    .initial_asymmetry_half_width_v
                )
            ),
            maximum=(
                initial_asymmetry_v
                + float(
                    qpt_policy
                    .initial_asymmetry_half_width_v
                )
            ),
        ),
    ]

    initial_point: list[
        float
    ] = [
        initial_focus_v,
        initial_asymmetry_v,
    ]

    for parameter_name in (
        CUP4_STEERER_PARAMETERS
    ):
        center = float(
            current_state.parameters[
                parameter_name
            ]
        )

        local_profile = _local_profile(
            profile,
            parameter_name,
            center,
            optimization_policy
            .steerer_half_width_v,
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
            raise Cup4OptimizationError(
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

    def qpt_feasible(
        point: tuple[
            float,
            ...
        ],
    ) -> bool:
        if len(
            point
        ) != 4:
            return False

        return qpt_cfa_is_feasible(
            frozen_common_v,
            float(
                point[
                    0
                ]
            ),
            float(
                point[
                    1
                ]
            ),
        )

    return OptimizationProblem(
        axes=tuple(
            axes
        ),
        initial_point=tuple(
            initial_point
        ),
        maximize=True,
        safety_predicate=(
            qpt_feasible
        ),
        comparison=(
            comparison_policy_comparator(
                policy=(
                    comparison_policy
                )
            )
        ),
    )


@dataclass
class _Cup4PrimaryRCDSEvaluator:
    """
    Stateful bridge from one reduced Cup-4 RCDS point
    (F, A, X2, Y2) to one real transmission measurement.

    QPT common mode C and the complete upstream Cup-3 solution
    remain frozen. Hardware execution remains exclusively behind
    sirius.safe_transition.apply_state().
    """

    adapter: object

    working_state: MachineState
    cup3_reference_state: MachineState

    tracker: SourceReferenceTracker

    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ]

    measurement_policy: MeasurementPolicy

    frozen_common_v: float

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
        self.cup3_reference_state.validate()

        self.frozen_common_v = float(
            self.frozen_common_v
        )

        if not math.isfinite(
            self.frozen_common_v
        ):
            raise ValueError(
                "Cup-4 frozen QPT common mode must be finite"
            )

        if self.working_state.cup != 4:
            raise ValueError(
                "Cup-4 RCDS evaluator requires Cup 4"
            )

        if self.working_state.stage not in (
            None,
            4,
        ):
            raise ValueError(
                "Cup-4 RCDS evaluator requires stage 4 "
                "or no stage assignment"
            )

        if self.cup3_reference_state.cup != 3:
            raise ValueError(
                "Cup-4 RCDS evaluator requires a Cup-3 "
                "reference state"
            )

        if not math.isclose(
            float(
                self.working_state.mass_u
            ),
            float(
                self.cup3_reference_state.mass_u
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Cup-3 reference and Cup-4 working state "
                "use different ion masses"
            )

        _assert_upstream_frozen(
            self.working_state,
            self.cup3_reference_state,
        )

        _assert_qpt_common_frozen(
            self.working_state,
            self.frozen_common_v,
        )

        for parameter_name in (
            CUP4_PRIMARY_PARAMETERS
        ):
            if (
                parameter_name
                not in self.working_state.parameters
            ):
                raise ValueError(
                    "Cup-4 state is missing "
                    f"{parameter_name}"
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
            CUP4_RCDS_AXIS_NAMES
        ):
            raise ValueError(
                "Cup-4 primary RCDS point must contain "
                "F, A, X2 and Y2"
            )

        requested_point = tuple(
            float(
                value
            )
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
                "Cup-4 RCDS point must contain only finite values"
            )

        if (
            self.maintenance_hook
            is not None
        ):
            refreshed = (
                self.maintenance_hook(
                    self.working_state
                )
            )

            refreshed.validate()

            if not math.isclose(
                float(
                    refreshed.mass_u
                ),
                float(
                    self.working_state.mass_u
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise Cup4OptimizationError(
                    "RCDS maintenance hook changed ion mass"
                )

            if refreshed.cup != 4:
                raise Cup4OptimizationError(
                    "RCDS maintenance hook did not restore Cup 4"
                )

            if refreshed.stage not in (
                None,
                4,
            ):
                raise Cup4OptimizationError(
                    "RCDS maintenance hook changed optimization stage"
                )

            self.working_state = (
                refreshed
            )

        _assert_upstream_frozen(
            self.working_state,
            self.cup3_reference_state,
        )

        _assert_qpt_common_frozen(
            self.working_state,
            self.frozen_common_v,
        )

        (
            requested_focus_v,
            requested_asymmetry_v,
            requested_x2_v,
            requested_y2_v,
        ) = requested_point

        qpt_commands = (
            qpt_commands_from_cfa(
                self.frozen_common_v,
                requested_focus_v,
                requested_asymmetry_v,
            )
        )

        parameters = dict(
            self.working_state.parameters
        )

        parameters.update(
            qpt_commands.parameters
        )

        parameters[
            CUP4_STEERER_PARAMETERS[
                0
            ]
        ] = requested_x2_v

        parameters[
            CUP4_STEERER_PARAMETERS[
                1
            ]
        ] = requested_y2_v

        readbacks = dict(
            self.working_state.readbacks
        )

        # No readback acquired before this candidate may be
        # reused as verification of a newly requested command.
        for parameter_name in (
            CUP4_PRIMARY_PARAMETERS
        ):
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
            cup=4,
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
                    CUP4_RCDS_AXIS_NAMES
                ),
                "objective": (
                    "cup1_normalized_transmission"
                ),
                "qpt_common_command_v": (
                    self.frozen_common_v
                ),
                "qpt_focus_command_v": (
                    requested_focus_v
                ),
                "qpt_asymmetry_command_v": (
                    requested_asymmetry_v
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

        if observed.cup != 4:
            raise Cup4OptimizationError(
                "Safe transition did not preserve Cup 4"
            )

        if observed.stage not in (
            None,
            4,
        ):
            raise Cup4OptimizationError(
                "Safe transition changed Cup-4 optimization stage"
            )

        if not math.isclose(
            float(
                observed.mass_u
            ),
            float(
                self.working_state.mass_u
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise Cup4OptimizationError(
                "Safe transition changed ion mass"
            )

        self.working_state = (
            observed
        )

        _assert_upstream_frozen(
            self.working_state,
            self.cup3_reference_state,
        )

        _assert_qpt_common_frozen(
            self.working_state,
            self.frozen_common_v,
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
            raise Cup4OptimizationError(
                "Cup-4 RCDS evaluation requires "
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
            raise Cup4OptimizationError(
                "Cup-1 source reference and Cup-4 state "
                "use different ion masses"
            )

        candidate_transmission = (
            transmission_from_reference(
                4,
                candidate_measurement,
                candidate_reference,
            )
        )

        if self.logger is not None:
            self.logger.log_measurement(
                candidate_measurement,
                cup=4,
                state_id=(
                    self.working_state.state_id
                ),
                purpose="cup4_rcds_candidate",
            )

            self.logger.log_transmission(
                candidate_transmission
            )

        return ObjectiveEvaluation(
            # RCDS requires the optimizer-space point here.
            # The actual observed physical state is retained
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
                "qpt_common_v": (
                    self.frozen_common_v
                ),
                "qpt_focus_v": (
                    requested_focus_v
                ),
                "qpt_asymmetry_v": (
                    requested_asymmetry_v
                ),
            },
        )


def _confirm_primary_rcds_best(
    evaluator: _Cup4PrimaryRCDSEvaluator,
    result: OptimizationResult,
) -> ObjectiveEvaluation:
    """Safely return to and freshly measure the Cup-4 RCDS best point."""

    if result.optimizer_name != "rcds":
        raise ValueError(
            "Cup-4 primary best confirmation requires an RCDS result"
        )

    axis_names = tuple(
        result.metadata.get(
            "axis_names",
            (),
        )
    )

    if axis_names != (
        CUP4_RCDS_AXIS_NAMES
    ):
        raise ValueError(
            "RCDS result axes do not match the Cup-4 primary axes"
        )

    if not result.best_evaluation.safe:
        raise Cup4OptimizationError(
            "RCDS best evaluation is not marked safe"
        )

    best_point = tuple(
        float(
            value
        )
        for value
        in result.best_evaluation.point
    )

    if len(
        best_point
    ) != len(
        CUP4_RCDS_AXIS_NAMES
    ):
        raise ValueError(
            "RCDS best point dimension does not match Cup 4"
        )

    confirmation = evaluator(
        best_point
    )

    if not confirmation.safe:
        raise Cup4OptimizationError(
            "Fresh Cup-4 RCDS best-point confirmation was not safe"
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
            for (
                actual,
                expected,
            )
            in zip(
                confirmation.point,
                best_point,
            )
        )
    ):
        raise Cup4OptimizationError(
            "Best-point confirmation returned the wrong optimizer point"
        )

    coordinates = (
        evaluate_qpt(
            evaluator.working_state
        ).command_coordinates
    )

    physical_point = (
        float(
            coordinates.global_focus_v
        ),
        float(
            coordinates.asymmetry_v
        ),
        float(
            evaluator.working_state.parameters[
                CUP4_STEERER_PARAMETERS[
                    0
                ]
            ]
        ),
        float(
            evaluator.working_state.parameters[
                CUP4_STEERER_PARAMETERS[
                    1
                ]
            ]
        ),
    )

    if any(
        not _commands_equal(
            actual,
            expected,
        )
        for (
            actual,
            expected,
        )
        in zip(
            physical_point,
            best_point,
        )
    ):
        raise Cup4OptimizationError(
            "Physical Cup-4 state did not return to the RCDS best point"
        )

    _assert_upstream_frozen(
        evaluator.working_state,
        evaluator.cup3_reference_state,
    )

    _assert_qpt_common_frozen(
        evaluator.working_state,
        evaluator.frozen_common_v,
    )

    return confirmation


def _run_primary_rcds(
    adapter,
    working_state: MachineState,
    cup3_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    optimization_policy: Cup4OptimizationPolicy,
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
    """Run the isolated primary Cup-4 RCDS optimization."""

    problem = _build_primary_rcds_problem(
        working_state,
        profile,
        comparison_policy,
        optimization_policy,
    )

    frozen_common_v = float(
        evaluate_qpt(
            working_state
        ).command_coordinates.common_v
    )

    evaluator = _Cup4PrimaryRCDSEvaluator(
        adapter=adapter,
        working_state=working_state,
        cup3_reference_state=(
            cup3_reference_state
        ),
        tracker=tracker,
        settling_policies=(
            settling_policies
        ),
        measurement_policy=(
            measurement_policy
        ),
        frozen_common_v=(
            frozen_common_v
        ),
        noise_floor_a=(
            noise_floor_a
        ),
        logger=logger,
        maintenance_hook=(
            maintenance_hook
        ),
    )

    optimizer = (
        RobustConjugateDirectionOptimizer(
            policy=rcds_policy
        )
    )

    result = optimizer.optimize(
        problem,
        evaluator,
    )

    if logger is not None:
        logger.log_optimizer_trace(
            result,
            stage=4,
            cup=4,
        )

    confirmation = (
        _confirm_primary_rcds_best(
            evaluator,
            result,
        )
    )

    return (
        result,
        confirmation,
        evaluator.working_state,
    )


def _log_reference_check(
    logger: RunLogger | None,
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
        noise_floor_a=(
            noise_floor_a
        ),
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

    return (
        check.working_state_after
    )


def _make_final_state(
    state: MachineState,
) -> MachineState:
    qpt = evaluate_qpt(
        state
    )

    result = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            state.parameters
        ),
        readbacks=dict(
            state.readbacks
        ),
        cup=4,
        stage=4,
        role="stage_best",
        rfq=deepcopy(
            state.rfq
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata={
            **deepcopy(
                state.metadata
            ),
            "optimized_stage": 4,
            "objective": (
                "cup1_normalized_transmission"
            ),
            "qpt_common_command_v": (
                qpt.command_coordinates.common_v
            ),
            "qpt_focus_command_v": (
                qpt.command_coordinates.global_focus_v
            ),
            "qpt_asymmetry_command_v": (
                qpt.command_coordinates.asymmetry_v
            ),
            "qpt_outer_strength_command_v": (
                qpt.command_coordinates.outer_strength_v
            ),
            "qpt_middle_strength_command_v": (
                qpt.command_coordinates.middle_strength_v
            ),
            "qpt_common_observed_v": (
                qpt.best_available_coordinates.common_v
            ),
            "qpt_focus_observed_v": (
                qpt.best_available_coordinates.global_focus_v
            ),
            "qpt_asymmetry_observed_v": (
                qpt.best_available_coordinates.asymmetry_v
            ),
        },
    )

    result.validate()

    return result


def _update_profile(
    profile: MassProfile,
    final_state: MachineState,
) -> None:
    for parameter_name in (
        CUP4_PRIMARY_PARAMETERS
    ):
        profile.set_best_command(
            parameter_name,
            final_state.parameters[
                parameter_name
            ],
        )

    profile.set_best_state(
        "cup4_best",
        final_state.state_id,
    )

    qpt = evaluate_qpt(
        final_state
    )

    profile.metadata[
        "cup4_qpt"
    ] = {
        "common_command_v": (
            qpt.command_coordinates.common_v
        ),
        "focus_command_v": (
            qpt.command_coordinates.global_focus_v
        ),
        "asymmetry_command_v": (
            qpt.command_coordinates.asymmetry_v
        ),
        "outer_strength_command_v": (
            qpt.command_coordinates.outer_strength_v
        ),
        "middle_strength_command_v": (
            qpt.command_coordinates.middle_strength_v
        ),
        "common_observed_v": (
            qpt.best_available_coordinates.common_v
        ),
        "focus_observed_v": (
            qpt.best_available_coordinates.global_focus_v
        ),
        "asymmetry_observed_v": (
            qpt.best_available_coordinates.asymmetry_v
        ),
    }


def optimize_cup4(
    adapter,
    current_state: MachineState,
    cup3_reference_state: MachineState,
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
        Cup4OptimizationPolicy | None
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
) -> Cup4OptimizationResult:
    """
    Optimize transport from the frozen Cup-3 solution to Cup 4.

    Optical degrees of freedom:

        QPT:
            F = global triplet focus
            A = middle-vs-outer balance

        Steering:
            X2
            Y2

    QPT common mode C and all upstream Cup-1/Cup-2/Cup-3 parameters
    remain frozen.
    """

    policy = (
        optimization_policy
        if optimization_policy is not None
        else Cup4OptimizationPolicy()
    )

    _validate_inputs(
        current_state,
        cup3_reference_state,
        profile,
        tracker,
        settling_policies,
    )

    working_state = (
        capture_readbacks(
            adapter,
            current_state,
        )
    )

    initial_state = (
        working_state
    )

    _assert_upstream_frozen(
        working_state,
        cup3_reference_state,
    )

    initial_qpt = evaluate_qpt(
        working_state
    )

    frozen_common_v = float(
        initial_qpt.command_coordinates.common_v
    )

    reference_checks: list[
        SourceReferenceCheckResult
    ] = []

    if logger is not None:
        logger.log_event(
            "cup4_optimization_started",
            {
                "state_id": (
                    initial_state.state_id
                ),
                "mass_u": (
                    initial_state.mass_u
                ),
                "qpt_common_v": (
                    frozen_common_v
                ),
                "initial_qpt": (
                    initial_qpt
                ),
            },
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

        if refreshed.cup != 4:
            raise Cup4OptimizationError(
                "Source-reference maintenance did not restore Cup 4"
            )

        _assert_upstream_frozen(
            refreshed,
            cup3_reference_state,
        )

        _assert_qpt_common_frozen(
            refreshed,
            frozen_common_v,
        )

        return refreshed

    # Fresh source reference before beginning the downstream optimization
    # if the existing one has expired.
    working_state = maintenance_hook(
        working_state
    )

    # --------------------------------------------------------------
    # Phase A: coarse-to-fine QPT F/A search.
    # --------------------------------------------------------------

    primary_optimization = None
    primary_confirmation = None

    if primary_rcds_policy is not None:
        (
            primary_optimization,
            primary_confirmation,
            working_state,
        ) = _run_primary_rcds(
            adapter,
            working_state,
            cup3_reference_state,
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

        _assert_upstream_frozen(
            working_state,
            cup3_reference_state,
        )

        _assert_qpt_common_frozen(
            working_state,
            frozen_common_v,
        )

        initial_qpt_scan = None
        steerer_scans = []
        final_qpt_scan = None

    else:
        initial_qpt_scan = (
            scan_qpt_focus_asymmetry_2d(
                adapter,
                working_state,
                tracker,
                policy.qpt_scan,
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
            initial_qpt_scan.final_state
        )

        _assert_upstream_frozen(
            working_state,
            cup3_reference_state,
        )

        _assert_qpt_common_frozen(
            working_state,
            frozen_common_v,
        )

        # --------------------------------------------------------------
        # Phase B: X2/Y2 local coordinate descent.
        # --------------------------------------------------------------

        steerer_scans: list[
            TransmissionScanResult
        ] = []

        for _ in range(
            policy.steerer_passes
        ):
            for parameter_name in (
                CUP4_STEERER_PARAMETERS
            ):
                working_state = maintenance_hook(
                    working_state
                )

                local_profile = _local_profile(
                    profile,
                    parameter_name,
                    working_state.parameters[
                        parameter_name
                    ],
                    policy.steerer_half_width_v,
                )

                scan = (
                    scan_parameter_transmission_1d(
                        adapter,
                        working_state,
                        local_profile,
                        tracker,
                        parameter_name,
                        policy.steerer_scan,
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

                steerer_scans.append(
                    scan
                )

                _assert_upstream_frozen(
                    working_state,
                    cup3_reference_state,
                )

                _assert_qpt_common_frozen(
                    working_state,
                    frozen_common_v,
                )

        # --------------------------------------------------------------
        # Phase C: final local F/A refinement after steering.
        # --------------------------------------------------------------

        working_state = maintenance_hook(
            working_state
        )

        final_qpt_scan = (
            scan_qpt_focus_asymmetry_2d(
                adapter,
                working_state,
                tracker,
                policy.final_qpt_scan,
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
            final_qpt_scan.final_state
        )

        _assert_upstream_frozen(
            working_state,
            cup3_reference_state,
        )

        _assert_qpt_common_frozen(
            working_state,
            frozen_common_v,
        )

        # --------------------------------------------------------------
        # Phase D: final source normalization and characterization.
        # --------------------------------------------------------------

    working_state = maintenance_hook(
        working_state
    )

    working_state = capture_readbacks(
        adapter,
        working_state,
    )

    _assert_upstream_frozen(
        working_state,
        cup3_reference_state,
    )

    _assert_qpt_common_frozen(
        working_state,
        frozen_common_v,
    )

    final_state = _make_final_state(
        working_state
    )

    final_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=(
                noise_floor_a
            ),
        )
    )

    if (
        final_measurement.below_noise_floor
        or final_measurement.mean_a <= 0
    ):
        raise Cup4OptimizationNoBeamError(
            "Final Cup-4 current is not a valid transport signal"
        )

    final_reference = tracker.latest

    if final_reference is None:
        raise Cup4OptimizationError(
            "Cup-1 source reference disappeared during Cup-4 optimization"
        )

    final_transmission = (
        transmission_from_reference(
            4,
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
            "cup4_best",
        )

        logger.log_measurement(
            final_measurement,
            cup=4,
            state_id=(
                final_state.state_id
            ),
            purpose="cup4_final",
        )

        logger.log_transmission(
            final_transmission
        )

        logger.log_event(
            "cup4_optimization_completed",
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
                "qpt": (
                    evaluate_qpt(
                        final_state
                    )
                ),
                "steerer_x2_v": (
                    final_state.parameters[
                        "steerer_x2_v"
                    ]
                ),
                "steerer_y2_v": (
                    final_state.parameters[
                        "steerer_y2_v"
                    ]
                ),
                "reference_checks": (
                    len(
                        reference_checks
                    )
                ),
            },
        )

    return Cup4OptimizationResult(
        initial_state=(
            initial_state
        ),
        initial_qpt_scan=(
            initial_qpt_scan
        ),
        steerer_scans=tuple(
            steerer_scans
        ),
        final_qpt_scan=(
            final_qpt_scan
        ),
        reference_checks=tuple(
            reference_checks
        ),
        final_state=(
            final_state
        ),
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
