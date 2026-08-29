from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

from sirius.comparison import ComparisonPolicy
from sirius.esa_model import (
    ESAVoltagePrediction,
    evaluate_esa,
    predict_esa_voltage,
    require_valid_esa_prediction,
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
from sirius.scan1d import ScanPolicy
from sirius.settling import SettlingPolicy
from sirius.state import MachineState, utc_now_iso
from sirius.transition import (
    AppliedStateResult,
    capture_readbacks,
)
from sirius.safe_transition import apply_state
from sirius.transmission_scan1d import (
    TransmissionScanResult,
    scan_parameter_transmission_1d,
)


ESA_PARAMETER = "esa_voltage_v"

CUP5_QPT_PARAMETERS = (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
)

CUP5_LOCAL_RETUNE_PARAMETERS = (
    *CUP5_QPT_PARAMETERS,
    "steerer_x2_v",
    "steerer_y2_v",
)

# Everything upstream of the Cup-4 transport section remains hard-frozen.
CUP5_FROZEN_UPSTREAM_PARAMETERS = (
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

CUP5_REQUIRED_PARAMETERS = (
    *CUP5_FROZEN_UPSTREAM_PARAMETERS,
    *CUP5_LOCAL_RETUNE_PARAMETERS,
    ESA_PARAMETER,
)


class Cup5OptimizationError(RuntimeError):
    pass


class Cup5OptimizationNoBeamError(Cup5OptimizationError):
    pass


@dataclass(frozen=True)
class ESASeedApplication:
    prediction: ESAVoltagePrediction
    requested_voltage_v: float
    state_before: MachineState
    transition: AppliedStateResult
    state_after: MachineState


@dataclass(frozen=True)
class Cup5OptimizationPolicy:
    esa_energy_per_volt: float = 10.0

    initial_esa_half_width_v: float = 400.0
    initial_esa_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(100.0, 25.0, 5.0)
        )
    )

    steerer_half_width_v: float = 60.0
    steerer_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(20.0, 5.0, 1.0)
        )
    )
    steerer_passes: int = 1

    local_qpt_scan: QPT2DScanPolicy = field(
        default_factory=lambda: QPT2DScanPolicy(
            initial_focus_half_width_v=250.0,
            initial_asymmetry_half_width_v=250.0,
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

    final_esa_half_width_v: float = 100.0
    final_esa_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(25.0, 5.0, 1.0)
        )
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("esa_energy_per_volt", self.esa_energy_per_volt),
            ("initial_esa_half_width_v", self.initial_esa_half_width_v),
            ("steerer_half_width_v", self.steerer_half_width_v),
            ("final_esa_half_width_v", self.final_esa_half_width_v),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        if self.steerer_passes < 1:
            raise ValueError("steerer_passes must be at least 1")


@dataclass(frozen=True)
class Cup5OptimizationResult:
    initial_state: MachineState

    esa_seed: ESASeedApplication
    initial_esa_scan: TransmissionScanResult

    steerer_scans: tuple[
        TransmissionScanResult,
        ...
    ]

    qpt_scan: QPT2DScanResult
    final_esa_scan: TransmissionScanResult

    reference_checks: tuple[
        SourceReferenceCheckResult,
        ...
    ]

    final_state: MachineState

    final_measurement: BeamMeasurement
    final_reference: SourceReference
    final_transmission: TransmissionResult


def _commands_equal(first: float, second: float) -> bool:
    return math.isclose(
        float(first),
        float(second),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _assert_same_commands(
    before: MachineState,
    after: MachineState,
    parameters: tuple[str, ...],
    *,
    message: str,
) -> None:
    for parameter_name in parameters:
        if not _commands_equal(
            before.parameters[parameter_name],
            after.parameters[parameter_name],
        ):
            raise Cup5OptimizationError(
                f"{message}: {parameter_name} changed"
            )


def _assert_upstream_frozen(
    state: MachineState,
    cup4_reference_state: MachineState,
) -> None:
    for parameter_name in CUP5_FROZEN_UPSTREAM_PARAMETERS:
        if not _commands_equal(
            state.parameters[parameter_name],
            cup4_reference_state.parameters[parameter_name],
        ):
            raise Cup5OptimizationError(
                f"{parameter_name} changed during Cup-5 optimization"
            )

    if state.rfq != cup4_reference_state.rfq:
        raise Cup5OptimizationError(
            "RFQ configuration changed during Cup-5 optimization"
        )


def _assert_qpt_common_frozen(
    state: MachineState,
    common_v: float,
) -> None:
    current = evaluate_qpt(
        state
    ).command_coordinates.common_v

    if not _commands_equal(current, common_v):
        raise Cup5OptimizationError(
            "QPT common mode changed during Cup-5 optimization"
        )


def _validate_inputs(
    current_state: MachineState,
    cup4_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[str, SettlingPolicy],
) -> None:
    current_state.validate()
    cup4_reference_state.validate()
    profile.validate()

    if current_state.cup != 5:
        raise ValueError(
            "Cup-5 optimization requires Cup 5"
        )

    if current_state.stage not in (None, 5):
        raise ValueError(
            "Cup-5 optimization requires stage 5 or no stage assignment"
        )

    if cup4_reference_state.cup != 4:
        raise ValueError(
            "Saved Cup-4 reference state must select Cup 4"
        )

    if not math.isclose(
        current_state.mass_u,
        cup4_reference_state.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Cup-4 and Cup-5 states use different ion masses"
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

    for parameter_name in CUP5_REQUIRED_PARAMETERS:
        if parameter_name not in current_state.parameters:
            raise ValueError(
                f"Cup-5 state is missing {parameter_name}"
            )

        if parameter_name not in settling_policies:
            raise KeyError(
                f"No settling policy configured for {parameter_name}"
            )

    # Cup 5 must START from the complete Cup-4 transport solution.
    for parameter_name in (
        *CUP5_FROZEN_UPSTREAM_PARAMETERS,
        *CUP5_LOCAL_RETUNE_PARAMETERS,
    ):
        if parameter_name not in cup4_reference_state.parameters:
            raise ValueError(
                f"Cup-4 reference is missing {parameter_name}"
            )

        if not _commands_equal(
            current_state.parameters[parameter_name],
            cup4_reference_state.parameters[parameter_name],
        ):
            raise ValueError(
                f"{parameter_name} must initially match the Cup-4 reference"
            )

    if current_state.rfq != cup4_reference_state.rfq:
        raise ValueError(
            "Cup-5 RFQ state must initially match Cup 4"
        )

    if tracker.latest is None:
        raise ValueError(
            "Cup-5 optimization requires an existing Cup-1 reference"
        )


def _local_profile(
    profile: MassProfile,
    parameter_name: str,
    center: float,
    half_width: float,
) -> MassProfile:
    local = deepcopy(profile)

    definition = PARAMETERS[
        parameter_name
    ]

    learned_minimum, learned_maximum = profile.effective_bounds(
        parameter_name
    )

    minimum = max(
        float(definition.minimum),
        float(learned_minimum),
        float(center) - float(half_width),
    )

    maximum = min(
        float(definition.maximum),
        float(learned_maximum),
        float(center) + float(half_width),
    )

    # Always retain the actual current command even if older learned
    # evidence is narrower.
    minimum = min(minimum, float(center))
    maximum = max(maximum, float(center))

    minimum = max(
        minimum,
        float(definition.minimum),
    )
    maximum = min(
        maximum,
        float(definition.maximum),
    )

    if maximum <= minimum:
        raise ValueError(
            f"Local scan window collapsed for {parameter_name}"
        )

    local.set_learned_range(
        parameter_name,
        minimum,
        maximum,
        source="cup5_local_window",
    )

    return local


def _apply_esa_seed(
    adapter,
    state: MachineState,
    settling_policies: Mapping[str, SettlingPolicy],
    *,
    energy_per_volt: float,
    logger=None,
) -> ESASeedApplication:
    prediction = predict_esa_voltage(
        state,
        energy_per_volt=energy_per_volt,
    )

    seed_voltage = require_valid_esa_prediction(
        prediction
    )

    parameters = dict(
        state.parameters
    )
    parameters[
        ESA_PARAMETER
    ] = seed_voltage

    readbacks = dict(
        state.readbacks
    )
    readbacks.pop(
        ESA_PARAMETER,
        None,
    )

    target = MachineState(
        mass_u=state.mass_u,
        parameters=parameters,
        readbacks=readbacks,
        cup=state.cup,
        stage=state.stage,
        role="physics_seed",
        rfq=deepcopy(state.rfq),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata={
            **deepcopy(state.metadata),
            "esa_seed_energy_per_volt": (
                prediction.energy_per_volt
            ),
            "esa_seed_beam_energy_ev": (
                prediction.beam_energy_best_available_ev
            ),
            "esa_seed_voltage_v": seed_voltage,
        },
    )

    target.validate()

    transition = apply_state(
        adapter,
        current=state,
        target=target,
        settling_policies=settling_policies,
        select_target_cup=False,
    )

    result = ESASeedApplication(
        prediction=prediction,
        requested_voltage_v=seed_voltage,
        state_before=state,
        transition=transition,
        state_after=transition.observed_state,
    )

    if logger is not None:
        logger.log_state_transition(
            transition
        )

        logger.log_event(
            "cup5_esa_seed_applied",
            {
                "beam_energy_ev": (
                    prediction.beam_energy_best_available_ev
                ),
                "energy_per_volt": (
                    prediction.energy_per_volt
                ),
                "esa_seed_voltage_v": (
                    seed_voltage
                ),
                "state_id": (
                    transition.observed_state.state_id
                ),
            },
        )

    return result


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
        state_id=result.reference.state_id,
        purpose="periodic_cup1_source_reference",
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
    settling_policies: Mapping[str, SettlingPolicy],
    measurement_policy: MeasurementPolicy,
    *,
    noise_floor_a: float | None,
    logger,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
    results: list[SourceReferenceCheckResult],
) -> MachineState:
    now = monotonic()

    if not tracker.is_due(now):
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

    results.append(check)

    _log_reference_check(
        logger,
        check,
    )

    return check.working_state_after


def _make_final_state(
    state: MachineState,
    cup4_reference_state: MachineState,
) -> MachineState:
    esa = evaluate_esa(state)
    qpt = evaluate_qpt(state)

    result = MachineState(
        mass_u=state.mass_u,
        parameters=dict(state.parameters),
        readbacks=dict(state.readbacks),
        cup=5,
        stage=5,
        role="stage_best",
        rfq=deepcopy(state.rfq),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata={
            **deepcopy(state.metadata),

            "optimized_stage": 5,
            "objective": (
                "cup1_normalized_transmission"
            ),

            "esa_command_v": (
                esa.esa.command_v
            ),
            "esa_observed_v": (
                esa.esa.value_v
            ),
            "esa_beam_energy_command_ev": (
                esa.beam_energy_command_ev
            ),
            "esa_beam_energy_observed_ev": (
                esa.beam_energy_best_available_ev
            ),
            "esa_energy_per_volt_command": (
                esa.energy_per_volt_command
            ),
            "esa_energy_per_volt_observed": (
                esa.energy_per_volt_best_available
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

            "cup5_x2_shift_v": (
                state.parameters["steerer_x2_v"]
                - cup4_reference_state.parameters["steerer_x2_v"]
            ),
            "cup5_y2_shift_v": (
                state.parameters["steerer_y2_v"]
                - cup4_reference_state.parameters["steerer_y2_v"]
            ),
        },
    )

    result.validate()

    return result


def _update_profile(
    profile: MassProfile,
    final_state: MachineState,
    cup4_reference_state: MachineState,
) -> None:
    # Do not overwrite the stored Cup-4 QPT/X2/Y2 solution. Those local
    # changes belong to the Cup-5 final state, not the stage-4 optimum.
    profile.set_best_command(
        ESA_PARAMETER,
        final_state.parameters[
            ESA_PARAMETER
        ],
    )

    profile.set_best_state(
        "cup5_best",
        final_state.state_id,
    )

    esa = evaluate_esa(
        final_state
    )
    qpt = evaluate_qpt(
        final_state
    )

    profile.metadata[
        "cup5_esa"
    ] = {
        "command_v": (
            esa.esa.command_v
        ),
        "observed_v": (
            esa.esa.value_v
        ),
        "beam_energy_command_ev": (
            esa.beam_energy_command_ev
        ),
        "beam_energy_observed_ev": (
            esa.beam_energy_best_available_ev
        ),
        "energy_per_volt_command": (
            esa.energy_per_volt_command
        ),
        "energy_per_volt_observed": (
            esa.energy_per_volt_best_available
        ),
    }

    profile.metadata[
        "cup5_local_transport"
    ] = {
        "qpt_common_v": (
            qpt.command_coordinates.common_v
        ),
        "qpt_focus_v": (
            qpt.command_coordinates.global_focus_v
        ),
        "qpt_asymmetry_v": (
            qpt.command_coordinates.asymmetry_v
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
        "steerer_x2_shift_from_cup4_v": (
            final_state.parameters["steerer_x2_v"]
            - cup4_reference_state.parameters["steerer_x2_v"]
        ),
        "steerer_y2_shift_from_cup4_v": (
            final_state.parameters["steerer_y2_v"]
            - cup4_reference_state.parameters["steerer_y2_v"]
        ),
    }


def optimize_cup5(
    adapter,
    current_state: MachineState,
    cup4_reference_state: MachineState,
    cup1_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[str, SettlingPolicy],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    *,
    optimization_policy: Cup5OptimizationPolicy | None = None,
    noise_floor_a: float | None = None,
    logger=None,
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], str] = utc_now_iso,
) -> Cup5OptimizationResult:
    policy = (
        optimization_policy
        if optimization_policy is not None
        else Cup5OptimizationPolicy()
    )

    _validate_inputs(
        current_state,
        cup4_reference_state,
        profile,
        tracker,
        settling_policies,
    )

    working_state = capture_readbacks(
        adapter,
        current_state,
    )

    initial_state = working_state

    _assert_upstream_frozen(
        working_state,
        cup4_reference_state,
    )

    frozen_qpt_common_v = (
        evaluate_qpt(
            working_state
        ).command_coordinates.common_v
    )

    reference_checks: list[
        SourceReferenceCheckResult
    ] = []

    def maintenance_hook(
        state: MachineState,
    ) -> MachineState:
        refreshed = _refresh_reference_if_due(
            adapter,
            state,
            cup1_reference_state,
            tracker,
            settling_policies,
            measurement_policy,
            noise_floor_a=noise_floor_a,
            logger=logger,
            monotonic=monotonic,
            utc_now=utc_now,
            results=reference_checks,
        )

        if refreshed.cup != 5:
            raise Cup5OptimizationError(
                "Source-reference maintenance did not restore Cup 5"
            )

        _assert_upstream_frozen(
            refreshed,
            cup4_reference_state,
        )

        _assert_qpt_common_frozen(
            refreshed,
            frozen_qpt_common_v,
        )

        return refreshed

    if logger is not None:
        logger.log_event(
            "cup5_optimization_started",
            {
                "state_id": (
                    initial_state.state_id
                ),
                "mass_u": (
                    initial_state.mass_u
                ),
                "qpt_common_v": (
                    frozen_qpt_common_v
                ),
            },
        )

    # --------------------------------------------------------------
    # Phase A: refresh source reference if needed, then apply physics
    # ESA seed.
    # --------------------------------------------------------------

    working_state = maintenance_hook(
        working_state
    )

    transport_before_seed = working_state

    esa_seed = _apply_esa_seed(
        adapter,
        working_state,
        settling_policies,
        energy_per_volt=(
            policy.esa_energy_per_volt
        ),
        logger=logger,
    )

    working_state = (
        esa_seed.state_after
    )

    _assert_upstream_frozen(
        working_state,
        cup4_reference_state,
    )

    _assert_same_commands(
        transport_before_seed,
        working_state,
        CUP5_LOCAL_RETUNE_PARAMETERS,
        message=(
            "ESA seed unexpectedly changed Cup-4 transport"
        ),
    )

    _assert_qpt_common_frozen(
        working_state,
        frozen_qpt_common_v,
    )

    # --------------------------------------------------------------
    # Phase B: local ESA coarse-to-fine scan around the physics seed.
    # The scanner baseline itself is the experimental test of the seed.
    # --------------------------------------------------------------

    esa_profile = _local_profile(
        profile,
        ESA_PARAMETER,
        working_state.parameters[
            ESA_PARAMETER
        ],
        policy.initial_esa_half_width_v,
    )

    initial_esa_scan = (
        scan_parameter_transmission_1d(
            adapter,
            working_state,
            esa_profile,
            tracker,
            ESA_PARAMETER,
            policy.initial_esa_scan,
            settling_policies,
            measurement_policy,
            comparison_policy,
            noise_floor_a=noise_floor_a,
            logger=logger,
            maintenance_hook=maintenance_hook,
        )
    )

    working_state = (
        initial_esa_scan.final_state
    )

    _assert_upstream_frozen(
        working_state,
        cup4_reference_state,
    )

    _assert_same_commands(
        transport_before_seed,
        working_state,
        CUP5_LOCAL_RETUNE_PARAMETERS,
        message=(
            "Initial ESA scan unexpectedly changed Cup-4 transport"
        ),
    )

    # --------------------------------------------------------------
    # Phase C: small X2/Y2 retunes.
    # --------------------------------------------------------------

    steerer_scans: list[
        TransmissionScanResult
    ] = []

    for _ in range(
        policy.steerer_passes
    ):
        for parameter_name in (
            "steerer_x2_v",
            "steerer_y2_v",
        ):
            working_state = maintenance_hook(
                working_state
            )

            qpt_before = working_state

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
                    noise_floor_a=noise_floor_a,
                    logger=logger,
                    maintenance_hook=maintenance_hook,
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
                cup4_reference_state,
            )

            _assert_same_commands(
                qpt_before,
                working_state,
                CUP5_QPT_PARAMETERS,
                message=(
                    f"{parameter_name} scan unexpectedly changed QPT"
                ),
            )

            _assert_qpt_common_frozen(
                working_state,
                frozen_qpt_common_v,
            )

    # --------------------------------------------------------------
    # Phase D: small local QPT F/A correction for transmission through
    # the ESA. C remains frozen.
    # --------------------------------------------------------------

    working_state = maintenance_hook(
        working_state
    )

    steering_before_qpt = working_state

    qpt_scan = scan_qpt_focus_asymmetry_2d(
        adapter,
        working_state,
        tracker,
        policy.local_qpt_scan,
        settling_policies,
        measurement_policy,
        comparison_policy,
        noise_floor_a=noise_floor_a,
        logger=logger,
        maintenance_hook=maintenance_hook,
    )

    working_state = (
        qpt_scan.final_state
    )

    _assert_upstream_frozen(
        working_state,
        cup4_reference_state,
    )

    _assert_same_commands(
        steering_before_qpt,
        working_state,
        (
            "steerer_x2_v",
            "steerer_y2_v",
            ESA_PARAMETER,
        ),
        message=(
            "QPT refinement unexpectedly changed steering or ESA"
        ),
    )

    _assert_qpt_common_frozen(
        working_state,
        frozen_qpt_common_v,
    )

    # --------------------------------------------------------------
    # Phase E: final fine ESA refinement with QPT and steering frozen.
    # --------------------------------------------------------------

    working_state = maintenance_hook(
        working_state
    )

    transport_before_final_esa = (
        working_state
    )

    final_esa_profile = _local_profile(
        profile,
        ESA_PARAMETER,
        working_state.parameters[
            ESA_PARAMETER
        ],
        policy.final_esa_half_width_v,
    )

    final_esa_scan = (
        scan_parameter_transmission_1d(
            adapter,
            working_state,
            final_esa_profile,
            tracker,
            ESA_PARAMETER,
            policy.final_esa_scan,
            settling_policies,
            measurement_policy,
            comparison_policy,
            noise_floor_a=noise_floor_a,
            logger=logger,
            maintenance_hook=maintenance_hook,
        )
    )

    working_state = (
        final_esa_scan.final_state
    )

    _assert_upstream_frozen(
        working_state,
        cup4_reference_state,
    )

    _assert_same_commands(
        transport_before_final_esa,
        working_state,
        CUP5_LOCAL_RETUNE_PARAMETERS,
        message=(
            "Final ESA scan unexpectedly changed Cup-4 transport"
        ),
    )

    _assert_qpt_common_frozen(
        working_state,
        frozen_qpt_common_v,
    )

    # --------------------------------------------------------------
    # Phase F: final source-normalized Cup-5 measurement.
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
        cup4_reference_state,
    )

    _assert_qpt_common_frozen(
        working_state,
        frozen_qpt_common_v,
    )

    final_state = _make_final_state(
        working_state,
        cup4_reference_state,
    )

    final_measurement = measure_beam_current(
        adapter,
        measurement_policy,
        noise_floor_a=noise_floor_a,
    )

    if (
        final_measurement.below_noise_floor
        or final_measurement.mean_a <= 0
    ):
        raise Cup5OptimizationNoBeamError(
            "Final Cup-5 current is not a valid transport signal"
        )

    final_reference = tracker.latest

    if final_reference is None:
        raise Cup5OptimizationError(
            "Cup-1 source reference disappeared during Cup-5 optimization"
        )

    final_transmission = transmission_from_reference(
        5,
        final_measurement,
        final_reference,
    )

    _update_profile(
        profile,
        final_state,
        cup4_reference_state,
    )

    if logger is not None:
        logger.save_state(
            final_state,
            "cup5_best",
        )

        logger.log_measurement(
            final_measurement,
            cup=5,
            state_id=final_state.state_id,
            purpose="cup5_final",
        )

        logger.log_transmission(
            final_transmission
        )

        esa = evaluate_esa(
            final_state
        )

        qpt = evaluate_qpt(
            final_state
        )

        logger.log_event(
            "cup5_optimization_completed",
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
                "esa_command_v": (
                    esa.esa.command_v
                ),
                "esa_observed_v": (
                    esa.esa.value_v
                ),
                "esa_effective_energy_per_volt": (
                    esa.energy_per_volt_best_available
                ),
                "qpt_focus_v": (
                    qpt.command_coordinates.global_focus_v
                ),
                "qpt_asymmetry_v": (
                    qpt.command_coordinates.asymmetry_v
                ),
                "steerer_x2_v": (
                    final_state.parameters["steerer_x2_v"]
                ),
                "steerer_y2_v": (
                    final_state.parameters["steerer_y2_v"]
                ),
                "reference_checks": (
                    len(reference_checks)
                ),
            },
        )

    return Cup5OptimizationResult(
        initial_state=initial_state,
        esa_seed=esa_seed,
        initial_esa_scan=initial_esa_scan,
        steerer_scans=tuple(
            steerer_scans
        ),
        qpt_scan=qpt_scan,
        final_esa_scan=final_esa_scan,
        reference_checks=tuple(
            reference_checks
        ),
        final_state=final_state,
        final_measurement=final_measurement,
        final_reference=final_reference,
        final_transmission=final_transmission,
    )