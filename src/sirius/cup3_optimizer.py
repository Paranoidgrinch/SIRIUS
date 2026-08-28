from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping

from sirius.comparison import ComparisonPolicy
from sirius.cup3_coordinates import (
    EndElectrodeCoordinatePolicy,
    GuidefieldCoordinatePolicy,
    optimize_end_electrode_coordinates,
    optimize_guidefield_coordinates,
)
from sirius.derived_scan1d import DerivedScanResult
from sirius.cooler_electrodes import (
    evaluate_cooler_end_electrodes,
)
from sirius.cooler_model import (
    ion_cooler_energy_state,
)
from sirius.guidefield_model import (
    evaluate_guidefield,
)
from sirius.mass_profile import MassProfile
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
from sirius.residual_energy_scan import (
    ResidualEnergyScanPolicy,
    ResidualEnergyScanResult,
    scan_residual_energy,
)
from sirius.rfq_matching import (
    LCSetting,
    RFQMatchingPolicy,
    RFQMatchingResult,
    RFQTargetQPolicy,
    RFQTargetQResult,
    search_rfq_resonance,
    set_target_q,
)
from sirius.scan1d import ScanPolicy
from sirius.settling import SettlingPolicy
from sirius.state import (
    MachineState,
    RFQState,
    utc_now_iso,
)
from sirius.transition import (
    capture_readbacks,
)
from sirius.transmission_scan1d import (
    TransmissionScanResult,
    scan_parameter_transmission_1d,
)


CUP3_FROZEN_SOURCE_PARAMETERS = (
    "sputter_voltage_v",
    "extraction_voltage_v",
    "magnet_current_a",
)

CUP3_PRIMARY_PARAMETERS = (
    "ion_cooler_voltage_v",
    "deceleration_voltage_v",
    "acceleration_voltage_v",
    "guidefield1_voltage_v",
    "guidefield2_voltage_v",
)

CUP3_UPSTREAM_RETUNE_PARAMETERS = (
    "einzel_lens_voltage_v",
    "lens2_voltage_v",
    "steerer_x1_v",
    "steerer_y1_v",
)

CUP3_REQUIRED_PARAMETERS = (
    *CUP3_FROZEN_SOURCE_PARAMETERS,
    *CUP3_PRIMARY_PARAMETERS,
    *CUP3_UPSTREAM_RETUNE_PARAMETERS,
)


class Cup3OptimizationError(RuntimeError):
    pass


class Cup3OptimizationNoBeamError(
    Cup3OptimizationError
):
    pass


@dataclass(frozen=True)
class Cup3OptimizationPolicy:
    """
    Conservative Cup-3 optimization policy.

    HV1/HV4 and GF1/GF2 are optimized in derived differential/common
    coordinates rather than as independent supplies.
    """

    residual_scan: ResidualEnergyScanPolicy = field(
        default_factory=lambda: ResidualEnergyScanPolicy(
            minimum_ev=10.0,
            maximum_ev=120.0,
            steps_ev=(
                10.0,
                1.0,
            ),
        )
    )

    final_residual_half_width_ev: float = 5.0

    final_residual_steps_ev: tuple[
        float,
        ...
    ] = (
        1.0,
        0.25,
    )

    end_electrode_policy: (
        EndElectrodeCoordinatePolicy
    ) = field(
        default_factory=(
            EndElectrodeCoordinatePolicy
        )
    )

    guidefield_policy: (
        GuidefieldCoordinatePolicy
    ) = field(
        default_factory=(
            GuidefieldCoordinatePolicy
        )
    )

    einzel_half_width_v: float = 500.0

    einzel_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                100.0,
                25.0,
            )
        )
    )

    lens2_half_width_v: float = 500.0

    lens2_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                100.0,
                25.0,
            )
        )
    )

    steerer_half_width_v: float = 30.0

    steerer_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                10.0,
                2.0,
            )
        )
    )

    upstream_passes: int = 1

    # Compatibility overrides for existing callers/tests.
    # None means: use the pass count stored in the nested policy.
    electrode_passes: int | None = None
    guidefield_passes: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            (
                "final_residual_half_width_ev",
                self.final_residual_half_width_ev,
            ),
            (
                "einzel_half_width_v",
                self.einzel_half_width_v,
            ),
            (
                "lens2_half_width_v",
                self.lens2_half_width_v,
            ),
            (
                "steerer_half_width_v",
                self.steerer_half_width_v,
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

        if not self.final_residual_steps_ev:
            raise ValueError(
                "final_residual_steps_ev must not be empty"
            )

        for step in self.final_residual_steps_ev:
            if not math.isfinite(
                float(step)
            ) or step <= 0:
                raise ValueError(
                    "Final residual-energy steps must be positive and finite"
                )

        for previous, current in zip(
            self.final_residual_steps_ev,
            self.final_residual_steps_ev[1:],
        ):
            if current >= previous:
                raise ValueError(
                    "Final residual-energy steps must decrease"
                )

        if self.upstream_passes < 1:
            raise ValueError(
                "upstream_passes must be at least 1"
            )

        for name, value in (
            (
                "electrode_passes",
                self.electrode_passes,
            ),
            (
                "guidefield_passes",
                self.guidefield_passes,
            ),
        ):
            if (
                value is not None
                and value < 1
            ):
                raise ValueError(
                    f"{name} must be at least 1 when supplied"
                )


@dataclass(frozen=True)
class Cup3OptimizationResult:
    initial_state: MachineState

    rfq_matching: RFQMatchingResult
    initial_q_result: RFQTargetQResult

    residual_scan: ResidualEnergyScanResult

    electrode_scans: tuple[
        DerivedScanResult,
        ...
    ]

    guidefield_scans: tuple[
        DerivedScanResult,
        ...
    ]

    upstream_scans: tuple[
        TransmissionScanResult,
        ...
    ]

    final_residual_scan: (
        ResidualEnergyScanResult
    )

    final_q_result: RFQTargetQResult

    reference_checks: tuple[
        SourceReferenceCheckResult,
        ...
    ]

    final_state: MachineState

    final_measurement: BeamMeasurement
    final_reference: SourceReference
    final_transmission: TransmissionResult


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
        float(definition.minimum),
        min(
            float(center),
            float(learned_minimum),
        ),
    )

    allowed_maximum = min(
        float(definition.maximum),
        max(
            float(center),
            float(learned_maximum),
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

    local.set_learned_range(
        parameter_name,
        minimum,
        maximum,
        source="cup3_local_window",
    )

    return local


def _validate_inputs(
    current_state: MachineState,
    cup1_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> None:
    current_state.validate()
    cup1_reference_state.validate()
    profile.validate()

    if current_state.cup != 3:
        raise ValueError(
            "Cup-3 optimization requires cup 3"
        )

    if current_state.stage not in (
        None,
        3,
    ):
        raise ValueError(
            "Cup-3 optimization requires stage 3 or no stage assignment"
        )

    if cup1_reference_state.cup != 1:
        raise ValueError(
            "Saved Cup-1 reference state must select cup 1"
        )

    if not math.isclose(
        current_state.mass_u,
        cup1_reference_state.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Cup-1 and Cup-3 states use different ion masses"
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
        CUP3_REQUIRED_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"Cup-3 state is missing {parameter_name}"
            )

    for parameter_name in (
        CUP3_FROZEN_SOURCE_PARAMETERS
    ):
        if (
            parameter_name
            not in cup1_reference_state.parameters
        ):
            raise ValueError(
                f"Cup-1 reference is missing {parameter_name}"
            )

        if not _commands_equal(
            current_state.parameters[
                parameter_name
            ],
            cup1_reference_state.parameters[
                parameter_name
            ],
        ):
            raise ValueError(
                f"{parameter_name} must match the Cup-1 reference"
            )

    required_settling = (
        *CUP3_PRIMARY_PARAMETERS,
        *CUP3_UPSTREAM_RETUNE_PARAMETERS,
    )

    for parameter_name in (
        required_settling
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
            "Cup-3 optimization requires an existing Cup-1 reference"
        )


def _assert_source_frozen(
    state: MachineState,
    cup1_reference_state: MachineState,
) -> None:
    for parameter_name in (
        CUP3_FROZEN_SOURCE_PARAMETERS
    ):
        expected = (
            cup1_reference_state.parameters[
                parameter_name
            ]
        )

        actual = state.parameters[
            parameter_name
        ]

        if not _commands_equal(
            actual,
            expected,
        ):
            raise Cup3OptimizationError(
                f"{parameter_name} changed during Cup-3 optimization"
            )


def _with_rfq_result(
    state: MachineState,
    matching: RFQMatchingResult,
    q_result: RFQTargetQResult,
) -> MachineState:
    result = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            state.parameters
        ),
        readbacks=dict(
            state.readbacks
        ),
        cup=state.cup,
        stage=state.stage,
        role=state.role,
        rfq=RFQState(
            frequency_hz=(
                matching.best_frequency_hz
            ),
            generator_amplitude_vpp=(
                q_result.generator_amplitude_vpp
            ),
            inductance_uh=(
                matching.best_setting.inductance_uh
            ),
            capacitance_pf=(
                matching.best_setting.capacitance_pf
            ),
            rfq_vpp_measured=(
                q_result.measured_rfq_vpp
            ),
            q_target=(
                q_result.target_q
            ),
            q_nominal=None,
            q_measured=(
                q_result.measured_q
            ),
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata=deepcopy(
            state.metadata
        ),
    )

    result.validate()

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
    final_state: MachineState,
    q_result: RFQTargetQResult,
) -> None:
    for parameter_name in (
        CUP3_PRIMARY_PARAMETERS
    ):
        profile.set_best_command(
            parameter_name,
            final_state.parameters[
                parameter_name
            ],
        )

    profile.set_best_state(
        "cup3_best",
        final_state.state_id,
    )

    profile.metadata[
        "cup3_rfq"
    ] = {
        "frequency_hz": (
            q_result.frequency_hz
        ),
        "inductance_uh": (
            q_result.setting.inductance_uh
        ),
        "capacitance_pf": (
            q_result.setting.capacitance_pf
        ),
        "generator_amplitude_vpp": (
            q_result.generator_amplitude_vpp
        ),
        "rfq_vpp_measured": (
            q_result.measured_rfq_vpp
        ),
        "q_target": (
            q_result.target_q
        ),
        "q_measured": (
            q_result.measured_q
        ),
    }


def _final_state(
    state: MachineState,
    profile: MassProfile,
    matching: RFQMatchingResult,
    q_result: RFQTargetQResult,
) -> MachineState:
    cooler = ion_cooler_energy_state(
        state
    )

    guidefield = evaluate_guidefield(
        state,
        profile=profile,
    )

    electrodes = (
        evaluate_cooler_end_electrodes(
            state
        )
    )

    result = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            state.parameters
        ),
        readbacks=dict(
            state.readbacks
        ),
        cup=3,
        stage=3,
        role="stage_best",
        rfq=RFQState(
            frequency_hz=(
                matching.best_frequency_hz
            ),
            generator_amplitude_vpp=(
                q_result.generator_amplitude_vpp
            ),
            inductance_uh=(
                matching.best_setting.inductance_uh
            ),
            capacitance_pf=(
                matching.best_setting.capacitance_pf
            ),
            rfq_vpp_measured=(
                q_result.measured_rfq_vpp
            ),
            q_target=(
                q_result.target_q
            ),
            q_nominal=None,
            q_measured=(
                q_result.measured_q
            ),
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata={
            **deepcopy(state.metadata),
            "optimized_stage": 3,
            "objective": (
                "cup1_normalized_transmission"
            ),
            "residual_energy_command_ev": (
                cooler.residual_energy_command_ev
            ),
            "residual_energy_observed_ev": (
                cooler.residual_energy_best_available_ev
            ),
            "guidefield_command_difference_v": (
                guidefield.command_difference_v
            ),
            "guidefield_observed_difference_v": (
                guidefield.best_available_difference_v
            ),
            "guidefield_command_common_v": (
                (
                    guidefield.gf1.command_v
                    + guidefield.gf2.command_v
                )
                / 2.0
            ),
            "end_electrode_command_difference_v": (
                electrodes.command_end_difference_v
            ),
            "end_electrode_command_common_v": (
                electrodes.command_common_bias_v
            ),
        },
    )

    result.validate()

    return result


def optimize_cup3(
    adapter,
    rfq_hardware,
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
    lc_candidates: Iterable[
        LCSetting
    ],
    rfq_matching_policy: RFQMatchingPolicy,
    rfq_q_policy: RFQTargetQPolicy,
    target_q: float,
    optimization_policy: (
        Cup3OptimizationPolicy | None
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
) -> Cup3OptimizationResult:
    """
    Complete first-generation Cup-3 optimizer.

    Source energy and analyzing magnet remain frozen to the Cup-1
    reference commands.

    RFQ matching is performed first and then frozen. Beam transport is
    optimized against source-normalized T_1->3.
    """

    policy = (
        optimization_policy
        if optimization_policy is not None
        else Cup3OptimizationPolicy()
    )

    _validate_inputs(
        current_state,
        cup1_reference_state,
        profile,
        tracker,
        settling_policies,
    )

    working_state = capture_readbacks(
        adapter,
        current_state,
    )

    initial_state = working_state

    _assert_source_frozen(
        working_state,
        cup1_reference_state,
    )

    reference_checks: list[
        SourceReferenceCheckResult
    ] = []

    if logger is not None:
        logger.log_event(
            "cup3_optimization_started",
            {
                "state_id": (
                    initial_state.state_id
                ),
                "mass_u": (
                    initial_state.mass_u
                ),
                "target_q": (
                    target_q
                ),
                "commands": (
                    initial_state.parameters
                ),
                "readbacks": (
                    initial_state.readbacks
                ),
            },
        )

    # --------------------------------------------------------------
    # Phase A: RFQ resonance.
    # --------------------------------------------------------------

    matching = search_rfq_resonance(
        rfq_hardware,
        mass_u=(
            working_state.mass_u
        ),
        lc_candidates=(
            lc_candidates
        ),
        policy=(
            rfq_matching_policy
        ),
        logger=logger,
    )

    # --------------------------------------------------------------
    # Phase B: establish measured q.
    # --------------------------------------------------------------

    initial_q_result = set_target_q(
        rfq_hardware,
        matching,
        target_q,
        rfq_q_policy,
        logger=logger,
    )

    working_state = _with_rfq_result(
        working_state,
        matching,
        initial_q_result,
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

        _assert_source_frozen(
            refreshed,
            cup1_reference_state,
        )

        return refreshed

    try:
        # Refresh source reference after RF-only work if necessary.
        working_state = (
            maintenance_hook(
                working_state
            )
        )

        # ----------------------------------------------------------
        # Phase C: residual energy.
        # ----------------------------------------------------------

        residual_scan = (
            scan_residual_energy(
                adapter,
                working_state,
                profile,
                tracker,
                policy.residual_scan,
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
            residual_scan.final_state
        )

        _assert_source_frozen(
            working_state,
            cup1_reference_state,
        )

        # ----------------------------------------------------------
        # Phase D: entrance / exit electrodes.
        #
        # Optimize differential and common coordinates instead of
        # independently scanning HV1 and HV4.
        # ----------------------------------------------------------

        end_policy = (
            policy.end_electrode_policy
        )

        if (
            policy.electrode_passes
            is not None
        ):
            end_policy = replace(
                end_policy,
                passes=(
                    policy.electrode_passes
                ),
            )

        electrode_phase = (
            optimize_end_electrode_coordinates(
                adapter,
                working_state,
                tracker,
                settling_policies,
                measurement_policy,
                comparison_policy,
                policy=end_policy,
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
            electrode_phase.final_state
        )

        electrode_scans = list(
            electrode_phase.scans
        )

        _assert_source_frozen(
            working_state,
            cup1_reference_state,
        )

        # ----------------------------------------------------------
        # Phase E: guidefields.
        #
        # Optimize GF1-GF2 first, then common mode. If direction is
        # unknown, both difference signs are probed where hardware
        # limits allow and the MassProfile may learn forward_sign.
        # ----------------------------------------------------------

        guide_policy = (
            policy.guidefield_policy
        )

        if (
            policy.guidefield_passes
            is not None
        ):
            guide_policy = replace(
                guide_policy,
                passes=(
                    policy.guidefield_passes
                ),
            )

        guidefield_phase = (
            optimize_guidefield_coordinates(
                adapter,
                working_state,
                profile,
                tracker,
                settling_policies,
                measurement_policy,
                comparison_policy,
                policy=guide_policy,
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
            guidefield_phase.final_state
        )

        guidefield_scans = list(
            guidefield_phase.scans
        )

        _assert_source_frozen(
            working_state,
            cup1_reference_state,
        )

        # ----------------------------------------------------------
        # Phase F: small upstream retunes.
        # ----------------------------------------------------------

        upstream_scans: list[
            TransmissionScanResult
        ] = []

        upstream_definitions = (
            (
                "einzel_lens_voltage_v",
                policy.einzel_half_width_v,
                policy.einzel_scan,
            ),
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
        )

        for _ in range(
            policy.upstream_passes
        ):
            for (
                parameter_name,
                half_width,
                scan_policy,
            ) in upstream_definitions:
                working_state = (
                    maintenance_hook(
                        working_state
                    )
                )

                local_profile = (
                    _local_profile(
                        profile,
                        parameter_name,
                        working_state.parameters[
                            parameter_name
                        ],
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

                upstream_scans.append(
                    scan
                )

                _assert_source_frozen(
                    working_state,
                    cup1_reference_state,
                )

        # ----------------------------------------------------------
        # Phase G1: final residual-energy fine scan.
        # ----------------------------------------------------------

        best_target = (
            residual_scan.best_target_residual_energy_ev
        )

        final_minimum = max(
            policy.residual_scan.minimum_ev,
            best_target
            - policy.final_residual_half_width_ev,
        )

        final_maximum = min(
            policy.residual_scan.maximum_ev,
            best_target
            + policy.final_residual_half_width_ev,
        )

        if final_maximum <= final_minimum:
            raise Cup3OptimizationError(
                "Invalid final residual-energy refinement window"
            )

        final_residual_policy = (
            ResidualEnergyScanPolicy(
                minimum_ev=(
                    final_minimum
                ),
                maximum_ev=(
                    final_maximum
                ),
                steps_ev=(
                    policy.final_residual_steps_ev
                ),
            )
        )

        final_residual_scan = (
            scan_residual_energy(
                adapter,
                working_state,
                profile,
                tracker,
                final_residual_policy,
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
            final_residual_scan.final_state
        )

        _assert_source_frozen(
            working_state,
            cup1_reference_state,
        )

        # ----------------------------------------------------------
        # Phase G2: re-measure / re-establish q at the frozen resonance.
        # ----------------------------------------------------------

        final_q_policy = replace(
            rfq_q_policy,
            initial_generator_vpp=(
                initial_q_result.generator_amplitude_vpp
            ),
        )

        final_q_result = set_target_q(
            rfq_hardware,
            matching,
            target_q,
            final_q_policy,
            logger=logger,
        )

        working_state = _with_rfq_result(
            working_state,
            matching,
            final_q_result,
        )

        # Final source normalization must be current.
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

        _assert_source_frozen(
            working_state,
            cup1_reference_state,
        )

        final_state = _final_state(
            working_state,
            profile,
            matching,
            final_q_result,
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
            raise Cup3OptimizationNoBeamError(
                "Final Cup-3 current is not a valid transport signal"
            )

        final_reference = (
            tracker.latest
        )

        if final_reference is None:
            raise Cup3OptimizationError(
                "Cup-1 source reference disappeared during Cup-3 optimization"
            )

        final_transmission = (
            transmission_from_reference(
                3,
                final_measurement,
                final_reference,
            )
        )

        _update_profile(
            profile,
            final_state,
            final_q_result,
        )

        if logger is not None:
            logger.save_state(
                final_state,
                "cup3_best",
            )

            logger.log_measurement(
                final_measurement,
                cup=3,
                state_id=(
                    final_state.state_id
                ),
                purpose=(
                    "cup3_final"
                ),
            )

            logger.log_transmission(
                final_transmission
            )

            logger.log_event(
                "cup3_optimization_completed",
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
                    "q_measured": (
                        final_q_result.measured_q
                    ),
                    "residual_energy": (
                        ion_cooler_energy_state(
                            final_state
                        )
                    ),
                    "guidefield": (
                        evaluate_guidefield(
                            final_state,
                            profile=profile,
                        )
                    ),
                    "end_electrodes": (
                        evaluate_cooler_end_electrodes(
                            final_state
                        )
                    ),
                    "reference_checks": (
                        len(reference_checks)
                    ),
                },
            )

        return Cup3OptimizationResult(
            initial_state=(
                initial_state
            ),
            rfq_matching=(
                matching
            ),
            initial_q_result=(
                initial_q_result
            ),
            residual_scan=(
                residual_scan
            ),
            electrode_scans=tuple(
                electrode_scans
            ),
            guidefield_scans=tuple(
                guidefield_scans
            ),
            upstream_scans=tuple(
                upstream_scans
            ),
            final_residual_scan=(
                final_residual_scan
            ),
            final_q_result=(
                final_q_result
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
        )

    except Exception:
        # If downstream optimization fails after RF has been established,
        # leave the RFQ in the safer RF-off condition.
        rfq_hardware.set_generator_amplitude_vpp(
            0.0
        )

        raise