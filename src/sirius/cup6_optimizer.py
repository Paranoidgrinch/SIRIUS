from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

from sirius.comparison import ComparisonPolicy
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
from sirius.transition import capture_readbacks
from sirius.transmission_scan1d import (
    TransmissionScanResult,
    scan_parameter_transmission_1d,
)


LENS4_PARAMETER = "lens4_voltage_v"

CUP6_STEERER_PARAMETERS = (
    "steerer_x3_v",
    "steerer_y3_v",
)

CUP6_PRIMARY_PARAMETERS = (
    LENS4_PARAMETER,
    *CUP6_STEERER_PARAMETERS,
)

# Complete transport solution established before Cup 6.
CUP6_FROZEN_UPSTREAM_PARAMETERS = (
    # Cup 1
    "sputter_voltage_v",
    "extraction_voltage_v",
    "einzel_lens_voltage_v",
    "magnet_current_a",

    # Cup 2
    "lens2_voltage_v",
    "steerer_x1_v",
    "steerer_y1_v",

    # Cup 3
    "ion_cooler_voltage_v",
    "deceleration_voltage_v",
    "acceleration_voltage_v",
    "guidefield1_voltage_v",
    "guidefield2_voltage_v",

    # Cup 4
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
    "steerer_x2_v",
    "steerer_y2_v",

    # Cup 5
    "esa_voltage_v",
)

CUP6_REQUIRED_PARAMETERS = (
    *CUP6_FROZEN_UPSTREAM_PARAMETERS,
    *CUP6_PRIMARY_PARAMETERS,
)


class Cup6OptimizationError(RuntimeError):
    pass


class Cup6OptimizationNoBeamError(
    Cup6OptimizationError
):
    pass


@dataclass(frozen=True)
class Cup6OptimizationPolicy:
    """
    Final transport optimization before Cup 6.

    Lens4 is optimized first because it establishes the gross focusing
    condition at the final cup. X3/Y3 then center the beam, followed by
    a small final Lens4 refinement.
    """

    initial_lens4_half_width_v: float = 2500.0

    initial_lens4_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                500.0,
                100.0,
                25.0,
            )
        )
    )

    steerer_half_width_v: float = 100.0

    steerer_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                25.0,
                5.0,
                1.0,
            )
        )
    )

    steerer_passes: int = 2

    final_lens4_half_width_v: float = 300.0

    final_lens4_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                50.0,
                10.0,
                2.0,
            )
        )
    )

    def __post_init__(self) -> None:
        for name, value in (
            (
                "initial_lens4_half_width_v",
                self.initial_lens4_half_width_v,
            ),
            (
                "steerer_half_width_v",
                self.steerer_half_width_v,
            ),
            (
                "final_lens4_half_width_v",
                self.final_lens4_half_width_v,
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

        if self.steerer_passes < 1:
            raise ValueError(
                "steerer_passes must be at least 1"
            )


@dataclass(frozen=True)
class Cup6OptimizationResult:
    initial_state: MachineState

    initial_lens4_scan: TransmissionScanResult

    steerer_scans: tuple[
        TransmissionScanResult,
        ...
    ]

    final_lens4_scan: TransmissionScanResult

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


def _validate_inputs(
    current_state: MachineState,
    cup5_reference_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> None:
    current_state.validate()
    cup5_reference_state.validate()
    profile.validate()

    if current_state.cup != 6:
        raise ValueError(
            "Cup-6 optimization requires Cup 6"
        )

    if current_state.stage not in (
        None,
        6,
    ):
        raise ValueError(
            "Cup-6 optimization requires stage 6 or no stage assignment"
        )

    if cup5_reference_state.cup != 5:
        raise ValueError(
            "Saved Cup-5 reference state must select Cup 5"
        )

    if not math.isclose(
        current_state.mass_u,
        cup5_reference_state.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Cup-5 and Cup-6 states use different ion masses"
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
        CUP6_REQUIRED_PARAMETERS
    ):
        if (
            parameter_name
            not in current_state.parameters
        ):
            raise ValueError(
                f"Cup-6 state is missing {parameter_name}"
            )

        if (
            parameter_name
            not in settling_policies
        ):
            raise KeyError(
                f"No settling policy configured for {parameter_name}"
            )

    # Cup 6 must start from the complete Cup-5 transport solution.
    for parameter_name in (
        CUP6_FROZEN_UPSTREAM_PARAMETERS
    ):
        if (
            parameter_name
            not in cup5_reference_state.parameters
        ):
            raise ValueError(
                f"Cup-5 reference is missing {parameter_name}"
            )

        if not _commands_equal(
            current_state.parameters[
                parameter_name
            ],
            cup5_reference_state.parameters[
                parameter_name
            ],
        ):
            raise ValueError(
                f"{parameter_name} must initially match "
                "the Cup-5 reference"
            )

    if (
        current_state.rfq
        != cup5_reference_state.rfq
    ):
        raise ValueError(
            "Cup-6 RFQ state must initially match Cup 5"
        )

    if tracker.latest is None:
        raise ValueError(
            "Cup-6 optimization requires an existing Cup-1 reference"
        )


def _assert_upstream_frozen(
    state: MachineState,
    cup5_reference_state: MachineState,
) -> None:
    for parameter_name in (
        CUP6_FROZEN_UPSTREAM_PARAMETERS
    ):
        actual = state.parameters[
            parameter_name
        ]

        expected = (
            cup5_reference_state.parameters[
                parameter_name
            ]
        )

        if not _commands_equal(
            actual,
            expected,
        ):
            raise Cup6OptimizationError(
                f"{parameter_name} changed during Cup-6 optimization"
            )

    if state.rfq != cup5_reference_state.rfq:
        raise Cup6OptimizationError(
            "RFQ configuration changed during Cup-6 optimization"
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

    minimum = max(
        float(definition.minimum),
        float(learned_minimum),
        float(center)
        - float(half_width),
    )

    maximum = min(
        float(definition.maximum),
        float(learned_maximum),
        float(center)
        + float(half_width),
    )

    # An older learned range may be narrower than the actually loaded
    # state. Never exclude the current hardware command.
    minimum = min(
        minimum,
        float(center),
    )

    maximum = max(
        maximum,
        float(center),
    )

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
        source="cup6_local_window",
    )

    return local


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

    return (
        check.working_state_after
    )


def _make_final_state(
    state: MachineState,
    cup5_reference_state: MachineState,
) -> MachineState:
    result = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            state.parameters
        ),
        readbacks=dict(
            state.readbacks
        ),
        cup=6,
        stage=6,
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

            "optimized_stage": 6,
            "objective": (
                "cup1_normalized_transmission"
            ),

            "lens4_command_v": (
                state.parameters[
                    LENS4_PARAMETER
                ]
            ),

            "lens4_observed_v": (
                state.readbacks.get(
                    LENS4_PARAMETER
                )
            ),

            "steerer_x3_v": (
                state.parameters[
                    "steerer_x3_v"
                ]
            ),

            "steerer_y3_v": (
                state.parameters[
                    "steerer_y3_v"
                ]
            ),

            "cup6_lens4_shift_from_initial_v": (
                state.parameters[
                    LENS4_PARAMETER
                ]
                - cup5_reference_state.parameters.get(
                    LENS4_PARAMETER,
                    state.parameters[
                        LENS4_PARAMETER
                    ],
                )
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
        CUP6_PRIMARY_PARAMETERS
    ):
        profile.set_best_command(
            parameter_name,
            final_state.parameters[
                parameter_name
            ],
        )

    profile.set_best_state(
        "cup6_best",
        final_state.state_id,
    )

    profile.metadata[
        "cup6_transport"
    ] = {
        "lens4_command_v": (
            final_state.parameters[
                LENS4_PARAMETER
            ]
        ),
        "lens4_observed_v": (
            final_state.readbacks.get(
                LENS4_PARAMETER
            )
        ),
        "steerer_x3_v": (
            final_state.parameters[
                "steerer_x3_v"
            ]
        ),
        "steerer_y3_v": (
            final_state.parameters[
                "steerer_y3_v"
            ]
        ),
    }


def optimize_cup6(
    adapter,
    current_state: MachineState,
    cup5_reference_state: MachineState,
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
        Cup6OptimizationPolicy | None
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
) -> Cup6OptimizationResult:
    """
    Optimize the final transport section from the frozen Cup-5 solution
    to Cup 6.

    Only Lens4 and X3/Y3 may change.
    """

    policy = (
        optimization_policy
        if optimization_policy is not None
        else Cup6OptimizationPolicy()
    )

    _validate_inputs(
        current_state,
        cup5_reference_state,
        profile,
        tracker,
        settling_policies,
    )

    working_state = capture_readbacks(
        adapter,
        current_state,
    )

    initial_state = (
        working_state
    )

    _assert_upstream_frozen(
        working_state,
        cup5_reference_state,
    )

    reference_checks: list[
        SourceReferenceCheckResult
    ] = []

    def maintenance_hook(
        state: MachineState,
    ) -> MachineState:
        refreshed = (
            _refresh_reference_if_due(
                adapter,
                state,
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

        if refreshed.cup != 6:
            raise Cup6OptimizationError(
                "Source-reference maintenance did not restore Cup 6"
            )

        _assert_upstream_frozen(
            refreshed,
            cup5_reference_state,
        )

        return refreshed

    if logger is not None:
        logger.log_event(
            "cup6_optimization_started",
            {
                "state_id": (
                    initial_state.state_id
                ),
                "mass_u": (
                    initial_state.mass_u
                ),
                "lens4_voltage_v": (
                    initial_state.parameters[
                        LENS4_PARAMETER
                    ]
                ),
                "steerer_x3_v": (
                    initial_state.parameters[
                        "steerer_x3_v"
                    ]
                ),
                "steerer_y3_v": (
                    initial_state.parameters[
                        "steerer_y3_v"
                    ]
                ),
            },
        )

    # --------------------------------------------------------------
    # Phase A: gross/fine Lens4 focusing.
    # --------------------------------------------------------------

    working_state = maintenance_hook(
        working_state
    )

    lens4_profile = _local_profile(
        profile,
        LENS4_PARAMETER,
        working_state.parameters[
            LENS4_PARAMETER
        ],
        policy.initial_lens4_half_width_v,
    )

    initial_lens4_scan = (
        scan_parameter_transmission_1d(
            adapter,
            working_state,
            lens4_profile,
            tracker,
            LENS4_PARAMETER,
            policy.initial_lens4_scan,
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
        initial_lens4_scan.final_state
    )

    _assert_upstream_frozen(
        working_state,
        cup5_reference_state,
    )

    # --------------------------------------------------------------
    # Phase B: X3/Y3 local steering.
    # --------------------------------------------------------------

    steerer_scans: list[
        TransmissionScanResult
    ] = []

    for _ in range(
        policy.steerer_passes
    ):
        for parameter_name in (
            CUP6_STEERER_PARAMETERS
        ):
            working_state = (
                maintenance_hook(
                    working_state
                )
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
                cup5_reference_state,
            )

    # --------------------------------------------------------------
    # Phase C: final Lens4 refinement after beam centering.
    # --------------------------------------------------------------

    working_state = maintenance_hook(
        working_state
    )

    final_lens4_profile = (
        _local_profile(
            profile,
            LENS4_PARAMETER,
            working_state.parameters[
                LENS4_PARAMETER
            ],
            policy.final_lens4_half_width_v,
        )
    )

    final_lens4_scan = (
        scan_parameter_transmission_1d(
            adapter,
            working_state,
            final_lens4_profile,
            tracker,
            LENS4_PARAMETER,
            policy.final_lens4_scan,
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
        final_lens4_scan.final_state
    )

    _assert_upstream_frozen(
        working_state,
        cup5_reference_state,
    )

    # --------------------------------------------------------------
    # Phase D: final source-normalized Cup-6 measurement.
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
        cup5_reference_state,
    )

    final_state = _make_final_state(
        working_state,
        cup5_reference_state,
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
        raise Cup6OptimizationNoBeamError(
            "Final Cup-6 current is not a valid transport signal"
        )

    final_reference = (
        tracker.latest
    )

    if final_reference is None:
        raise Cup6OptimizationError(
            "Cup-1 source reference disappeared during Cup-6 optimization"
        )

    final_transmission = (
        transmission_from_reference(
            6,
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
            "cup6_best",
        )

        logger.log_measurement(
            final_measurement,
            cup=6,
            state_id=(
                final_state.state_id
            ),
            purpose="cup6_final",
        )

        logger.log_transmission(
            final_transmission
        )

        logger.log_event(
            "cup6_optimization_completed",
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
                "lens4_voltage_v": (
                    final_state.parameters[
                        LENS4_PARAMETER
                    ]
                ),
                "lens4_readback_v": (
                    final_state.readbacks.get(
                        LENS4_PARAMETER
                    )
                ),
                "steerer_x3_v": (
                    final_state.parameters[
                        "steerer_x3_v"
                    ]
                ),
                "steerer_y3_v": (
                    final_state.parameters[
                        "steerer_y3_v"
                    ]
                ),
                "reference_checks": (
                    len(
                        reference_checks
                    )
                ),
            },
        )

    return Cup6OptimizationResult(
        initial_state=(
            initial_state
        ),
        initial_lens4_scan=(
            initial_lens4_scan
        ),
        steerer_scans=tuple(
            steerer_scans
        ),
        final_lens4_scan=(
            final_lens4_scan
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