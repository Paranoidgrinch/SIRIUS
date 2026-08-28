from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
)
from sirius.coupled_coordinates import (
    end_electrode_common_bounds,
    end_electrode_common_builder,
    end_electrode_common_command,
    end_electrode_difference_bounds,
    end_electrode_difference_builder,
    end_electrode_difference_command,
    guidefield_common_bounds,
    guidefield_common_builder,
    guidefield_common_command,
    guidefield_difference_bounds,
    guidefield_difference_builder,
    guidefield_difference_command,
)
from sirius.derived_scan1d import (
    DerivedScanResult,
    scan_derived_coordinate_transmission_1d,
)
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.objective import (
    ScalarComparison,
    ScalarEstimate,
    compare_estimates,
)
from sirius.reference import (
    SourceReferenceTracker,
    TransmissionResult,
)
from sirius.scan1d import ScanPolicy
from sirius.settling import SettlingPolicy
from sirius.state import MachineState


@dataclass(frozen=True)
class EndElectrodeCoordinatePolicy:
    """
    HV1/HV4 optimization in physical derived coordinates.

    difference:
        deceleration - acceleration

    common:
        (deceleration + acceleration) / 2
    """

    difference_half_width_v: float = 2000.0

    difference_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                500.0,
                100.0,
            )
        )
    )

    common_half_width_v: float = 1000.0

    common_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                250.0,
                50.0,
            )
        )
    )

    passes: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            (
                "difference_half_width_v",
                self.difference_half_width_v,
            ),
            (
                "common_half_width_v",
                self.common_half_width_v,
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

        if self.passes < 1:
            raise ValueError(
                "passes must be at least 1"
            )


@dataclass(frozen=True)
class GuidefieldCoordinatePolicy:
    """
    Guidefield optimization in differential/common coordinates.

    During initial direction learning, the differential scan tries to
    include both signs whenever the physical supply limits permit it.
    """

    difference_half_width_v: float = 16.0

    difference_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                4.0,
                1.0,
            )
        )
    )

    common_half_width_v: float = 8.0

    common_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                2.0,
                0.5,
            )
        )
    )

    passes: int = 1

    learn_forward_sign: bool = True

    probe_both_signs_when_unknown: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            (
                "difference_half_width_v",
                self.difference_half_width_v,
            ),
            (
                "common_half_width_v",
                self.common_half_width_v,
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

        if self.passes < 1:
            raise ValueError(
                "passes must be at least 1"
            )


@dataclass(frozen=True)
class GuidefieldDirectionEvidence:
    previous_sign: int | None

    proposed_sign: int | None

    positive_coordinate_v: float | None
    positive_transmission: float | None

    negative_coordinate_v: float | None
    negative_transmission: float | None

    comparison: ScalarComparison | None

    profile_updated: bool

    reason: str


@dataclass(frozen=True)
class EndElectrodeCoordinateResult:
    initial_state: MachineState

    scans: tuple[
        DerivedScanResult,
        ...
    ]

    final_state: MachineState


@dataclass(frozen=True)
class GuidefieldCoordinateResult:
    initial_state: MachineState

    scans: tuple[
        DerivedScanResult,
        ...
    ]

    direction_evidence: tuple[
        GuidefieldDirectionEvidence,
        ...
    ]

    final_state: MachineState


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


def _local_window(
    *,
    current: float,
    feasible_minimum: float,
    feasible_maximum: float,
    half_width: float,
    include_both_signs: bool = False,
) -> tuple[
    float,
    float,
]:
    """
    Build a local scan window while always retaining the current state.

    For unknown guidefield direction, include both positive and negative
    command-space differences whenever the hardware-feasible interval
    crosses zero.
    """

    current = _finite(
        "current coordinate",
        current,
    )

    feasible_minimum = _finite(
        "feasible minimum",
        feasible_minimum,
    )

    feasible_maximum = _finite(
        "feasible maximum",
        feasible_maximum,
    )

    half_width = _finite(
        "half width",
        half_width,
    )

    if half_width <= 0:
        raise ValueError(
            "half_width must be greater than zero"
        )

    if not (
        feasible_minimum
        <= current
        <= feasible_maximum
    ):
        raise ValueError(
            "Current coordinate lies outside feasible hardware interval"
        )

    minimum = max(
        feasible_minimum,
        current
        - half_width,
    )

    maximum = min(
        feasible_maximum,
        current
        + half_width,
    )

    if (
        include_both_signs
        and feasible_minimum < 0
        and feasible_maximum > 0
    ):
        minimum = max(
            feasible_minimum,
            min(
                minimum,
                -half_width,
            ),
        )

        maximum = min(
            feasible_maximum,
            max(
                maximum,
                half_width,
            ),
        )

    # Always retain the actual current state.
    minimum = min(
        minimum,
        current,
    )

    maximum = max(
        maximum,
        current,
    )

    if maximum <= minimum:
        raise ValueError(
            "Derived-coordinate scan window collapsed"
        )

    return (
        float(minimum),
        float(maximum),
    )


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


def _best_side_observation(
    scan: DerivedScanResult,
    *,
    positive: bool,
):
    observations = []

    initial_coordinate = float(
        scan.initial_coordinate
    )

    if (
        positive
        and initial_coordinate > 1e-12
    ) or (
        not positive
        and initial_coordinate < -1e-12
    ):
        observations.append(
            (
                initial_coordinate,
                scan.baseline_transmission,
                scan.baseline_measurement,
            )
        )

    for point in scan.points:
        coordinate = float(
            point.coordinate_value
        )

        if (
            positive
            and coordinate > 1e-12
        ) or (
            not positive
            and coordinate < -1e-12
        ):
            observations.append(
                (
                    coordinate,
                    point.transmission,
                    point.measurement,
                )
            )

    if not observations:
        return None

    return max(
        observations,
        key=lambda item: (
            item[1].transmission
        ),
    )


def learn_guidefield_direction_from_scan(
    profile: MassProfile,
    scan: DerivedScanResult,
    comparison_policy: ComparisonPolicy,
) -> GuidefieldDirectionEvidence:
    """
    Learn command-space guidefield direction from measured transmission.

    Positive and negative sides are represented by their best observed
    transmission. The two estimates are then compared with the same
    uncertainty-aware policy used by the optimizer.

    No statistically supported difference -> no profile change.
    """

    profile.validate()

    previous_sign = (
        profile.guidefield_forward_sign
    )

    positive = (
        _best_side_observation(
            scan,
            positive=True,
        )
    )

    negative = (
        _best_side_observation(
            scan,
            positive=False,
        )
    )

    if positive is None:
        return GuidefieldDirectionEvidence(
            previous_sign=previous_sign,
            proposed_sign=None,
            positive_coordinate_v=None,
            positive_transmission=None,
            negative_coordinate_v=(
                None
                if negative is None
                else negative[0]
            ),
            negative_transmission=(
                None
                if negative is None
                else negative[1].transmission
            ),
            comparison=None,
            profile_updated=False,
            reason=(
                "no_positive_difference_evidence"
            ),
        )

    if negative is None:
        return GuidefieldDirectionEvidence(
            previous_sign=previous_sign,
            proposed_sign=None,
            positive_coordinate_v=(
                positive[0]
            ),
            positive_transmission=(
                positive[1].transmission
            ),
            negative_coordinate_v=None,
            negative_transmission=None,
            comparison=None,
            profile_updated=False,
            reason=(
                "no_negative_difference_evidence"
            ),
        )

    # Baseline = negative side, candidate = positive side.
    comparison = compare_estimates(
        _estimate(
            negative[1],
            negative[2],
        ),
        _estimate(
            positive[1],
            positive[2],
        ),
        comparison_policy,
    )

    if (
        comparison.decision
        == ComparisonDecision.BETTER
    ):
        proposed_sign = 1
        reason = (
            "positive_difference_statistically_better"
        )

    elif (
        comparison.decision
        == ComparisonDecision.WORSE
    ):
        proposed_sign = -1
        reason = (
            "negative_difference_statistically_better"
        )

    else:
        proposed_sign = None
        reason = (
            "difference_signs_indistinguishable"
        )

    updated = False

    if (
        proposed_sign is not None
        and proposed_sign
        != previous_sign
    ):
        profile.set_guidefield_forward_sign(
            proposed_sign
        )

        updated = True

    return GuidefieldDirectionEvidence(
        previous_sign=previous_sign,
        proposed_sign=(
            proposed_sign
        ),
        positive_coordinate_v=(
            positive[0]
        ),
        positive_transmission=(
            positive[1].transmission
        ),
        negative_coordinate_v=(
            negative[0]
        ),
        negative_transmission=(
            negative[1].transmission
        ),
        comparison=comparison,
        profile_updated=updated,
        reason=reason,
    )


def optimize_end_electrode_coordinates(
    adapter,
    current_state: MachineState,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    *,
    policy: EndElectrodeCoordinatePolicy,
    noise_floor_a: float | None = None,
    logger=None,
    maintenance_hook: Callable[
        [MachineState],
        MachineState,
    ] | None = None,
) -> EndElectrodeCoordinateResult:
    """
    Optimize HV1/HV4 in differential/common-mode coordinates.
    """

    current_state.validate()

    if current_state.cup != 3:
        raise ValueError(
            "End-electrode coordinate optimization requires Cup 3"
        )

    initial_state = (
        current_state
    )

    working_state = (
        current_state
    )

    scans: list[
        DerivedScanResult
    ] = []

    for pass_index in range(
        1,
        policy.passes + 1,
    ):
        # ----------------------------------------------------------
        # Difference at fixed common mode.
        # ----------------------------------------------------------

        feasible_minimum, feasible_maximum = (
            end_electrode_difference_bounds(
                working_state
            )
        )

        current_difference = (
            end_electrode_difference_command(
                working_state
            )
        )

        minimum, maximum = (
            _local_window(
                current=(
                    current_difference
                ),
                feasible_minimum=(
                    feasible_minimum
                ),
                feasible_maximum=(
                    feasible_maximum
                ),
                half_width=(
                    policy.difference_half_width_v
                ),
            )
        )

        difference_scan = (
            scan_derived_coordinate_transmission_1d(
                adapter,
                working_state,
                tracker,
                coordinate_name=(
                    "end_electrode_difference_v"
                ),
                minimum=minimum,
                maximum=maximum,
                coordinate_reader=(
                    end_electrode_difference_command
                ),
                command_builder=(
                    end_electrode_difference_builder
                ),
                affected_parameters=(
                    "deceleration_voltage_v",
                    "acceleration_voltage_v",
                ),
                scan_policy=(
                    policy.difference_scan
                ),
                settling_policies=(
                    settling_policies
                ),
                measurement_policy=(
                    measurement_policy
                ),
                comparison_policy=(
                    comparison_policy
                ),
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
            difference_scan.final_state
        )

        scans.append(
            difference_scan
        )

        # ----------------------------------------------------------
        # Common mode at fixed best difference.
        # ----------------------------------------------------------

        feasible_minimum, feasible_maximum = (
            end_electrode_common_bounds(
                working_state
            )
        )

        current_common = (
            end_electrode_common_command(
                working_state
            )
        )

        minimum, maximum = (
            _local_window(
                current=current_common,
                feasible_minimum=(
                    feasible_minimum
                ),
                feasible_maximum=(
                    feasible_maximum
                ),
                half_width=(
                    policy.common_half_width_v
                ),
            )
        )

        common_scan = (
            scan_derived_coordinate_transmission_1d(
                adapter,
                working_state,
                tracker,
                coordinate_name=(
                    "end_electrode_common_v"
                ),
                minimum=minimum,
                maximum=maximum,
                coordinate_reader=(
                    end_electrode_common_command
                ),
                command_builder=(
                    end_electrode_common_builder
                ),
                affected_parameters=(
                    "deceleration_voltage_v",
                    "acceleration_voltage_v",
                ),
                scan_policy=(
                    policy.common_scan
                ),
                settling_policies=(
                    settling_policies
                ),
                measurement_policy=(
                    measurement_policy
                ),
                comparison_policy=(
                    comparison_policy
                ),
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
            common_scan.final_state
        )

        scans.append(
            common_scan
        )

        if logger is not None:
            logger.log_event(
                "cup3_end_coordinate_pass_completed",
                {
                    "pass": (
                        pass_index
                    ),
                    "difference_v": (
                        end_electrode_difference_command(
                            working_state
                        )
                    ),
                    "common_v": (
                        end_electrode_common_command(
                            working_state
                        )
                    ),
                },
            )

    return EndElectrodeCoordinateResult(
        initial_state=initial_state,
        scans=tuple(
            scans
        ),
        final_state=working_state,
    )


def optimize_guidefield_coordinates(
    adapter,
    current_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    *,
    policy: GuidefieldCoordinatePolicy,
    noise_floor_a: float | None = None,
    logger=None,
    maintenance_hook: Callable[
        [MachineState],
        MachineState,
    ] | None = None,
) -> GuidefieldCoordinateResult:
    """
    Optimize GF1/GF2 in differential/common-mode coordinates.

    If no forward direction is known, the differential window is expanded
    to include both signs whenever hardware bounds permit. Direction is
    learned only from uncertainty-supported transmission evidence.
    """

    current_state.validate()
    profile.validate()

    if current_state.cup != 3:
        raise ValueError(
            "Guidefield coordinate optimization requires Cup 3"
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

    initial_state = (
        current_state
    )

    working_state = (
        current_state
    )

    scans: list[
        DerivedScanResult
    ] = []

    evidence: list[
        GuidefieldDirectionEvidence
    ] = []

    for pass_index in range(
        1,
        policy.passes + 1,
    ):
        # ----------------------------------------------------------
        # Difference at fixed common mode.
        # ----------------------------------------------------------

        feasible_minimum, feasible_maximum = (
            guidefield_difference_bounds(
                working_state
            )
        )

        current_difference = (
            guidefield_difference_command(
                working_state
            )
        )

        probe_both_signs = (
            policy.probe_both_signs_when_unknown
            and profile.guidefield_forward_sign
            is None
        )

        minimum, maximum = (
            _local_window(
                current=(
                    current_difference
                ),
                feasible_minimum=(
                    feasible_minimum
                ),
                feasible_maximum=(
                    feasible_maximum
                ),
                half_width=(
                    policy.difference_half_width_v
                ),
                include_both_signs=(
                    probe_both_signs
                ),
            )
        )

        difference_scan = (
            scan_derived_coordinate_transmission_1d(
                adapter,
                working_state,
                tracker,
                coordinate_name=(
                    "guidefield_difference_v"
                ),
                minimum=minimum,
                maximum=maximum,
                coordinate_reader=(
                    guidefield_difference_command
                ),
                command_builder=(
                    guidefield_difference_builder
                ),
                affected_parameters=(
                    "guidefield1_voltage_v",
                    "guidefield2_voltage_v",
                ),
                scan_policy=(
                    policy.difference_scan
                ),
                settling_policies=(
                    settling_policies
                ),
                measurement_policy=(
                    measurement_policy
                ),
                comparison_policy=(
                    comparison_policy
                ),
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
            difference_scan.final_state
        )

        scans.append(
            difference_scan
        )

        if policy.learn_forward_sign:
            direction = (
                learn_guidefield_direction_from_scan(
                    profile,
                    difference_scan,
                    comparison_policy,
                )
            )

            evidence.append(
                direction
            )

            if logger is not None:
                logger.log_event(
                    "guidefield_direction_evidence",
                    direction,
                )

        # ----------------------------------------------------------
        # Common mode at fixed best difference.
        # ----------------------------------------------------------

        feasible_minimum, feasible_maximum = (
            guidefield_common_bounds(
                working_state
            )
        )

        current_common = (
            guidefield_common_command(
                working_state
            )
        )

        minimum, maximum = (
            _local_window(
                current=current_common,
                feasible_minimum=(
                    feasible_minimum
                ),
                feasible_maximum=(
                    feasible_maximum
                ),
                half_width=(
                    policy.common_half_width_v
                ),
            )
        )

        common_scan = (
            scan_derived_coordinate_transmission_1d(
                adapter,
                working_state,
                tracker,
                coordinate_name=(
                    "guidefield_common_v"
                ),
                minimum=minimum,
                maximum=maximum,
                coordinate_reader=(
                    guidefield_common_command
                ),
                command_builder=(
                    guidefield_common_builder
                ),
                affected_parameters=(
                    "guidefield1_voltage_v",
                    "guidefield2_voltage_v",
                ),
                scan_policy=(
                    policy.common_scan
                ),
                settling_policies=(
                    settling_policies
                ),
                measurement_policy=(
                    measurement_policy
                ),
                comparison_policy=(
                    comparison_policy
                ),
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
            common_scan.final_state
        )

        scans.append(
            common_scan
        )

        if logger is not None:
            logger.log_event(
                "cup3_guidefield_coordinate_pass_completed",
                {
                    "pass": (
                        pass_index
                    ),
                    "difference_v": (
                        guidefield_difference_command(
                            working_state
                        )
                    ),
                    "common_v": (
                        guidefield_common_command(
                            working_state
                        )
                    ),
                    "learned_forward_sign": (
                        profile.guidefield_forward_sign
                    ),
                },
            )

    return GuidefieldCoordinateResult(
        initial_state=initial_state,
        scans=tuple(
            scans
        ),
        direction_evidence=tuple(
            evidence
        ),
        final_state=working_state,
    )