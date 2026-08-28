from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping

from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.reference import (
    SourceReference,
    TransmissionResult,
    transmission_from_reference,
)
from sirius.settling import SettlingPolicy
from sirius.state import (
    MachineState,
    utc_now_iso,
)
from sirius.transition import (
    AppliedStateResult,
    apply_state,
    capture_readbacks,
)


FINAL_CUP_SEQUENCE = (
    1,
    2,
    3,
    4,
    5,
    6,
    1,
)


class FinalCharacterizationError(
    RuntimeError
):
    pass


class FrozenConfigurationChangedError(
    FinalCharacterizationError
):
    pass


class InvalidInitialCup1ReferenceError(
    FinalCharacterizationError
):
    pass


class InvalidFinalCup1ReferenceError(
    FinalCharacterizationError
):
    pass


@dataclass(frozen=True)
class FinalCharacterizationPolicy:
    """
    Final beamline characterization under one frozen machine setting.

    cup_settle_s is intentionally an operational timing parameter rather
    than a physics constant.
    """

    cup_settle_s: float = 1.0

    sequence: tuple[
        int,
        ...
    ] = FINAL_CUP_SEQUENCE

    def __post_init__(self) -> None:
        if not math.isfinite(
            float(
                self.cup_settle_s
            )
        ):
            raise ValueError(
                "cup_settle_s must be finite"
            )

        if self.cup_settle_s < 0:
            raise ValueError(
                "cup_settle_s must be non-negative"
            )

        if tuple(
            self.sequence
        ) != FINAL_CUP_SEQUENCE:
            raise ValueError(
                "Final characterization must use canonical sequence "
                "Cup1->Cup2->Cup3->Cup4->Cup5->Cup6->Cup1"
            )


@dataclass(frozen=True)
class FrozenConfiguration:
    mass_u: float

    parameters: dict[
        str,
        float,
    ]

    rfq: object

    source_state_id: str


@dataclass(frozen=True)
class FinalCharacterizationPoint:
    sequence_index: int
    cup: int

    transition: AppliedStateResult

    state: MachineState

    measurement: BeamMeasurement

    measured_at_monotonic_s: float
    measured_at_utc: str

    relative_to_initial_cup1: (
        TransmissionResult | None
    )

    @property
    def below_noise_floor(
        self,
    ) -> bool:
        return bool(
            self.measurement.below_noise_floor
        )


@dataclass(frozen=True)
class Cup1DriftResult:
    start_current_a: float
    end_current_a: float

    elapsed_s: float

    ratio: float
    ratio_sem: float

    drift_fraction: float
    drift_fraction_sem: float

    drift_percent: float
    drift_percent_sem: float


@dataclass(frozen=True)
class FinalCharacterizationResult:
    frozen_configuration: (
        FrozenConfiguration
    )

    frozen_state: MachineState

    points: tuple[
        FinalCharacterizationPoint,
        ...
    ]

    initial_cup1_reference: SourceReference

    cup1_drift: Cup1DriftResult

    final_state: MachineState

    @property
    def transmissions_by_cup(
        self,
    ) -> dict[
        int,
        TransmissionResult,
    ]:
        """
        Return the final frozen-settings transmissions for Cups 2..6.

        The second Cup-1 point is excluded because it is a drift check,
        not an additional beamline transmission.
        """

        result: dict[
            int,
            TransmissionResult,
        ] = {}

        for point in self.points:
            if point.cup == 1:
                continue

            if (
                point.relative_to_initial_cup1
                is not None
            ):
                result[
                    point.cup
                ] = (
                    point.relative_to_initial_cup1
                )

        return result

    @property
    def all_downstream_above_noise(
        self,
    ) -> bool:
        return all(
            not point.measurement.below_noise_floor
            for point
            in self.points
            if point.cup != 1
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


def _freeze_configuration(
    state: MachineState,
) -> FrozenConfiguration:
    state.validate()

    return FrozenConfiguration(
        mass_u=float(
            state.mass_u
        ),
        parameters={
            name: float(value)
            for name, value
            in state.parameters.items()
        },
        rfq=deepcopy(
            state.rfq
        ),
        source_state_id=(
            state.state_id
        ),
    )


def _assert_frozen_configuration(
    state: MachineState,
    frozen: FrozenConfiguration,
) -> None:
    """
    Hard characterization guard.

    Readbacks are allowed to move.
    Cup selection is allowed to move.

    Parameter commands and RFQ configuration are not.
    """

    state.validate()

    if not math.isclose(
        state.mass_u,
        frozen.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise FrozenConfigurationChangedError(
            "Ion mass changed during final characterization"
        )

    frozen_names = set(
        frozen.parameters
    )

    current_names = set(
        state.parameters
    )

    if current_names != frozen_names:
        added = sorted(
            current_names
            - frozen_names
        )

        removed = sorted(
            frozen_names
            - current_names
        )

        raise FrozenConfigurationChangedError(
            "Machine parameter set changed during final characterization: "
            f"added={added}, removed={removed}"
        )

    for parameter_name in sorted(
        frozen.parameters
    ):
        expected = (
            frozen.parameters[
                parameter_name
            ]
        )

        actual = (
            state.parameters[
                parameter_name
            ]
        )

        if not _commands_equal(
            actual,
            expected,
        ):
            raise FrozenConfigurationChangedError(
                f"{parameter_name} changed during final characterization: "
                f"{expected} -> {actual}"
            )

    if state.rfq != frozen.rfq:
        raise FrozenConfigurationChangedError(
            "RFQ configuration changed during final characterization"
        )


def _cup_target_state(
    state: MachineState,
    cup: int,
    frozen: FrozenConfiguration,
) -> MachineState:
    if not 1 <= int(cup) <= 6:
        raise ValueError(
            "Cup must be between 1 and 6"
        )

    _assert_frozen_configuration(
        state,
        frozen,
    )

    target = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            frozen.parameters
        ),
        readbacks=dict(
            state.readbacks
        ),
        cup=int(
            cup
        ),
        stage=6,
        role="final_characterization",
        rfq=deepcopy(
            frozen.rfq
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata={
            **deepcopy(
                state.metadata
            ),
            "final_characterization": True,
            "frozen_source_state_id": (
                frozen.source_state_id
            ),
            "characterization_cup": int(
                cup
            ),
        },
    )

    target.validate()

    return target


def _select_cup_without_retuning(
    adapter,
    current_state: MachineState,
    cup: int,
    frozen: FrozenConfiguration,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> tuple[
    AppliedStateResult,
    MachineState,
]:
    """
    Change only the selected Faraday cup.

    apply_state() receives exactly the frozen machine commands, so any
    detected parameter change is a characterization failure.
    """

    target = _cup_target_state(
        current_state,
        cup,
        frozen,
    )

    transition = apply_state(
        adapter,
        current=current_state,
        target=target,
        settling_policies=(
            settling_policies
        ),
        select_target_cup=True,
    )

    observed = (
        transition.observed_state
    )

    if observed.cup != cup:
        raise FinalCharacterizationError(
            f"Requested Cup {cup}, but resulting state reports "
            f"Cup {observed.cup}"
        )

    _assert_frozen_configuration(
        observed,
        frozen,
    )

    observed = capture_readbacks(
        adapter,
        observed,
    )

    if observed.cup != cup:
        raise FinalCharacterizationError(
            f"Cup changed unexpectedly after readback capture: "
            f"expected {cup}, got {observed.cup}"
        )

    _assert_frozen_configuration(
        observed,
        frozen,
    )

    return (
        transition,
        observed,
    )


def _validate_initial_state(
    state: MachineState,
) -> None:
    state.validate()

    if state.cup != 6:
        raise ValueError(
            "Final characterization must start from the final Cup-6 state"
        )

    if state.stage not in (
        None,
        6,
    ):
        raise ValueError(
            "Final characterization requires stage 6 or no stage assignment"
        )

    if not state.parameters:
        raise ValueError(
            "Final characterization requires a non-empty machine state"
        )


def _cup1_drift(
    start_point: FinalCharacterizationPoint,
    end_point: FinalCharacterizationPoint,
    initial_reference: SourceReference,
) -> Cup1DriftResult:
    if start_point.cup != 1:
        raise ValueError(
            "Start drift point must be Cup 1"
        )

    if end_point.cup != 1:
        raise ValueError(
            "End drift point must be Cup 1"
        )

    ratio_result = (
        transmission_from_reference(
            1,
            end_point.measurement,
            initial_reference,
        )
    )

    elapsed = (
        end_point.measured_at_monotonic_s
        - start_point.measured_at_monotonic_s
    )

    ratio = float(
        ratio_result.transmission
    )

    ratio_sem = float(
        ratio_result.transmission_sem
    )

    drift_fraction = (
        ratio
        - 1.0
    )

    return Cup1DriftResult(
        start_current_a=(
            start_point.measurement.mean_a
        ),
        end_current_a=(
            end_point.measurement.mean_a
        ),
        elapsed_s=(
            max(
                0.0,
                float(
                    elapsed
                ),
            )
        ),
        ratio=ratio,
        ratio_sem=ratio_sem,
        drift_fraction=(
            drift_fraction
        ),
        drift_fraction_sem=(
            ratio_sem
        ),
        drift_percent=(
            drift_fraction
            * 100.0
        ),
        drift_percent_sem=(
            ratio_sem
            * 100.0
        ),
    )


def characterize_final_transmission(
    adapter,
    final_cup6_state: MachineState,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    *,
    policy: (
        FinalCharacterizationPolicy | None
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
    sleeper: Callable[
        [float],
        None,
    ] = time.sleep,
) -> FinalCharacterizationResult:
    """
    Characterize Cups 1..6 using ONE frozen final machine setting.

    Canonical sequence:

        Cup1 -> Cup2 -> Cup3 -> Cup4 -> Cup5 -> Cup6 -> Cup1

    No source-reference restoration is performed here because restoring
    the Cup-1 optimization state would violate the frozen-settings
    requirement.

    Only Faraday-cup selection may change.
    """

    active_policy = (
        policy
        if policy is not None
        else FinalCharacterizationPolicy()
    )

    _validate_initial_state(
        final_cup6_state
    )

    physical_state = (
        capture_readbacks(
            adapter,
            final_cup6_state,
        )
    )

    _validate_initial_state(
        physical_state
    )

    frozen = _freeze_configuration(
        physical_state
    )

    _assert_frozen_configuration(
        physical_state,
        frozen,
    )

    if logger is not None:
        logger.save_state(
            physical_state,
            "final_characterization_frozen",
        )

        logger.log_event(
            "final_characterization_started",
            {
                "source_state_id": (
                    frozen.source_state_id
                ),
                "mass_u": (
                    frozen.mass_u
                ),
                "sequence": list(
                    active_policy.sequence
                ),
                "cup_settle_s": (
                    active_policy.cup_settle_s
                ),
                "parameter_count": (
                    len(
                        frozen.parameters
                    )
                ),
            },
        )

    points: list[
        FinalCharacterizationPoint
    ] = []

    initial_reference: (
        SourceReference | None
    ) = None

    for sequence_index, cup in enumerate(
        active_policy.sequence,
        start=1,
    ):
        (
            transition,
            physical_state,
        ) = _select_cup_without_retuning(
            adapter,
            physical_state,
            cup,
            frozen,
            settling_policies,
        )

        if active_policy.cup_settle_s > 0:
            sleeper(
                active_policy.cup_settle_s
            )

        # One more guard immediately before measuring.
        _assert_frozen_configuration(
            physical_state,
            frozen,
        )

        measured_at_monotonic = (
            monotonic()
        )

        measured_at_utc = (
            utc_now()
        )

        measurement = measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=(
                noise_floor_a
            ),
        )

        relative: (
            TransmissionResult | None
        ) = None

        if sequence_index == 1:
            if cup != 1:
                raise FinalCharacterizationError(
                    "First final-characterization point must be Cup 1"
                )

            if (
                measurement.below_noise_floor
                or measurement.mean_a <= 0
            ):
                raise InvalidInitialCup1ReferenceError(
                    "Initial frozen-settings Cup-1 current is not a "
                    "valid transmission reference"
                )

            initial_reference = SourceReference(
                measurement=measurement,
                state_id=(
                    physical_state.state_id
                ),
                mass_u=(
                    physical_state.mass_u
                ),
                monotonic_s=(
                    measured_at_monotonic
                ),
                created_at_utc=(
                    measured_at_utc
                ),
            )

        else:
            if initial_reference is None:
                raise FinalCharacterizationError(
                    "Initial Cup-1 reference was not created"
                )

            relative = (
                transmission_from_reference(
                    cup,
                    measurement,
                    initial_reference,
                )
            )

        point = FinalCharacterizationPoint(
            sequence_index=(
                sequence_index
            ),
            cup=(
                cup
            ),
            transition=(
                transition
            ),
            state=(
                physical_state
            ),
            measurement=(
                measurement
            ),
            measured_at_monotonic_s=(
                measured_at_monotonic
            ),
            measured_at_utc=(
                measured_at_utc
            ),
            relative_to_initial_cup1=(
                relative
            ),
        )

        points.append(
            point
        )

        if logger is not None:
            logger.log_state_transition(
                transition
            )

            logger.log_measurement(
                measurement,
                cup=cup,
                state_id=(
                    physical_state.state_id
                ),
                purpose=(
                    "final_characterization"
                ),
            )

            if relative is not None:
                logger.log_transmission(
                    relative
                )

            logger.log_event(
                "final_characterization_point",
                {
                    "sequence_index": (
                        sequence_index
                    ),
                    "cup": (
                        cup
                    ),
                    "state_id": (
                        physical_state.state_id
                    ),
                    "current_a": (
                        measurement.mean_a
                    ),
                    "sem_a": (
                        measurement.sem_a
                    ),
                    "below_noise_floor": (
                        measurement.below_noise_floor
                    ),
                    "relative_to_initial_cup1": (
                        None
                        if relative is None
                        else relative.transmission
                    ),
                    "relative_sem": (
                        None
                        if relative is None
                        else relative.transmission_sem
                    ),
                },
            )

    if initial_reference is None:
        raise FinalCharacterizationError(
            "Final characterization produced no Cup-1 reference"
        )

    start_point = (
        points[0]
    )

    end_point = (
        points[-1]
    )

    if end_point.cup != 1:
        raise FinalCharacterizationError(
            "Final characterization did not end at Cup 1"
        )

    if (
        end_point.measurement.below_noise_floor
        or end_point.measurement.mean_a <= 0
    ):
        raise InvalidFinalCup1ReferenceError(
            "Final Cup-1 drift measurement is not a valid source signal"
        )

    drift = _cup1_drift(
        start_point,
        end_point,
        initial_reference,
    )

    _assert_frozen_configuration(
        end_point.state,
        frozen,
    )

    result = FinalCharacterizationResult(
        frozen_configuration=(
            frozen
        ),
        frozen_state=(
            points[0].state
        ),
        points=tuple(
            points
        ),
        initial_cup1_reference=(
            initial_reference
        ),
        cup1_drift=(
            drift
        ),
        final_state=(
            end_point.state
        ),
    )

    if logger is not None:
        transmissions = (
            result.transmissions_by_cup
        )

        logger.log_event(
            "final_characterization_completed",
            {
                "source_state_id": (
                    frozen.source_state_id
                ),
                "final_state_id": (
                    end_point.state.state_id
                ),
                "cup1_start_a": (
                    drift.start_current_a
                ),
                "cup1_end_a": (
                    drift.end_current_a
                ),
                "cup1_drift_percent": (
                    drift.drift_percent
                ),
                "cup1_drift_percent_sem": (
                    drift.drift_percent_sem
                ),
                "elapsed_s": (
                    drift.elapsed_s
                ),
                "all_downstream_above_noise": (
                    result.all_downstream_above_noise
                ),
                "transmissions": {
                    str(cup): {
                        "transmission": (
                            transmission.transmission
                        ),
                        "transmission_sem": (
                            transmission.transmission_sem
                        ),
                        "transmission_percent": (
                            transmission.transmission_percent
                        ),
                    }
                    for cup, transmission
                    in transmissions.items()
                },
            },
        )

    return result