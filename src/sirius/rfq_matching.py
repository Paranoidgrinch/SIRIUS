from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from sirius.rfq_model import (
    LC_MAX_CAPACITANCE_PF,
    LC_MAX_INDUCTANCE_UH,
    RFQ_OPERATIONAL_Q_MAX,
    ideal_lc_resonance_frequency_hz,
    mathieu_q_from_vpp,
    rfq_vpp_for_q,
    validate_q_target,
)


class RFQHardware(Protocol):
    """
    SIRIUS-internal RFQ hardware contract.

    A later FLAVIA adapter will translate these calls to the real
    backend / LC controller / function generator / scope interfaces.
    """

    def set_frequency_hz(
        self,
        value: float,
    ) -> None:
        ...

    def set_generator_amplitude_vpp(
        self,
        value: float,
    ) -> None:
        ...

    def set_matching(
        self,
        inductance_uh: float,
        capacitance_pf: float,
    ) -> None:
        ...

    def read_rfq_vpp(
        self,
    ) -> float:
        ...


@dataclass(frozen=True)
class LCSetting:
    inductance_uh: float
    capacitance_pf: float

    def validate(self) -> None:
        if not math.isfinite(
            float(self.inductance_uh)
        ):
            raise ValueError(
                "Inductance must be finite"
            )

        if not math.isfinite(
            float(self.capacitance_pf)
        ):
            raise ValueError(
                "Capacitance must be finite"
            )

        if self.inductance_uh <= 0:
            raise ValueError(
                "Inductance must be greater than zero"
            )

        if self.capacitance_pf <= 0:
            raise ValueError(
                "Capacitance must be greater than zero"
            )

        if (
            self.inductance_uh
            > LC_MAX_INDUCTANCE_UH
        ):
            raise ValueError(
                "Inductance exceeds LC hardware range"
            )

        if (
            self.capacitance_pf
            > LC_MAX_CAPACITANCE_PF
        ):
            raise ValueError(
                "Capacitance exceeds LC hardware range"
            )


@dataclass(frozen=True)
class RFQMatchingPolicy:
    """
    Conservative resonance-search policy.

    No assumed RF voltage gain is required. The scope measurement
    is used directly at every tested point.
    """

    probe_generator_vpp: float

    requested_frequency_hz: float
    frequency_half_width_hz: float

    coarse_frequency_step_hz: float
    fine_frequency_step_hz: float

    top_lc_candidates: int = 5

    measurements_per_point: int = 3
    measurement_interval_s: float = 0.0

    hardware_settle_s: float = 0.0

    q_abort_limit: float = (
        RFQ_OPERATIONAL_Q_MAX
    )

    leave_probe_on: bool = False

    def __post_init__(self) -> None:
        positive_values = (
            (
                "probe_generator_vpp",
                self.probe_generator_vpp,
            ),
            (
                "requested_frequency_hz",
                self.requested_frequency_hz,
            ),
            (
                "frequency_half_width_hz",
                self.frequency_half_width_hz,
            ),
            (
                "coarse_frequency_step_hz",
                self.coarse_frequency_step_hz,
            ),
            (
                "fine_frequency_step_hz",
                self.fine_frequency_step_hz,
            ),
        )

        for name, value in positive_values:
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

        if (
            self.fine_frequency_step_hz
            >= self.coarse_frequency_step_hz
        ):
            raise ValueError(
                "Fine frequency step must be smaller than coarse step"
            )

        if self.top_lc_candidates < 1:
            raise ValueError(
                "top_lc_candidates must be at least 1"
            )

        if self.measurements_per_point < 1:
            raise ValueError(
                "measurements_per_point must be at least 1"
            )

        if self.measurement_interval_s < 0:
            raise ValueError(
                "measurement_interval_s must be non-negative"
            )

        if self.hardware_settle_s < 0:
            raise ValueError(
                "hardware_settle_s must be non-negative"
            )

        if not (
            0
            < self.q_abort_limit
            <= RFQ_OPERATIONAL_Q_MAX
        ):
            raise ValueError(
                "q_abort_limit must be within 0..0.9"
            )


@dataclass(frozen=True)
class RFQResonancePoint:
    setting: LCSetting

    frequency_hz: float

    generator_amplitude_vpp: float

    measured_rfq_vpp: float
    measured_q: float

    samples_vpp: tuple[float, ...]

    scan_level: str


@dataclass(frozen=True)
class RFQMatchingResult:
    mass_u: float

    best_setting: LCSetting
    best_frequency_hz: float

    probe_generator_vpp: float

    best_measured_rfq_vpp: float
    best_measured_q: float

    points: tuple[
        RFQResonancePoint,
        ...
    ]


@dataclass(frozen=True)
class RFQTargetQPolicy:
    """
    Closed-loop generator-amplitude adjustment at an already matched
    RFQ resonance.

    generator_max_vpp is deliberately mandatory. SIRIUS must not invent
    a real hardware amplitude limit.
    """

    generator_max_vpp: float

    initial_generator_vpp: float

    relative_q_tolerance: float = 0.02

    maximum_scale_up: float = 1.5
    maximum_scale_down: float = 0.5

    max_iterations: int = 10

    measurements_per_iteration: int = 3
    measurement_interval_s: float = 0.0
    hardware_settle_s: float = 0.0

    q_abort_limit: float = (
        RFQ_OPERATIONAL_Q_MAX
    )

    def __post_init__(self) -> None:
        for name, value in (
            (
                "generator_max_vpp",
                self.generator_max_vpp,
            ),
            (
                "initial_generator_vpp",
                self.initial_generator_vpp,
            ),
            (
                "relative_q_tolerance",
                self.relative_q_tolerance,
            ),
            (
                "maximum_scale_up",
                self.maximum_scale_up,
            ),
            (
                "maximum_scale_down",
                self.maximum_scale_down,
            ),
        ):
            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

        if self.generator_max_vpp <= 0:
            raise ValueError(
                "generator_max_vpp must be greater than zero"
            )

        if not (
            0
            < self.initial_generator_vpp
            <= self.generator_max_vpp
        ):
            raise ValueError(
                "initial_generator_vpp must lie within generator range"
            )

        if self.relative_q_tolerance <= 0:
            raise ValueError(
                "relative_q_tolerance must be greater than zero"
            )

        if self.maximum_scale_up <= 1:
            raise ValueError(
                "maximum_scale_up must be greater than 1"
            )

        if not (
            0
            < self.maximum_scale_down
            < 1
        ):
            raise ValueError(
                "maximum_scale_down must lie between 0 and 1"
            )

        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1"
            )

        if self.measurements_per_iteration < 1:
            raise ValueError(
                "measurements_per_iteration must be at least 1"
            )

        if self.measurement_interval_s < 0:
            raise ValueError(
                "measurement_interval_s must be non-negative"
            )

        if self.hardware_settle_s < 0:
            raise ValueError(
                "hardware_settle_s must be non-negative"
            )

        if not (
            0
            < self.q_abort_limit
            <= RFQ_OPERATIONAL_Q_MAX
        ):
            raise ValueError(
                "q_abort_limit must lie within 0..0.9"
            )


@dataclass(frozen=True)
class RFQTargetQIteration:
    iteration: int

    generator_amplitude_vpp: float

    measured_rfq_vpp: float
    measured_q: float

    samples_vpp: tuple[float, ...]


@dataclass(frozen=True)
class RFQTargetQResult:
    mass_u: float

    target_q: float

    setting: LCSetting
    frequency_hz: float

    generator_amplitude_vpp: float

    measured_rfq_vpp: float
    measured_q: float

    required_rfq_vpp: float

    iterations: tuple[
        RFQTargetQIteration,
        ...
    ]


class RFQMatchingError(RuntimeError):
    pass


class RFQUnsafeAmplitudeError(
    RFQMatchingError
):
    pass


class RFQNoSignalError(
    RFQMatchingError
):
    pass


class RFQTargetQNotReachedError(
    RFQMatchingError
):
    pass


def _positive_finite(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return value


def _frequency_grid(
    minimum_hz: float,
    maximum_hz: float,
    step_hz: float,
) -> tuple[float, ...]:
    minimum = _positive_finite(
        "frequency minimum",
        minimum_hz,
    )

    maximum = _positive_finite(
        "frequency maximum",
        maximum_hz,
    )

    step = _positive_finite(
        "frequency step",
        step_hz,
    )

    if maximum < minimum:
        raise ValueError(
            "Frequency maximum must not be below minimum"
        )

    values: list[float] = []

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
                "Frequency grid is unreasonably large"
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

    return tuple(values)


def rank_lc_candidates(
    candidates: Iterable[
        LCSetting
    ],
    requested_frequency_hz: float,
) -> tuple[LCSetting, ...]:
    requested = _positive_finite(
        "requested frequency",
        requested_frequency_hz,
    )

    candidates = tuple(
        candidates
    )

    if not candidates:
        raise ValueError(
            "At least one LC candidate is required"
        )

    for setting in candidates:
        setting.validate()

    return tuple(
        sorted(
            candidates,
            key=lambda setting: abs(
                ideal_lc_resonance_frequency_hz(
                    setting.inductance_uh,
                    setting.capacitance_pf,
                )
                - requested
            ),
        )
    )


def _measure_vpp(
    hardware: RFQHardware,
    *,
    sample_count: int,
    interval_s: float,
    sleeper: Callable[
        [float],
        None,
    ],
) -> tuple[
    float,
    tuple[float, ...],
]:
    samples: list[float] = []

    for index in range(
        sample_count
    ):
        value = float(
            hardware.read_rfq_vpp()
        )

        if not math.isfinite(value):
            raise RFQMatchingError(
                "Scope returned non-finite RFQ Vpp"
            )

        if value < 0:
            raise RFQMatchingError(
                "Scope returned negative RFQ Vpp"
            )

        samples.append(
            value
        )

        if (
            index
            < sample_count - 1
            and interval_s > 0
        ):
            sleeper(
                interval_s
            )

    # Median is intentionally used instead of a single read to suppress
    # occasional scope / communication outliers.
    measured = float(
        statistics.median(
            samples
        )
    )

    return (
        measured,
        tuple(samples),
    )


def _check_measured_q(
    mass_u: float,
    frequency_hz: float,
    measured_vpp: float,
    *,
    q_abort_limit: float,
) -> float:
    q_value = mathieu_q_from_vpp(
        mass_u,
        frequency_hz,
        measured_vpp,
    )

    if q_value > q_abort_limit:
        raise RFQUnsafeAmplitudeError(
            "Measured RFQ amplitude exceeds the configured "
            f"q safety limit: q={q_value:.6f}, "
            f"limit={q_abort_limit:.6f}"
        )

    return q_value


def search_rfq_resonance(
    hardware: RFQHardware,
    mass_u: float,
    lc_candidates: Iterable[
        LCSetting
    ],
    policy: RFQMatchingPolicy,
    *,
    logger=None,
    sleeper: Callable[
        [float],
        None,
    ] = time.sleep,
) -> RFQMatchingResult:
    """
    Search the RFQ resonance using a deliberately low probe amplitude.

    Procedure:
      1. Rank L/C candidates by ideal LC resonance proximity.
      2. Coarse-frequency scan the best candidates.
      3. Choose the highest measured RFQ Vpp.
      4. Fine-frequency scan around that maximum.
      5. Leave the best L/C and frequency selected.
      6. By default return generator amplitude to zero.

    Scope Vpp is authoritative throughout.
    """

    mass = _positive_finite(
        "ion mass",
        mass_u,
    )

    ranked = rank_lc_candidates(
        lc_candidates,
        policy.requested_frequency_hz,
    )

    selected_candidates = ranked[
        :policy.top_lc_candidates
    ]

    minimum_frequency = max(
        1.0,
        policy.requested_frequency_hz
        - policy.frequency_half_width_hz,
    )

    maximum_frequency = (
        policy.requested_frequency_hz
        + policy.frequency_half_width_hz
    )

    coarse_grid = _frequency_grid(
        minimum_frequency,
        maximum_frequency,
        policy.coarse_frequency_step_hz,
    )

    points: list[
        RFQResonancePoint
    ] = []

    if logger is not None:
        logger.log_event(
            "rfq_matching_started",
            {
                "mass_u": mass,
                "requested_frequency_hz": (
                    policy.requested_frequency_hz
                ),
                "probe_generator_vpp": (
                    policy.probe_generator_vpp
                ),
                "lc_candidates_total": (
                    len(ranked)
                ),
                "lc_candidates_tested": (
                    len(selected_candidates)
                ),
            },
        )

    try:
        hardware.set_generator_amplitude_vpp(
            0.0
        )

        for setting in selected_candidates:
            setting.validate()

            hardware.set_matching(
                setting.inductance_uh,
                setting.capacitance_pf,
            )

            if policy.hardware_settle_s > 0:
                sleeper(
                    policy.hardware_settle_s
                )

            for frequency in coarse_grid:
                hardware.set_frequency_hz(
                    frequency
                )

                hardware.set_generator_amplitude_vpp(
                    policy.probe_generator_vpp
                )

                if policy.hardware_settle_s > 0:
                    sleeper(
                        policy.hardware_settle_s
                    )

                measured_vpp, samples = (
                    _measure_vpp(
                        hardware,
                        sample_count=(
                            policy.measurements_per_point
                        ),
                        interval_s=(
                            policy.measurement_interval_s
                        ),
                        sleeper=sleeper,
                    )
                )

                try:
                    measured_q = (
                        _check_measured_q(
                            mass,
                            frequency,
                            measured_vpp,
                            q_abort_limit=(
                                policy.q_abort_limit
                            ),
                        )
                    )

                except RFQUnsafeAmplitudeError:
                    hardware.set_generator_amplitude_vpp(
                        0.0
                    )
                    raise

                point = RFQResonancePoint(
                    setting=setting,
                    frequency_hz=float(
                        frequency
                    ),
                    generator_amplitude_vpp=float(
                        policy.probe_generator_vpp
                    ),
                    measured_rfq_vpp=(
                        measured_vpp
                    ),
                    measured_q=(
                        measured_q
                    ),
                    samples_vpp=(
                        samples
                    ),
                    scan_level="coarse",
                )

                points.append(
                    point
                )

                if logger is not None:
                    logger.log_event(
                        "rfq_resonance_observation",
                        point,
                    )

        if not points:
            raise RFQNoSignalError(
                "RFQ resonance search produced no measurements"
            )

        coarse_best = max(
            points,
            key=lambda point: (
                point.measured_rfq_vpp
            ),
        )

        if coarse_best.measured_rfq_vpp <= 0:
            raise RFQNoSignalError(
                "No RFQ signal detected during resonance search"
            )

        # Fine scan only the best L/C combination.
        hardware.set_generator_amplitude_vpp(
            0.0
        )

        hardware.set_matching(
            coarse_best.setting.inductance_uh,
            coarse_best.setting.capacitance_pf,
        )

        fine_minimum = max(
            1.0,
            coarse_best.frequency_hz
            - policy.coarse_frequency_step_hz,
        )

        fine_maximum = (
            coarse_best.frequency_hz
            + policy.coarse_frequency_step_hz
        )

        fine_grid = _frequency_grid(
            fine_minimum,
            fine_maximum,
            policy.fine_frequency_step_hz,
        )

        for frequency in fine_grid:
            hardware.set_frequency_hz(
                frequency
            )

            hardware.set_generator_amplitude_vpp(
                policy.probe_generator_vpp
            )

            if policy.hardware_settle_s > 0:
                sleeper(
                    policy.hardware_settle_s
                )

            measured_vpp, samples = (
                _measure_vpp(
                    hardware,
                    sample_count=(
                        policy.measurements_per_point
                    ),
                    interval_s=(
                        policy.measurement_interval_s
                    ),
                    sleeper=sleeper,
                )
            )

            try:
                measured_q = (
                    _check_measured_q(
                        mass,
                        frequency,
                        measured_vpp,
                        q_abort_limit=(
                            policy.q_abort_limit
                        ),
                    )
                )

            except RFQUnsafeAmplitudeError:
                hardware.set_generator_amplitude_vpp(
                    0.0
                )
                raise

            point = RFQResonancePoint(
                setting=(
                    coarse_best.setting
                ),
                frequency_hz=float(
                    frequency
                ),
                generator_amplitude_vpp=float(
                    policy.probe_generator_vpp
                ),
                measured_rfq_vpp=(
                    measured_vpp
                ),
                measured_q=(
                    measured_q
                ),
                samples_vpp=(
                    samples
                ),
                scan_level="fine",
            )

            points.append(
                point
            )

            if logger is not None:
                logger.log_event(
                    "rfq_resonance_observation",
                    point,
                )

        best = max(
            points,
            key=lambda point: (
                point.measured_rfq_vpp
            ),
        )

        # Freeze matching network and frequency at the measured maximum.
        hardware.set_matching(
            best.setting.inductance_uh,
            best.setting.capacitance_pf,
        )

        hardware.set_frequency_hz(
            best.frequency_hz
        )

        if policy.leave_probe_on:
            hardware.set_generator_amplitude_vpp(
                policy.probe_generator_vpp
            )
        else:
            hardware.set_generator_amplitude_vpp(
                0.0
            )

        result = RFQMatchingResult(
            mass_u=mass,
            best_setting=(
                best.setting
            ),
            best_frequency_hz=(
                best.frequency_hz
            ),
            probe_generator_vpp=(
                policy.probe_generator_vpp
            ),
            best_measured_rfq_vpp=(
                best.measured_rfq_vpp
            ),
            best_measured_q=(
                best.measured_q
            ),
            points=tuple(
                points
            ),
        )

        if logger is not None:
            logger.log_event(
                "rfq_matching_completed",
                result,
            )

        return result

    except Exception:
        # Safe RF default on every failed search.
        hardware.set_generator_amplitude_vpp(
            0.0
        )
        raise


def set_target_q(
    hardware: RFQHardware,
    matching: RFQMatchingResult,
    target_q: float,
    policy: RFQTargetQPolicy,
    *,
    logger=None,
    sleeper: Callable[
        [float],
        None,
    ] = time.sleep,
) -> RFQTargetQResult:
    """
    Adjust generator amplitude using measured RFQ Vpp until the requested
    Mathieu q is reached.

    Frequency and L/C remain fixed at the previously measured resonance.

    Adjustment is deliberately gradual: each iteration may increase the
    generator command by at most maximum_scale_up.

    Any measured q above q_abort_limit immediately disables RF output.
    """

    target = validate_q_target(
        target_q
    )

    required_vpp = rfq_vpp_for_q(
        matching.mass_u,
        matching.best_frequency_hz,
        target,
    )

    hardware.set_generator_amplitude_vpp(
        0.0
    )

    hardware.set_matching(
        matching.best_setting.inductance_uh,
        matching.best_setting.capacitance_pf,
    )

    hardware.set_frequency_hz(
        matching.best_frequency_hz
    )

    generator_command = float(
        policy.initial_generator_vpp
    )

    iterations: list[
        RFQTargetQIteration
    ] = []

    if logger is not None:
        logger.log_event(
            "rfq_target_q_started",
            {
                "mass_u": (
                    matching.mass_u
                ),
                "target_q": target,
                "required_rfq_vpp": (
                    required_vpp
                ),
                "frequency_hz": (
                    matching.best_frequency_hz
                ),
                "setting": (
                    matching.best_setting
                ),
            },
        )

    try:
        for iteration_index in range(
            1,
            policy.max_iterations + 1,
        ):
            if (
                generator_command
                > policy.generator_max_vpp
            ):
                raise RFQTargetQNotReachedError(
                    "Required generator amplitude exceeds configured "
                    "hardware maximum"
                )

            hardware.set_generator_amplitude_vpp(
                generator_command
            )

            if policy.hardware_settle_s > 0:
                sleeper(
                    policy.hardware_settle_s
                )

            measured_vpp, samples = (
                _measure_vpp(
                    hardware,
                    sample_count=(
                        policy.measurements_per_iteration
                    ),
                    interval_s=(
                        policy.measurement_interval_s
                    ),
                    sleeper=sleeper,
                )
            )

            try:
                measured_q = (
                    _check_measured_q(
                        matching.mass_u,
                        matching.best_frequency_hz,
                        measured_vpp,
                        q_abort_limit=(
                            policy.q_abort_limit
                        ),
                    )
                )

            except RFQUnsafeAmplitudeError:
                hardware.set_generator_amplitude_vpp(
                    0.0
                )
                raise

            iteration = (
                RFQTargetQIteration(
                    iteration=(
                        iteration_index
                    ),
                    generator_amplitude_vpp=(
                        generator_command
                    ),
                    measured_rfq_vpp=(
                        measured_vpp
                    ),
                    measured_q=(
                        measured_q
                    ),
                    samples_vpp=(
                        samples
                    ),
                )
            )

            iterations.append(
                iteration
            )

            if logger is not None:
                logger.log_event(
                    "rfq_target_q_iteration",
                    iteration,
                )

            relative_error = abs(
                measured_q
                - target
            ) / target

            if (
                relative_error
                <= policy.relative_q_tolerance
            ):
                result = RFQTargetQResult(
                    mass_u=(
                        matching.mass_u
                    ),
                    target_q=target,
                    setting=(
                        matching.best_setting
                    ),
                    frequency_hz=(
                        matching.best_frequency_hz
                    ),
                    generator_amplitude_vpp=(
                        generator_command
                    ),
                    measured_rfq_vpp=(
                        measured_vpp
                    ),
                    measured_q=(
                        measured_q
                    ),
                    required_rfq_vpp=(
                        required_vpp
                    ),
                    iterations=tuple(
                        iterations
                    ),
                )

                if logger is not None:
                    logger.log_event(
                        "rfq_target_q_completed",
                        result,
                    )

                return result

            if measured_q <= 0:
                raise RFQNoSignalError(
                    "No usable RFQ signal while adjusting q"
                )

            requested_scale = (
                target
                / measured_q
            )

            scale = min(
                policy.maximum_scale_up,
                max(
                    policy.maximum_scale_down,
                    requested_scale,
                ),
            )

            next_command = (
                generator_command
                * scale
            )

            if (
                next_command
                > policy.generator_max_vpp
            ):
                next_command = (
                    policy.generator_max_vpp
                )

            if math.isclose(
                next_command,
                generator_command,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RFQTargetQNotReachedError(
                    "Generator command can no longer move toward target q"
                )

            generator_command = (
                next_command
            )

        raise RFQTargetQNotReachedError(
            "Target q was not reached within the configured iterations"
        )

    except Exception:
        # Failed q adjustment always removes RF drive.
        hardware.set_generator_amplitude_vpp(
            0.0
        )
        raise