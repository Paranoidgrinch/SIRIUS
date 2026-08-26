from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable


KEITHLEY_CURRENT_CHANNEL = "keithley/current_A"


@dataclass(frozen=True)
class MeasurementPolicy:
    """
    Adaptive beam-current measurement policy.

    The numerical defaults are initial software defaults only.
    They are not considered final hardware-tuned values.
    """

    min_samples: int = 6
    max_samples: int = 100

    min_duration_s: float = 0.3
    max_duration_s: float = 8.0
    poll_interval_s: float = 0.02

    relative_sem_target: float = 0.01
    absolute_sem_target_a: float = 1e-13

    drift_window_samples: int = 6
    drift_tolerance_factor: float = 2.0

    below_noise_sigma: float = 3.0

    def __post_init__(self) -> None:
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")

        if self.max_samples < self.min_samples:
            raise ValueError(
                "max_samples must be >= min_samples"
            )

        if self.min_duration_s < 0:
            raise ValueError(
                "min_duration_s must be non-negative"
            )

        if self.max_duration_s <= 0:
            raise ValueError(
                "max_duration_s must be greater than zero"
            )

        if self.max_duration_s < self.min_duration_s:
            raise ValueError(
                "max_duration_s must be >= min_duration_s"
            )

        if self.poll_interval_s <= 0:
            raise ValueError(
                "poll_interval_s must be greater than zero"
            )

        if self.relative_sem_target < 0:
            raise ValueError(
                "relative_sem_target must be non-negative"
            )

        if self.absolute_sem_target_a < 0:
            raise ValueError(
                "absolute_sem_target_a must be non-negative"
            )

        if self.drift_window_samples < 4:
            raise ValueError(
                "drift_window_samples must be at least 4"
            )

        if self.drift_tolerance_factor < 0:
            raise ValueError(
                "drift_tolerance_factor must be non-negative"
            )

        if self.below_noise_sigma < 0:
            raise ValueError(
                "below_noise_sigma must be non-negative"
            )


@dataclass(frozen=True)
class BeamCurrentSample:
    current_a: float
    source_timestamp: float | None
    elapsed_s: float


@dataclass(frozen=True)
class BeamMeasurement:
    mean_a: float
    sigma_a: float
    sem_a: float

    n: int
    duration_s: float

    relative_sem: float | None
    precision_threshold_a: float

    drift_delta_a: float | None

    stop_reason: str
    below_noise_floor: bool

    samples: tuple[BeamCurrentSample, ...]


class BeamMeasurementError(RuntimeError):
    pass


class BeamMeasurementNoDataError(BeamMeasurementError):
    pass


def _statistics(
    samples: list[BeamCurrentSample],
) -> tuple[float, float, float]:
    values = [
        sample.current_a
        for sample in samples
    ]

    mean_a = statistics.fmean(values)

    if len(values) >= 2:
        sigma_a = statistics.stdev(values)
    else:
        sigma_a = 0.0

    if len(values) > 0:
        sem_a = sigma_a / math.sqrt(len(values))
    else:
        sem_a = math.inf

    return mean_a, sigma_a, sem_a


def _precision_threshold(
    mean_a: float,
    policy: MeasurementPolicy,
) -> float:
    return max(
        policy.absolute_sem_target_a,
        abs(mean_a) * policy.relative_sem_target,
    )


def _relative_sem(
    mean_a: float,
    sem_a: float,
) -> float | None:
    if mean_a == 0:
        return None

    return sem_a / abs(mean_a)


def _recent_drift_delta(
    samples: list[BeamCurrentSample],
    window_samples: int,
) -> float | None:
    """
    Compare the first and second halves of the recent measurement window.

    This is deliberately simple and robust. It prevents SIRIUS from
    accepting a low SEM while the measured current is still systematically
    moving after a beamline change.
    """

    if len(samples) < window_samples:
        return None

    recent = samples[-window_samples:]

    split = len(recent) // 2

    first_half = [
        sample.current_a
        for sample in recent[:split]
    ]

    second_half = [
        sample.current_a
        for sample in recent[split:]
    ]

    first_mean = statistics.fmean(first_half)
    second_mean = statistics.fmean(second_half)

    return abs(second_mean - first_mean)


def _build_result(
    samples: list[BeamCurrentSample],
    duration_s: float,
    policy: MeasurementPolicy,
    stop_reason: str,
    below_noise_floor: bool,
) -> BeamMeasurement:
    mean_a, sigma_a, sem_a = _statistics(samples)

    threshold = _precision_threshold(
        mean_a,
        policy,
    )

    drift_delta = _recent_drift_delta(
        samples,
        policy.drift_window_samples,
    )

    return BeamMeasurement(
        mean_a=mean_a,
        sigma_a=sigma_a,
        sem_a=sem_a,
        n=len(samples),
        duration_s=duration_s,
        relative_sem=_relative_sem(
            mean_a,
            sem_a,
        ),
        precision_threshold_a=threshold,
        drift_delta_a=drift_delta,
        stop_reason=stop_reason,
        below_noise_floor=below_noise_floor,
        samples=tuple(samples),
    )


def measure_beam_current(
    adapter,
    policy: MeasurementPolicy,
    *,
    noise_floor_a: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> BeamMeasurement:
    """
    Adaptively measure the beam-current magnitude from FLAVIA.

    Only fresh DataModel samples are accepted when source timestamps are
    available.

    Measurement stops when one of the following occurs:

    - requested precision is reached and the recent signal is stable
    - signal is conservatively below a supplied noise floor
    - maximum number of fresh samples is reached
    - maximum measurement duration is reached

    If no fresh current sample is received before max_duration_s,
    BeamMeasurementNoDataError is raised.
    """

    if noise_floor_a is not None and noise_floor_a < 0:
        raise ValueError(
            "noise_floor_a must be non-negative"
        )

    start = monotonic()

    samples: list[BeamCurrentSample] = []

    last_source_timestamp: float | None = None

    while True:
        now = monotonic()
        elapsed = now - start

        if elapsed >= policy.max_duration_s:
            if not samples:
                raise BeamMeasurementNoDataError(
                    "No fresh Keithley current sample received"
                )

            return _build_result(
                samples=samples,
                duration_s=elapsed,
                policy=policy,
                stop_reason="max_duration",
                below_noise_floor=False,
            )

        snapshot = adapter.read_channel(
            KEITHLEY_CURRENT_CHANNEL
        )

        if snapshot is not None and snapshot.value is not None:
            source_timestamp = snapshot.timestamp

            is_fresh = (
                source_timestamp is None
                or source_timestamp != last_source_timestamp
            )

            if is_fresh:
                if source_timestamp is not None:
                    last_source_timestamp = source_timestamp

                current_a = abs(float(snapshot.value))

                if math.isfinite(current_a):
                    samples.append(
                        BeamCurrentSample(
                            current_a=current_a,
                            source_timestamp=source_timestamp,
                            elapsed_s=elapsed,
                        )
                    )

                    if len(samples) >= policy.max_samples:
                        return _build_result(
                            samples=samples,
                            duration_s=elapsed,
                            policy=policy,
                            stop_reason="max_samples",
                            below_noise_floor=False,
                        )

                    enough_data = (
                        len(samples) >= policy.min_samples
                        and elapsed >= policy.min_duration_s
                    )

                    if enough_data:
                        mean_a, _, sem_a = _statistics(
                            samples
                        )

                        threshold = _precision_threshold(
                            mean_a,
                            policy,
                        )

                        drift_delta = _recent_drift_delta(
                            samples,
                            policy.drift_window_samples,
                        )

                        drift_is_stable = (
                            drift_delta is not None
                            and drift_delta
                            <= (
                                threshold
                                * policy.drift_tolerance_factor
                            )
                        )

                        if noise_floor_a is not None:
                            upper_bound = (
                                mean_a
                                + policy.below_noise_sigma
                                * sem_a
                            )

                            if upper_bound <= noise_floor_a:
                                return _build_result(
                                    samples=samples,
                                    duration_s=elapsed,
                                    policy=policy,
                                    stop_reason="below_noise_floor",
                                    below_noise_floor=True,
                                )

                        if (
                            sem_a <= threshold
                            and drift_is_stable
                        ):
                            return _build_result(
                                samples=samples,
                                duration_s=elapsed,
                                policy=policy,
                                stop_reason="precision_reached",
                                below_noise_floor=False,
                            )

        sleep(policy.poll_interval_s)