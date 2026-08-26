from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from sirius.flavia_adapter import (
    READBACK_CHANNELS,
    STEERER_PARAMETERS,
)
from sirius.parameters import hardware_steerer_to_sirius


@dataclass(frozen=True)
class SettlingPolicy:
    """
    Rules for deciding whether a hardware readback has become stable.

    IMPORTANT:
    Stability is deliberately NOT defined as readback ~= command value.

    FLAVIA hardware can show substantial systematic differences between
    commanded and measured values. SIRIUS therefore waits for the readback
    itself to become stable.
    """

    max_readback_span: float
    relative_readback_span: float = 0.0

    timeout_s: float = 10.0
    poll_interval_s: float = 0.1
    minimum_wait_s: float = 0.2

    window_samples: int = 4

    def __post_init__(self) -> None:
        if self.max_readback_span < 0:
            raise ValueError(
                "max_readback_span must be non-negative"
            )

        if self.relative_readback_span < 0:
            raise ValueError(
                "relative_readback_span must be non-negative"
            )

        if self.timeout_s <= 0:
            raise ValueError(
                "timeout_s must be greater than zero"
            )

        if self.poll_interval_s <= 0:
            raise ValueError(
                "poll_interval_s must be greater than zero"
            )

        if self.minimum_wait_s < 0:
            raise ValueError(
                "minimum_wait_s must be non-negative"
            )

        if self.window_samples < 2:
            raise ValueError(
                "window_samples must be at least 2"
            )

    def allowed_span_for(self, mean_readback: float) -> float:
        return max(
            self.max_readback_span,
            abs(mean_readback) * self.relative_readback_span,
        )


@dataclass(frozen=True)
class SettlingResult:
    parameter: str

    command_value: float
    settled_readback: float

    readback_span: float
    allowed_span: float

    command_readback_delta: float

    elapsed_s: float
    samples: int
    window_samples: int


class SettlingTimeoutError(TimeoutError):
    def __init__(
        self,
        parameter: str,
        command_value: float,
        last_readback: float | None,
        elapsed_s: float,
        samples: int,
    ):
        self.parameter = parameter
        self.command_value = command_value
        self.last_readback = last_readback
        self.elapsed_s = elapsed_s
        self.samples = samples

        super().__init__(
            f"{parameter} did not reach a stable readback "
            f"after command {command_value} within "
            f"{elapsed_s:.3f} s; "
            f"last readback={last_readback}"
        )


def _convert_readback_to_sirius(
    parameter: str,
    raw_value: float,
) -> float:
    value = float(raw_value)

    if parameter in STEERER_PARAMETERS:
        value = hardware_steerer_to_sirius(value)

    return value


def wait_for_stable_readback(
    adapter,
    parameter: str,
    command_value: float,
    policy: SettlingPolicy,
    *,
    baseline_timestamp: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SettlingResult:
    """
    Wait until the physical readback itself becomes stable.

    The readback is NOT required to equal the commanded value.

    If FLAVIA provides timestamps, repeated polling of the same stale
    DataModel sample does not count as multiple observations.
    """

    if parameter not in READBACK_CHANNELS:
        raise KeyError(
            f"No readback channel registered for {parameter}"
        )

    channel_name = READBACK_CHANNELS[parameter]

    start = monotonic()

    window: deque[float] = deque(
        maxlen=policy.window_samples
    )

    samples = 0
    last_readback: float | None = None

    last_timestamp = baseline_timestamp

    while True:
        elapsed = monotonic() - start

        if elapsed > policy.timeout_s:
            raise SettlingTimeoutError(
                parameter=parameter,
                command_value=command_value,
                last_readback=last_readback,
                elapsed_s=elapsed,
                samples=samples,
            )

        if elapsed < policy.minimum_wait_s:
            sleep(policy.poll_interval_s)
            continue

        snapshot = adapter.read_channel(channel_name)

        if snapshot is not None and snapshot.value is not None:
            timestamp = snapshot.timestamp

            is_fresh = (
                timestamp is None
                or timestamp != last_timestamp
            )

            if is_fresh:
                if timestamp is not None:
                    last_timestamp = timestamp

                readback = _convert_readback_to_sirius(
                    parameter,
                    snapshot.value,
                )

                if math.isfinite(readback):
                    last_readback = readback
                    samples += 1
                    window.append(readback)

                    if len(window) == policy.window_samples:
                        values = list(window)

                        mean_readback = (
                            sum(values) / len(values)
                        )

                        span = max(values) - min(values)

                        allowed_span = (
                            policy.allowed_span_for(
                                mean_readback
                            )
                        )

                        if span <= allowed_span:
                            return SettlingResult(
                                parameter=parameter,
                                command_value=float(
                                    command_value
                                ),
                                settled_readback=mean_readback,
                                readback_span=span,
                                allowed_span=allowed_span,
                                command_readback_delta=(
                                    mean_readback
                                    - float(command_value)
                                ),
                                elapsed_s=elapsed,
                                samples=samples,
                                window_samples=len(values),
                            )

        sleep(policy.poll_interval_s)


def set_and_wait(
    adapter,
    parameter: str,
    command_value: float,
    policy: SettlingPolicy,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SettlingResult:
    """
    Send a command through FLAVIA and wait until its readback is stable.
    """

    channel_name = READBACK_CHANNELS.get(parameter)

    baseline_timestamp = None

    if channel_name is not None:
        baseline = adapter.read_channel(channel_name)

        if baseline is not None:
            baseline_timestamp = baseline.timestamp

    adapter.set_parameter(
        parameter,
        command_value,
    )

    return wait_for_stable_readback(
        adapter,
        parameter,
        command_value,
        policy,
        baseline_timestamp=baseline_timestamp,
        monotonic=monotonic,
        sleep=sleep,
    )