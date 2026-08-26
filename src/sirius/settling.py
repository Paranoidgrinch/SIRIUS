from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SettlingPolicy:
    """
    Rules for deciding whether a hardware parameter has reached its target.

    A parameter is considered settled only after several consecutive
    readbacks are within the allowed target tolerance.
    """

    absolute_tolerance: float
    relative_tolerance: float = 0.0
    timeout_s: float = 10.0
    poll_interval_s: float = 0.1
    consecutive_samples: int = 3

    def __post_init__(self) -> None:
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be non-negative")

        if self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative")

        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")

        if self.poll_interval_s < 0:
            raise ValueError("poll_interval_s must be non-negative")

        if self.consecutive_samples < 1:
            raise ValueError("consecutive_samples must be at least 1")

    def tolerance_for(self, target: float) -> float:
        return max(
            self.absolute_tolerance,
            abs(target) * self.relative_tolerance,
        )


@dataclass(frozen=True)
class SettlingResult:
    parameter: str
    target: float
    final_readback: float
    tolerance: float
    elapsed_s: float
    samples: int
    consecutive_samples: int

    @property
    def final_error(self) -> float:
        return self.final_readback - self.target

    @property
    def absolute_final_error(self) -> float:
        return abs(self.final_error)


class SettlingTimeoutError(TimeoutError):
    def __init__(
        self,
        parameter: str,
        target: float,
        last_readback: float | None,
        elapsed_s: float,
        samples: int,
    ):
        self.parameter = parameter
        self.target = target
        self.last_readback = last_readback
        self.elapsed_s = elapsed_s
        self.samples = samples

        super().__init__(
            f"{parameter} did not settle at {target} "
            f"within {elapsed_s:.3f} s; "
            f"last readback={last_readback}"
        )


def is_within_tolerance(
    value: float,
    target: float,
    tolerance: float,
) -> bool:
    if not math.isfinite(value):
        return False

    return abs(value - target) <= tolerance


def wait_for_parameter(
    adapter,
    parameter: str,
    target: float,
    policy: SettlingPolicy,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SettlingResult:
    """
    Wait until a FLAVIA parameter readback is stably at its target.

    A single in-tolerance sample is not sufficient. The parameter must
    remain within tolerance for policy.consecutive_samples readbacks.
    """

    target = float(target)
    tolerance = policy.tolerance_for(target)

    start = monotonic()

    samples = 0
    consecutive = 0
    last_readback: float | None = None

    while True:
        now = monotonic()
        elapsed = now - start

        if elapsed > policy.timeout_s:
            raise SettlingTimeoutError(
                parameter=parameter,
                target=target,
                last_readback=last_readback,
                elapsed_s=elapsed,
                samples=samples,
            )

        readback = adapter.read_parameter(parameter)

        if readback is not None:
            readback = float(readback)
            last_readback = readback
            samples += 1

            if is_within_tolerance(
                readback,
                target,
                tolerance,
            ):
                consecutive += 1

                if consecutive >= policy.consecutive_samples:
                    return SettlingResult(
                        parameter=parameter,
                        target=target,
                        final_readback=readback,
                        tolerance=tolerance,
                        elapsed_s=elapsed,
                        samples=samples,
                        consecutive_samples=consecutive,
                    )

            else:
                consecutive = 0

        sleep(policy.poll_interval_s)


def set_and_wait(
    adapter,
    parameter: str,
    target: float,
    policy: SettlingPolicy,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> SettlingResult:
    """
    Set a SIRIUS parameter through FLAVIA and wait for stable readback.
    """

    adapter.set_parameter(
        parameter,
        target,
    )

    return wait_for_parameter(
        adapter,
        parameter,
        target,
        policy,
        monotonic=monotonic,
        sleep=sleep,
    )