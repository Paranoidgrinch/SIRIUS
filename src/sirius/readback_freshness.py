from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable


class ReadbackFreshnessError(
    RuntimeError
):
    pass


class ReadbackTimestampMissingError(
    ReadbackFreshnessError
):
    pass


class ReadbackTimestampInvalidError(
    ReadbackFreshnessError
):
    pass


class ReadbackValueInvalidError(
    ReadbackFreshnessError
):
    pass


class ReadbackFreshnessTimeoutError(
    ReadbackFreshnessError
):
    pass


@dataclass(frozen=True)
class ReadbackFreshnessPolicy:
    """
    Post-command readback freshness handshake.

    timeout_s:
        Maximum time to wait for a timestamped readback newer than the
        pre-command freshness barrier.

    poll_interval_s:
        Poll interval for the FLAVIA DataModel.

    This policy proves temporal freshness only. Numerical settling remains
    a separate requirement.
    """

    timeout_s: float = 5.0

    poll_interval_s: float = 0.02

    def __post_init__(self) -> None:
        timeout = float(
            self.timeout_s
        )

        interval = float(
            self.poll_interval_s
        )

        if (
            not math.isfinite(
                timeout
            )
            or timeout <= 0
        ):
            raise ValueError(
                "timeout_s must be finite and greater than zero"
            )

        if (
            not math.isfinite(
                interval
            )
            or interval <= 0
        ):
            raise ValueError(
                "poll_interval_s must be finite and greater than zero"
            )


@dataclass(frozen=True)
class FreshReadbackObservation:
    parameter_name: str

    value: float

    source_timestamp: float

    elapsed_s: float

    stale_observations: int


def _finite_timestamp(
    value,
    *,
    parameter_name: str,
) -> float:
    if value is None:
        raise ReadbackTimestampMissingError(
            f"{parameter_name}: readback has no source timestamp"
        )

    try:
        timestamp = float(
            value
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ReadbackTimestampInvalidError(
            f"{parameter_name}: invalid readback timestamp {value!r}"
        ) from exc

    if not math.isfinite(
        timestamp
    ):
        raise ReadbackTimestampInvalidError(
            f"{parameter_name}: non-finite readback timestamp "
            f"{timestamp!r}"
        )

    return timestamp


def _finite_value(
    value,
    *,
    parameter_name: str,
) -> float:
    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ReadbackValueInvalidError(
            f"{parameter_name}: invalid readback value {value!r}"
        ) from exc

    if not math.isfinite(
        numeric
    ):
        raise ReadbackValueInvalidError(
            f"{parameter_name}: non-finite readback value {numeric!r}"
        )

    return numeric


def wait_for_fresh_parameter_readback(
    adapter,
    parameter_name: str,
    *,
    not_before_source_timestamp: float | None,
    policy: ReadbackFreshnessPolicy,
    monotonic: Callable[
        [],
        float,
    ] = time.monotonic,
    sleep: Callable[
        [float],
        None,
    ] = time.sleep,
) -> FreshReadbackObservation:
    """
    Wait until FLAVIA exposes a timestamped readback produced after the
    pre-command freshness barrier.

    The accepted condition is strictly:

        timestamp > not_before_source_timestamp

    Equality is stale.

    A timestamp that moves backwards is therefore also stale and can never
    acknowledge the command.

    No command/readback numerical equality is required.
    """

    if not isinstance(
        policy,
        ReadbackFreshnessPolicy,
    ):
        raise TypeError(
            "policy must be ReadbackFreshnessPolicy"
        )

    if (
        not_before_source_timestamp
        is not None
    ):
        barrier = _finite_timestamp(
            not_before_source_timestamp,
            parameter_name=(
                parameter_name
            ),
        )
    else:
        barrier = None

    reader = getattr(
        adapter,
        "read_parameter_snapshot",
        None,
    )

    if not callable(
        reader
    ):
        raise ReadbackFreshnessError(
            "Adapter does not expose read_parameter_snapshot(); "
            "post-command freshness cannot be proven"
        )

    start = monotonic()

    stale_observations = 0

    while True:
        now = monotonic()

        elapsed = (
            now
            - start
        )

        if (
            elapsed
            >= policy.timeout_s
        ):
            raise ReadbackFreshnessTimeoutError(
                f"{parameter_name}: no fresh post-command readback "
                f"received within {policy.timeout_s:g} s"
            )

        snapshot = reader(
            parameter_name
        )

        if (
            snapshot is None
            or snapshot.value is None
        ):
            sleep(
                policy.poll_interval_s
            )

            continue

        timestamp = _finite_timestamp(
            snapshot.timestamp,
            parameter_name=(
                parameter_name
            ),
        )

        if (
            barrier is not None
            and timestamp <= barrier
        ):
            stale_observations += 1

            sleep(
                policy.poll_interval_s
            )

            continue

        value = _finite_value(
            snapshot.value,
            parameter_name=(
                parameter_name
            ),
        )

        return FreshReadbackObservation(
            parameter_name=(
                parameter_name
            ),
            value=value,
            source_timestamp=(
                timestamp
            ),
            elapsed_s=(
                elapsed
            ),
            stale_observations=(
                stale_observations
            ),
        )