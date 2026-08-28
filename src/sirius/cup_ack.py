from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable


class CupSelectionError(RuntimeError):
    pass


class CupSelectionTimeoutError(
    CupSelectionError
):
    pass


class InvalidCupReadbackError(
    CupSelectionError
):
    pass


@dataclass(frozen=True)
class CupSelectionPolicy:
    """
    Positive acknowledgement policy for Faraday-cup selection.

    A cup is accepted only after the backend reports the requested cup
    for multiple consecutive polls.

    This deliberately distinguishes:

        command sent
        !=
        cup mechanically/electronically confirmed
    """

    timeout_s: float = 10.0

    poll_interval_s: float = 0.1

    minimum_wait_s: float = 0.2

    consecutive_confirmations: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            (
                "timeout_s",
                self.timeout_s,
            ),
            (
                "poll_interval_s",
                self.poll_interval_s,
            ),
            (
                "minimum_wait_s",
                self.minimum_wait_s,
            ),
        ):
            value = float(
                value
            )

            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative"
                )

        if self.timeout_s <= 0:
            raise ValueError(
                "timeout_s must be greater than zero"
            )

        if self.poll_interval_s <= 0:
            raise ValueError(
                "poll_interval_s must be greater than zero"
            )

        if self.minimum_wait_s > self.timeout_s:
            raise ValueError(
                "minimum_wait_s must not exceed timeout_s"
            )

        if self.consecutive_confirmations < 1:
            raise ValueError(
                "consecutive_confirmations must be at least 1"
            )


@dataclass(frozen=True)
class CupReadbackSample:
    elapsed_s: float
    selected_cup: int | None

    matched_target: bool


@dataclass(frozen=True)
class CupSelectionResult:
    requested_cup: int

    confirmed_cup: int

    elapsed_s: float

    confirmation_count: int

    samples: tuple[
        CupReadbackSample,
        ...
    ]

    @property
    def sample_count(
        self,
    ) -> int:
        return len(
            self.samples
        )


def _validate_cup(
    cup: int,
) -> int:
    if isinstance(
        cup,
        bool,
    ):
        raise ValueError(
            "Cup must be an integer between 1 and 6"
        )

    try:
        normalized = int(
            cup
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Cup must be an integer between 1 and 6"
        ) from exc

    if normalized != cup:
        raise ValueError(
            "Cup must be an integer between 1 and 6"
        )

    if not 1 <= normalized <= 6:
        raise ValueError(
            "Cup must be between 1 and 6"
        )

    return normalized


def _normalize_readback(
    value,
) -> int | None:
    """
    Normalize one backend cup readback.

    None is allowed and interpreted as "no valid acknowledgement yet".

    Invalid non-None values are considered a backend/data error rather
    than silently treated as another cup.
    """

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        raise InvalidCupReadbackError(
            f"Invalid cup readback: {value!r}"
        )

    try:
        normalized = int(
            value
        )
    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise InvalidCupReadbackError(
            f"Invalid cup readback: {value!r}"
        ) from exc

    try:
        numeric = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidCupReadbackError(
            f"Invalid cup readback: {value!r}"
        ) from exc

    if not math.isfinite(
        numeric
    ):
        raise InvalidCupReadbackError(
            f"Invalid cup readback: {value!r}"
        )

    if not math.isclose(
        numeric,
        float(
            normalized
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise InvalidCupReadbackError(
            f"Non-integer cup readback: {value!r}"
        )

    if not 1 <= normalized <= 6:
        raise InvalidCupReadbackError(
            f"Cup readback outside 1..6: {normalized}"
        )

    return normalized


def select_cup_and_wait(
    *,
    select_cup: Callable[
        [int],
        None,
    ],
    read_selected_cup: Callable[
        [],
        int | None,
    ],
    target_cup: int,
    policy: CupSelectionPolicy | None = None,
    monotonic: Callable[
        [],
        float,
    ] = time.monotonic,
    sleeper: Callable[
        [float],
        None,
    ] = time.sleep,
) -> CupSelectionResult:
    """
    Send one cup-selection command and wait for positive acknowledgement.

    The command is sent exactly once.

    A successful result requires:

      1. minimum_wait_s has elapsed, and
      2. target cup was reported for N consecutive polls.

    A different cup resets the consecutive-match counter.

    None means "not confirmed yet" and also resets the counter.
    """

    target = _validate_cup(
        target_cup
    )

    active_policy = (
        policy
        if policy is not None
        else CupSelectionPolicy()
    )

    start = monotonic()

    select_cup(
        target
    )

    samples: list[
        CupReadbackSample
    ] = []

    consecutive = 0

    while True:
        now = monotonic()

        elapsed = max(
            0.0,
            float(
                now
                - start
            ),
        )

        if elapsed > active_policy.timeout_s:
            raise CupSelectionTimeoutError(
                "Timed out waiting for Cup "
                f"{target} acknowledgement after "
                f"{elapsed:.3f} s; "
                f"last readback="
                f"{samples[-1].selected_cup if samples else None}"
            )

        raw = read_selected_cup()

        selected = _normalize_readback(
            raw
        )

        matched = (
            selected
            == target
        )

        samples.append(
            CupReadbackSample(
                elapsed_s=elapsed,
                selected_cup=(
                    selected
                ),
                matched_target=(
                    matched
                ),
            )
        )

        if matched:
            consecutive += 1
        else:
            consecutive = 0

        if (
            elapsed
            >= active_policy.minimum_wait_s
            and consecutive
            >= active_policy.consecutive_confirmations
        ):
            return CupSelectionResult(
                requested_cup=target,
                confirmed_cup=target,
                elapsed_s=elapsed,
                confirmation_count=(
                    consecutive
                ),
                samples=tuple(
                    samples
                ),
            )

        sleeper(
            active_policy.poll_interval_s
        )