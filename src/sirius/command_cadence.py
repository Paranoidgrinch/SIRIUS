from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable


class CommandCadenceError(
    RuntimeError
):
    pass


class CommandCadenceClockError(
    CommandCadenceError
):
    pass


@dataclass(frozen=True)
class CommandCadenceObservation:
    parameter_name: str

    minimum_interval_s: float

    previous_command_monotonic_s: float | None

    reserved_command_monotonic_s: float

    waited_s: float


@dataclass
class CommandCadenceController:
    """
    Stateful per-parameter command pacing.

    A reservation is made immediately before a physical command is issued.

    The timestamp persists across separate safe_transition.apply_state()
    calls as long as the same controller instance remains attached to the
    adapter.

    If a command subsequently fails, retaining the reservation is
    deliberately conservative: the next command is delayed rather than
    being allowed too early.
    """

    monotonic: Callable[
        [],
        float,
    ] = time.monotonic

    sleep: Callable[
        [float],
        None,
    ] = time.sleep

    _last_command_by_parameter: dict[
        str,
        float,
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def reserve(
        self,
        parameter_name: str,
        minimum_interval_s: float,
    ) -> CommandCadenceObservation:
        minimum_interval_s = float(
            minimum_interval_s
        )

        if (
            not math.isfinite(
                minimum_interval_s
            )
            or minimum_interval_s < 0
        ):
            raise ValueError(
                "minimum_interval_s must be finite and non-negative"
            )

        previous = (
            self._last_command_by_parameter.get(
                parameter_name
            )
        )

        now = float(
            self.monotonic()
        )

        if not math.isfinite(
            now
        ):
            raise CommandCadenceClockError(
                "Monotonic clock returned a non-finite value"
            )

        if (
            previous is not None
            and now < previous
        ):
            raise CommandCadenceClockError(
                "Monotonic clock moved backwards"
            )

        waited = 0.0

        if previous is not None:
            elapsed = (
                now
                - previous
            )

            remaining = (
                minimum_interval_s
                - elapsed
            )

            if remaining > 0:
                self.sleep(
                    remaining
                )

                waited = float(
                    remaining
                )

                now = float(
                    self.monotonic()
                )

                if (
                    not math.isfinite(
                        now
                    )
                    or now < previous
                ):
                    raise CommandCadenceClockError(
                        "Invalid monotonic clock after command-cadence wait"
                    )

                # Protect against clocks/test doubles whose sleep function
                # returned early.
                actual_interval = (
                    now
                    - previous
                )

                tolerance = 1e-9

                if (
                    actual_interval
                    + tolerance
                    < minimum_interval_s
                ):
                    raise CommandCadenceClockError(
                        "Command cadence wait completed before the "
                        "required minimum interval elapsed"
                    )

        self._last_command_by_parameter[
            parameter_name
        ] = now

        return CommandCadenceObservation(
            parameter_name=(
                parameter_name
            ),
            minimum_interval_s=(
                minimum_interval_s
            ),
            previous_command_monotonic_s=(
                previous
            ),
            reserved_command_monotonic_s=(
                now
            ),
            waited_s=(
                waited
            ),
        )

    def last_command_time(
        self,
        parameter_name: str,
    ) -> float | None:
        return (
            self._last_command_by_parameter.get(
                parameter_name
            )
        )