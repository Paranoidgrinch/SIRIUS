from __future__ import annotations

import math
import time

from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Iterable,
    Protocol,
)


class RFQSafetyError(
    RuntimeError
):
    pass


class RFQSafetyFault(
    RFQSafetyError
):
    pass


class RFQOffConfirmationTimeout(
    RFQSafetyFault
):
    pass


class RFQInvalidEnableReadback(
    RFQSafetyFault
):
    pass


class RFQUnsafeQError(
    RFQSafetyFault
):
    pass


class RFQFaultLatchedError(
    RFQSafetyFault
):
    pass


class RFQSafetyState(
    str,
    Enum,
):
    UNKNOWN = "unknown"

    RF_OFF_CONFIRMED = (
        "rf_off_confirmed"
    )

    CONFIGURED_OFF = (
        "configured_off"
    )

    RF_RAMPING = (
        "rf_ramping"
    )

    RF_ON_SAFE = (
        "rf_on_safe"
    )

    FAULT_LATCHED = (
        "fault_latched"
    )


class RFQSafetyHardware(
    Protocol
):
    """
    Abstract RFQ safety interface.

    These names are SIRIUS protocol methods, NOT assumed FLAVIA
    endpoints.

    A concrete hardware adapter must translate the actual machine API
    into these operations.
    """

    def request_rf_off(
        self,
    ) -> None:
        ...

    def read_rf_enabled(
        self,
    ) -> bool | None:
        ...

    def set_frequency_hz(
        self,
        frequency_hz: float,
    ) -> None:
        ...

    def set_matching_inductance_h(
        self,
        inductance_h: float,
    ) -> None:
        ...

    def set_matching_capacitance_f(
        self,
        capacitance_f: float,
    ) -> None:
        ...


@dataclass(frozen=True)
class RFQSafetyPolicy:
    q_limit: float = 0.9

    off_confirmation_timeout_s: float = 5.0

    off_poll_interval_s: float = 0.05

    off_minimum_wait_s: float = 0.0

    off_consecutive_confirmations: int = 2

    def __post_init__(
        self,
    ) -> None:
        q_limit = float(
            self.q_limit
        )

        if (
            not math.isfinite(
                q_limit
            )
            or q_limit <= 0
            or q_limit > 0.9
        ):
            raise ValueError(
                "q_limit must be finite, positive, and <= 0.9"
            )

        for name in (
            "off_confirmation_timeout_s",
            "off_poll_interval_s",
        ):
            value = float(
                getattr(
                    self,
                    name,
                )
            )

            if (
                not math.isfinite(
                    value
                )
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        wait = float(
            self.off_minimum_wait_s
        )

        if (
            not math.isfinite(
                wait
            )
            or wait < 0
        ):
            raise ValueError(
                "off_minimum_wait_s must be finite and non-negative"
            )

        if (
            self.off_consecutive_confirmations
            < 1
        ):
            raise ValueError(
                "off_consecutive_confirmations must be >= 1"
            )


@dataclass(frozen=True)
class RFQMatchingConfiguration:
    frequency_hz: float

    inductance_h: float

    capacitance_f: float

    def __post_init__(
        self,
    ) -> None:
        for name in (
            "frequency_hz",
            "inductance_h",
            "capacitance_f",
        ):
            value = float(
                getattr(
                    self,
                    name,
                )
            )

            if (
                not math.isfinite(
                    value
                )
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
                )


@dataclass(frozen=True)
class RFQOffConfirmation:
    elapsed_s: float

    confirmations: int

    observations: int


@dataclass(frozen=True)
class RFQDriveObservation:
    drive_setpoint: float

    measured_q: float


@dataclass(frozen=True)
class RFQDriveRampResult:
    observations: tuple[
        RFQDriveObservation,
        ...
    ]

    final_q: float | None


RFQDriveExecutor = Callable[
    [
        float,
    ],
    float,
]


class RFQSafetyController:
    """
    Fail-closed RFQ safety state machine.

    Important separation:

        - this class controls ordering / interlocks
        - the existing RFQ matching code chooses frequency/L/C candidates
        - the existing q controller chooses drive setpoints
        - the concrete hardware adapter performs actual device I/O

    No frequency, matching or RF-drive change may bypass the safety
    controller in real-machine operation.
    """

    def __init__(
        self,
        hardware: RFQSafetyHardware,
        policy: RFQSafetyPolicy | None = None,
        *,
        monotonic: Callable[
            [],
            float,
        ] = time.monotonic,
        sleep: Callable[
            [float],
            None,
        ] = time.sleep,
    ):
        self.hardware = (
            hardware
        )

        self.policy = (
            policy
            if policy is not None
            else RFQSafetyPolicy()
        )

        self.monotonic = (
            monotonic
        )

        self.sleep = (
            sleep
        )

        self.state = (
            RFQSafetyState.UNKNOWN
        )

        self.configuration: (
            RFQMatchingConfiguration
            | None
        ) = None

        self.last_measured_q: (
            float | None
        ) = None

        self.fault_reason: (
            str | None
        ) = None

    def _require_not_faulted(
        self,
    ) -> None:
        if (
            self.state
            == RFQSafetyState.FAULT_LATCHED
        ):
            raise RFQFaultLatchedError(
                "RFQ safety fault is latched"
                + (
                    ""
                    if self.fault_reason is None
                    else f": {self.fault_reason}"
                )
            )

    def _latch_fault(
        self,
        reason: str,
    ) -> None:
        self.state = (
            RFQSafetyState.FAULT_LATCHED
        )

        self.fault_reason = str(
            reason
        )

    def _best_effort_rf_off(
        self,
    ) -> None:
        try:
            self.hardware.request_rf_off()
        except Exception:
            # The original safety failure is more important.
            # Fault remains latched regardless.
            pass

    def _fail_safe(
        self,
        reason: str,
        *,
        exception_type=RFQSafetyFault,
    ):
        self._latch_fault(
            reason
        )

        self._best_effort_rf_off()

        raise exception_type(
            reason
        )

    def confirm_rf_off(
        self,
    ) -> RFQOffConfirmation:
        """
        Command RF OFF once and positively confirm it.

        None is not confirmation.
        True resets the consecutive-confirmation count.
        Any non-bool/non-None readback fails closed.
        """

        self.hardware.request_rf_off()

        start = float(
            self.monotonic()
        )

        if not math.isfinite(
            start
        ):
            self._fail_safe(
                "Invalid monotonic clock before RF-OFF confirmation"
            )

        confirmations = 0

        observations = 0

        while True:
            now = float(
                self.monotonic()
            )

            if (
                not math.isfinite(
                    now
                )
                or now < start
            ):
                self._fail_safe(
                    "Invalid monotonic clock during RF-OFF confirmation"
                )

            elapsed = (
                now
                - start
            )

            if (
                elapsed
                >= self.policy
                .off_confirmation_timeout_s
            ):
                self._fail_safe(
                    "RF OFF could not be positively confirmed within "
                    f"{self.policy.off_confirmation_timeout_s:g} s",
                    exception_type=(
                        RFQOffConfirmationTimeout
                    ),
                )

            enabled = (
                self.hardware
                .read_rf_enabled()
            )

            observations += 1

            if enabled is None:
                confirmations = 0

            elif isinstance(
                enabled,
                bool,
            ):
                if enabled:
                    confirmations = 0

                else:
                    if (
                        elapsed
                        >= self.policy
                        .off_minimum_wait_s
                    ):
                        confirmations += 1
                    else:
                        confirmations = 0

            else:
                self._fail_safe(
                    "Invalid RF-enable readback "
                    f"{enabled!r}; expected bool or None",
                    exception_type=(
                        RFQInvalidEnableReadback
                    ),
                )

            if (
                confirmations
                >= self.policy
                .off_consecutive_confirmations
            ):
                self.state = (
                    RFQSafetyState
                    .RF_OFF_CONFIRMED
                )

                self.last_measured_q = (
                    None
                )

                return RFQOffConfirmation(
                    elapsed_s=(
                        elapsed
                    ),
                    confirmations=(
                        confirmations
                    ),
                    observations=(
                        observations
                    ),
                )

            self.sleep(
                self.policy
                .off_poll_interval_s
            )

    def reconfigure(
        self,
        configuration: RFQMatchingConfiguration,
    ) -> None:
        """
        RF must be positively OFF before frequency/L/C changes.

        Configuration commands are never issued while RF-OFF
        confirmation is uncertain.
        """

        self._require_not_faulted()

        if not isinstance(
            configuration,
            RFQMatchingConfiguration,
        ):
            raise TypeError(
                "configuration must be RFQMatchingConfiguration"
            )

        self.confirm_rf_off()

        try:
            self.hardware.set_frequency_hz(
                float(
                    configuration.frequency_hz
                )
            )

            self.hardware.set_matching_inductance_h(
                float(
                    configuration.inductance_h
                )
            )

            self.hardware.set_matching_capacitance_f(
                float(
                    configuration.capacitance_f
                )
            )

        except Exception as exc:
            self._latch_fault(
                "RFQ reconfiguration failed"
            )

            self._best_effort_rf_off()

            raise RFQSafetyFault(
                "RFQ reconfiguration failed"
            ) from exc

        self.configuration = (
            configuration
        )

        self.state = (
            RFQSafetyState.CONFIGURED_OFF
        )

    def validate_q(
        self,
        measured_q,
    ) -> float:
        try:
            q_value = float(
                measured_q
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            self._latch_fault(
                "RFQ measured q is invalid"
            )

            self._best_effort_rf_off()

            raise RFQUnsafeQError(
                f"Invalid measured q: {measured_q!r}"
            ) from exc

        if (
            not math.isfinite(
                q_value
            )
            or q_value < 0
        ):
            self._fail_safe(
                f"Invalid measured RFQ q={q_value!r}",
                exception_type=(
                    RFQUnsafeQError
                ),
            )

        if (
            q_value
            > self.policy.q_limit
        ):
            self._fail_safe(
                "RFQ stability limit exceeded: "
                f"q={q_value:.6g} > "
                f"{self.policy.q_limit:.6g}",
                exception_type=(
                    RFQUnsafeQError
                ),
            )

        self.last_measured_q = (
            q_value
        )

        return q_value

    def run_drive_ramp(
        self,
        drive_setpoints: Iterable[
            float
        ],
        executor: RFQDriveExecutor,
    ) -> RFQDriveRampResult:
        """
        Execute a caller-defined RF-drive ramp.

        The caller/existing q controller chooses the drive setpoints.
        SIRIUS safety does NOT invent generator units or step sizes here.

        For every individual drive step:

            drive command
            -> hardware settling / Vpp measurement inside executor
            -> measured q
            -> safety validation
            -> only then next drive step

        Any exception or unsafe q immediately requests RF OFF and latches
        the safety fault.
        """

        self._require_not_faulted()

        if (
            self.state
            != RFQSafetyState.CONFIGURED_OFF
        ):
            raise RFQSafetyError(
                "RF drive ramp requires a successfully configured "
                "RFQ in confirmed-OFF state"
            )

        observations: list[
            RFQDriveObservation
        ] = []

        self.state = (
            RFQSafetyState.RF_RAMPING
        )

        try:
            for drive_setpoint in (
                drive_setpoints
            ):
                drive = float(
                    drive_setpoint
                )

                if not math.isfinite(
                    drive
                ):
                    raise ValueError(
                        "RF drive setpoint must be finite"
                    )

                measured_q = (
                    executor(
                        drive
                    )
                )

                q_value = (
                    self.validate_q(
                        measured_q
                    )
                )

                observations.append(
                    RFQDriveObservation(
                        drive_setpoint=(
                            drive
                        ),
                        measured_q=(
                            q_value
                        ),
                    )
                )

        except RFQSafetyFault:
            raise

        except Exception as exc:
            self._latch_fault(
                "RF drive ramp failed"
            )

            self._best_effort_rf_off()

            raise RFQSafetyFault(
                "RF drive ramp failed"
            ) from exc

        if observations:
            self.state = (
                RFQSafetyState.RF_ON_SAFE
            )

            final_q = (
                observations[
                    -1
                ].measured_q
            )

        else:
            # No RF command was issued.
            self.state = (
                RFQSafetyState.CONFIGURED_OFF
            )

            final_q = None

        return RFQDriveRampResult(
            observations=tuple(
                observations
            ),
            final_q=(
                final_q
            ),
        )

    def shutdown_rf(
        self,
    ) -> RFQOffConfirmation:
        """
        Explicit safe shutdown.
        """

        return self.confirm_rf_off()

    def reset_fault(
        self,
    ) -> RFQOffConfirmation:
        """
        A fault may only be cleared after RF OFF is positively confirmed.

        There is deliberately no software-only "clear fault" operation.
        """

        confirmation = (
            self.confirm_rf_off()
        )

        self.fault_reason = (
            None
        )

        self.configuration = (
            None
        )

        self.state = (
            RFQSafetyState
            .RF_OFF_CONFIRMED
        )

        return confirmation