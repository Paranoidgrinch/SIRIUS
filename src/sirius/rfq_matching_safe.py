from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

# Re-export the existing RFQ matching API. Application code can therefore
# switch import modules without changing its other imports.
from sirius.rfq_matching import *  # noqa: F401,F403

from sirius.rfq_matching import (
    RFQHardware,
    RFQMatchingError,
    RFQMatchingPolicy,
    RFQMatchingResult,
    RFQTargetQPolicy,
    RFQTargetQResult,
    RFQUnsafeAmplitudeError,
    rfq_vpp_for_q,
    search_rfq_resonance as _search_rfq_resonance,
    set_target_q as _set_target_q,
)

from sirius.rfq_safety import (
    RFQFaultLatchedError,
    RFQSafetyController,
    RFQSafetyPolicy,
    RFQSafetyState,
    RFQUnsafeQError,
)


class RFQSafetyCapabilityError(
    RFQMatchingError
):
    pass


class _ControllerHardwareBridge:
    """
    Bridge the existing RFQHardware contract to RFQSafetyController.

    No FLAVIA endpoint is invented here.

    RF OFF is the already existing generator-amplitude-zero operation.
    Positive OFF acknowledgement is a separate required hardware
    capability: read_rf_enabled() -> bool | None.
    """

    def __init__(
        self,
        hardware,
    ):
        self.hardware = hardware

        self._pending_inductance_uh: (
            float | None
        ) = None

    def request_rf_off(
        self,
    ) -> None:
        self.hardware.set_generator_amplitude_vpp(
            0.0
        )

    def read_rf_enabled(
        self,
    ) -> bool | None:
        reader = getattr(
            self.hardware,
            "read_rf_enabled",
            None,
        )

        if not callable(
            reader
        ):
            raise RFQSafetyCapabilityError(
                "RFQ hardware does not expose a positive RF-enable "
                "readback; safe matching cannot proceed"
            )

        return reader()

    # These three methods satisfy the complete RFQSafetyHardware
    # protocol as well. The safety matching wrapper below normally uses
    # execute_configuration_change(), because the existing RFQ API
    # already has a combined set_matching() operation.

    def set_frequency_hz(
        self,
        frequency_hz: float,
    ) -> None:
        self.hardware.set_frequency_hz(
            float(
                frequency_hz
            )
        )

    def set_matching_inductance_h(
        self,
        inductance_h: float,
    ) -> None:
        self._pending_inductance_uh = (
            float(
                inductance_h
            )
            * 1e6
        )

    def set_matching_capacitance_f(
        self,
        capacitance_f: float,
    ) -> None:
        if (
            self._pending_inductance_uh
            is None
        ):
            raise RFQSafetyCapabilityError(
                "Matching capacitance received before inductance"
            )

        self.hardware.set_matching(
            self._pending_inductance_uh,
            float(
                capacitance_f
            )
            * 1e12,
        )


class _SafetyInterlockedRFQHardware:
    """
    Safety-preserving proxy for the existing RFQ matching algorithms.

    The original algorithms remain responsible for:
      - selecting L/C candidates,
      - frequency grids,
      - generator amplitude progression,
      - scope averaging,
      - target-q convergence.

    This proxy is responsible only for:
      - RF-OFF confirmation before every frequency/matching change,
      - fault latching,
      - q validation from every measured RFQ-Vpp sample.
    """

    def __init__(
        self,
        hardware: RFQHardware,
        *,
        mass_u: float,
        q_abort_limit: float,
        safety_policy: RFQSafetyPolicy | None = None,
        sleeper: Callable[
            [float],
            None,
        ],
    ):
        reader = getattr(
            hardware,
            "read_rf_enabled",
            None,
        )

        if not callable(
            reader
        ):
            raise RFQSafetyCapabilityError(
                "Safe RFQ operation requires hardware.read_rf_enabled()"
            )

        self._hardware = (
            hardware
        )

        self._mass_u = float(
            mass_u
        )

        base_policy = (
            safety_policy
            if safety_policy is not None
            else RFQSafetyPolicy()
        )

        effective_q_limit = min(
            float(
                base_policy.q_limit
            ),
            float(
                q_abort_limit
            ),
        )

        effective_policy = replace(
            base_policy,
            q_limit=(
                effective_q_limit
            ),
        )

        self._bridge = (
            _ControllerHardwareBridge(
                hardware
            )
        )

        self.safety_controller = (
            RFQSafetyController(
                self._bridge,
                effective_policy,
                sleep=sleeper,
            )
        )

        self._frequency_hz: (
            float | None
        ) = None

        self._rf_drive_active = (
            False
        )

        self._pending_drive_validation = (
            False
        )

    def set_frequency_hz(
        self,
        value: float,
    ) -> None:
        frequency = float(
            value
        )

        def change() -> None:
            self._hardware.set_frequency_hz(
                frequency
            )

        self.safety_controller.execute_configuration_change(
            change
        )

        self._frequency_hz = (
            frequency
        )

        self._rf_drive_active = (
            False
        )

        self._pending_drive_validation = (
            False
        )

    def set_matching(
        self,
        inductance_uh: float,
        capacitance_pf: float,
    ) -> None:
        inductance = float(
            inductance_uh
        )

        capacitance = float(
            capacitance_pf
        )

        def change() -> None:
            self._hardware.set_matching(
                inductance,
                capacitance,
            )

        self.safety_controller.execute_configuration_change(
            change
        )

        self._rf_drive_active = (
            False
        )

        self._pending_drive_validation = (
            False
        )

    def set_generator_amplitude_vpp(
        self,
        value: float,
    ) -> None:
        command = float(
            value
        )

        if command == 0.0:
            self.safety_controller.confirm_rf_off()

            self._rf_drive_active = (
                False
            )

            self._pending_drive_validation = (
                False
            )

            return

        self.safety_controller.begin_drive_step()

        try:
            self._hardware.set_generator_amplitude_vpp(
                command
            )

        except Exception:
            self.safety_controller.latch_external_fault(
                "RF generator amplitude command failed"
            )

            raise

        self._rf_drive_active = (
            True
        )

        self._pending_drive_validation = (
            True
        )

    def read_rfq_vpp(
        self,
    ) -> float:
        try:
            measured_vpp = float(
                self._hardware.read_rfq_vpp()
            )

        except Exception:
            self.safety_controller.latch_external_fault(
                "RFQ Vpp readback failed"
            )

            raise

        if not self._rf_drive_active:
            return measured_vpp

        if self._frequency_hz is None:
            self.safety_controller.latch_external_fault(
                "RF drive active without a known RFQ frequency"
            )

            raise RFQSafetyCapabilityError(
                "RFQ frequency is unknown while RF drive is active"
            )

        # q is linear in RFQ Vpp at fixed mass/frequency.
        vpp_for_q_one = rfq_vpp_for_q(
            self._mass_u,
            self._frequency_hz,
            1.0,
            enforce_operational_limit=False,
        )

        measured_q = (
            measured_vpp
            / vpp_for_q_one
        )

        try:
            self.safety_controller.validate_q(
                measured_q
            )

        except RFQUnsafeQError as exc:
            # Preserve the public exception family already used by
            # rfq_matching.py while retaining the latched safety fault.
            raise RFQUnsafeAmplitudeError(
                str(
                    exc
                )
            ) from exc

        self._pending_drive_validation = (
            False
        )

        return measured_vpp

    def ensure_pending_drive_is_validated(
        self,
        *,
        measurements: int,
        interval_s: float,
        sleeper: Callable[
            [float],
            None,
        ],
    ) -> None:
        """
        Primarily handles RFQMatchingPolicy.leave_probe_on=True.

        The original search normally measures immediately after every
        positive drive command. The final leave-probe-on command is the
        one exception; validate it before returning to the caller.
        """

        if not self._pending_drive_validation:
            return

        count = max(
            1,
            int(
                measurements
            ),
        )

        for index in range(
            count
        ):
            self.read_rfq_vpp()

            if (
                interval_s > 0
                and index
                < count - 1
            ):
                sleeper(
                    interval_s
                )


def search_rfq_resonance(
    hardware: RFQHardware,
    mass_u: float,
    lc_candidates: Iterable,
    policy: RFQMatchingPolicy,
    *,
    logger=None,
    sleeper: Callable[
        [float],
        None,
    ] = __import__(
        "time"
    ).sleep,
    safety_policy: RFQSafetyPolicy | None = None,
) -> RFQMatchingResult:
    safe_hardware = (
        _SafetyInterlockedRFQHardware(
            hardware,
            mass_u=mass_u,
            q_abort_limit=(
                policy.q_abort_limit
            ),
            safety_policy=(
                safety_policy
            ),
            sleeper=sleeper,
        )
    )

    result = _search_rfq_resonance(
        safe_hardware,
        mass_u,
        lc_candidates,
        policy,
        logger=logger,
        sleeper=sleeper,
    )

    safe_hardware.ensure_pending_drive_is_validated(
        measurements=(
            policy.measurements_per_point
        ),
        interval_s=(
            policy.measurement_interval_s
        ),
        sleeper=sleeper,
    )

    return result


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
    ] = __import__(
        "time"
    ).sleep,
    safety_policy: RFQSafetyPolicy | None = None,
) -> RFQTargetQResult:
    safe_hardware = (
        _SafetyInterlockedRFQHardware(
            hardware,
            mass_u=(
                matching.mass_u
            ),
            q_abort_limit=(
                policy.q_abort_limit
            ),
            safety_policy=(
                safety_policy
            ),
            sleeper=sleeper,
        )
    )

    result = _set_target_q(
        safe_hardware,
        matching,
        target_q,
        policy,
        logger=logger,
        sleeper=sleeper,
    )

    safe_hardware.ensure_pending_drive_is_validated(
        measurements=(
            policy.measurements_per_iteration
        ),
        interval_s=(
            policy.measurement_interval_s
        ),
        sleeper=sleeper,
    )

    return result