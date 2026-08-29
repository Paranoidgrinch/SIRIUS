import pytest

from sirius.rfq_safety import (
    RFQFaultLatchedError,
    RFQInvalidEnableReadback,
    RFQMatchingConfiguration,
    RFQOffConfirmationTimeout,
    RFQSafetyController,
    RFQSafetyPolicy,
    RFQSafetyState,
    RFQUnsafeQError,
)


class Clock:
    def __init__(
        self,
    ):
        self.now = 0.0

    def monotonic(
        self,
    ):
        return self.now

    def sleep(
        self,
        seconds,
    ):
        self.now += float(
            seconds
        )


class Hardware:
    def __init__(
        self,
    ):
        self.events = []

        self.enabled_sequence = [
            False,
            False,
        ]

        self.read_index = 0

    def request_rf_off(
        self,
    ):
        self.events.append(
            "rf_off"
        )

    def read_rf_enabled(
        self,
    ):
        index = min(
            self.read_index,
            len(
                self.enabled_sequence
            ) - 1,
        )

        self.read_index += 1

        return self.enabled_sequence[
            index
        ]

    def set_frequency_hz(
        self,
        value,
    ):
        self.events.append(
            (
                "frequency",
                value,
            )
        )

    def set_matching_inductance_h(
        self,
        value,
    ):
        self.events.append(
            (
                "inductance",
                value,
            )
        )

    def set_matching_capacitance_f(
        self,
        value,
    ):
        self.events.append(
            (
                "capacitance",
                value,
            )
        )


def configuration():
    return RFQMatchingConfiguration(
        frequency_hz=1.8e6,
        inductance_h=200e-6,
        capacitance_f=10e-9,
    )


def controller(
    hardware=None,
):
    if hardware is None:
        hardware = Hardware()

    clock = Clock()

    return (
        RFQSafetyController(
            hardware,
            RFQSafetyPolicy(
                q_limit=0.9,
                off_confirmation_timeout_s=1.0,
                off_poll_interval_s=0.1,
                off_consecutive_confirmations=2,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        ),
        hardware,
        clock,
    )


def test_configuration_occurs_only_after_rf_off_confirmation():
    safety, hardware, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    assert hardware.events == [
        "rf_off",
        (
            "frequency",
            1.8e6,
        ),
        (
            "inductance",
            200e-6,
        ),
        (
            "capacitance",
            10e-9,
        ),
    ]

    assert (
        safety.state
        == RFQSafetyState.CONFIGURED_OFF
    )


def test_true_readback_does_not_confirm_rf_off():
    hardware = Hardware()

    hardware.enabled_sequence = [
        True,
        True,
        False,
        False,
    ]

    safety, hardware, _ = (
        controller(
            hardware
        )
    )

    result = (
        safety.confirm_rf_off()
    )

    assert (
        result.confirmations
        == 2
    )

    assert (
        result.observations
        == 4
    )


def test_none_does_not_confirm_rf_off():
    hardware = Hardware()

    hardware.enabled_sequence = [
        None,
        None,
        False,
        False,
    ]

    safety, _, _ = (
        controller(
            hardware
        )
    )

    result = (
        safety.confirm_rf_off()
    )

    assert (
        result.observations
        == 4
    )


def test_invalid_enable_readback_fails_closed():
    hardware = Hardware()

    hardware.enabled_sequence = [
        "off",
    ]

    safety, hardware, _ = (
        controller(
            hardware
        )
    )

    with pytest.raises(
        RFQInvalidEnableReadback
    ):
        safety.confirm_rf_off()

    assert (
        safety.state
        == RFQSafetyState.FAULT_LATCHED
    )

    # Initial RF-OFF request plus best-effort emergency RF-OFF.
    assert hardware.events.count(
        "rf_off"
    ) >= 2


def test_off_confirmation_timeout_prevents_configuration():
    hardware = Hardware()

    hardware.enabled_sequence = [
        True,
    ]

    safety, hardware, _ = (
        controller(
            hardware
        )
    )

    with pytest.raises(
        RFQOffConfirmationTimeout
    ):
        safety.reconfigure(
            configuration()
        )

    assert not any(
        isinstance(
            event,
            tuple,
        )
        and event[
            0
        ] == "frequency"
        for event
        in hardware.events
    )


def test_q_exactly_at_limit_is_allowed():
    safety, _, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    result = (
        safety.run_drive_ramp(
            [
                0.1,
                0.2,
            ],
            lambda drive:
                (
                    0.5
                    if drive == 0.1
                    else 0.9
                ),
        )
    )

    assert (
        result.final_q
        == pytest.approx(
            0.9
        )
    )

    assert (
        safety.state
        == RFQSafetyState.RF_ON_SAFE
    )


def test_q_above_limit_immediately_requests_rf_off():
    safety, hardware, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    commands = []

    def executor(
        drive,
    ):
        commands.append(
            drive
        )

        if drive == 0.2:
            return 0.91

        return 0.5

    with pytest.raises(
        RFQUnsafeQError
    ):
        safety.run_drive_ramp(
            [
                0.1,
                0.2,
                0.3,
            ],
            executor,
        )

    # Third RF-drive command must never be executed.
    assert commands == [
        0.1,
        0.2,
    ]

    assert (
        safety.state
        == RFQSafetyState.FAULT_LATCHED
    )

    assert hardware.events[
        -1
    ] == "rf_off"


@pytest.mark.parametrize(
    "bad_q",
    (
        float("nan"),
        float("inf"),
        -1.0,
        "bad",
    ),
)
def test_invalid_q_fails_closed(
    bad_q,
):
    safety, hardware, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    with pytest.raises(
        RFQUnsafeQError
    ):
        safety.run_drive_ramp(
            [
                0.1,
            ],
            lambda drive:
                bad_q,
        )

    assert (
        safety.state
        == RFQSafetyState.FAULT_LATCHED
    )

    assert hardware.events[
        -1
    ] == "rf_off"


def test_executor_failure_requests_rf_off():
    safety, hardware, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    def executor(
        drive,
    ):
        raise TimeoutError(
            "scope amplitude did not settle"
        )

    with pytest.raises(
        Exception,
        match="RF drive ramp failed",
    ):
        safety.run_drive_ramp(
            [
                0.1,
            ],
            executor,
        )

    assert (
        safety.state
        == RFQSafetyState.FAULT_LATCHED
    )

    assert hardware.events[
        -1
    ] == "rf_off"


def test_fault_is_latched_against_future_ramp():
    safety, _, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    with pytest.raises(
        RFQUnsafeQError
    ):
        safety.run_drive_ramp(
            [
                0.1,
            ],
            lambda drive:
                0.95,
        )

    with pytest.raises(
        RFQFaultLatchedError
    ):
        safety.run_drive_ramp(
            [
                0.1,
            ],
            lambda drive:
                0.2,
        )


def test_fault_reset_requires_positive_rf_off_confirmation():
    safety, hardware, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    with pytest.raises(
        RFQUnsafeQError
    ):
        safety.run_drive_ramp(
            [
                0.1,
            ],
            lambda drive:
                0.95,
        )

    hardware.enabled_sequence = [
        False,
        False,
    ]

    hardware.read_index = 0

    safety.reset_fault()

    assert (
        safety.state
        == RFQSafetyState.RF_OFF_CONFIRMED
    )

    assert (
        safety.fault_reason
        is None
    )


def test_reconfigure_from_safe_rf_on_turns_rf_off_first():
    safety, hardware, _ = (
        controller()
    )

    safety.reconfigure(
        configuration()
    )

    safety.run_drive_ramp(
        [
            0.1,
        ],
        lambda drive:
            0.5,
    )

    before = len(
        hardware.events
    )

    safety.reconfigure(
        RFQMatchingConfiguration(
            frequency_hz=2.0e6,
            inductance_h=180e-6,
            capacitance_f=12e-9,
        )
    )

    new_events = (
        hardware.events[
            before:
        ]
    )

    assert new_events[
        0
    ] == "rf_off"

    assert new_events[
        1
    ][
        0
    ] == "frequency"


def test_drive_ramp_requires_prior_configuration():
    safety, _, _ = (
        controller()
    )

    with pytest.raises(
        Exception,
        match="configured",
    ):
        safety.run_drive_ramp(
            [
                0.1,
            ],
            lambda drive:
                0.2,
        )