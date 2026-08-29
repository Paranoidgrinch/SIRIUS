import pytest

from sirius.command_cadence import (
    CommandCadenceClockError,
    CommandCadenceController,
)


class Clock:
    def __init__(
        self,
    ):
        self.now = 0.0

        self.sleeps = []

    def monotonic(
        self,
    ):
        return self.now

    def sleep(
        self,
        seconds,
    ):
        seconds = float(
            seconds
        )

        self.sleeps.append(
            seconds
        )

        self.now += seconds


def test_first_command_does_not_wait():
    clock = Clock()

    controller = (
        CommandCadenceController(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    result = controller.reserve(
        "lens2_voltage_v",
        0.5,
    )

    assert result.waited_s == 0.0
    assert clock.sleeps == []


def test_second_command_waits_for_remaining_interval():
    clock = Clock()

    controller = (
        CommandCadenceController(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    controller.reserve(
        "lens2_voltage_v",
        0.5,
    )

    clock.now = 0.2

    result = controller.reserve(
        "lens2_voltage_v",
        0.5,
    )

    assert (
        result.waited_s
        == pytest.approx(
            0.3
        )
    )

    assert (
        clock.now
        == pytest.approx(
            0.5
        )
    )


def test_no_wait_when_interval_already_elapsed():
    clock = Clock()

    controller = (
        CommandCadenceController(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    controller.reserve(
        "lens2_voltage_v",
        0.5,
    )

    clock.now = 1.0

    result = controller.reserve(
        "lens2_voltage_v",
        0.5,
    )

    assert result.waited_s == 0.0


def test_different_channels_have_independent_cadence():
    clock = Clock()

    controller = (
        CommandCadenceController(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    controller.reserve(
        "steerer_x1_v",
        0.5,
    )

    result = controller.reserve(
        "steerer_y1_v",
        0.5,
    )

    assert result.waited_s == 0.0


def test_zero_interval_is_supported_for_non_real_unit_tests():
    clock = Clock()

    controller = (
        CommandCadenceController(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    controller.reserve(
        "test",
        0.0,
    )

    controller.reserve(
        "test",
        0.0,
    )

    assert clock.sleeps == []


def test_clock_moving_backwards_is_hard_failure():
    clock = Clock()

    controller = (
        CommandCadenceController(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    clock.now = 5.0

    controller.reserve(
        "magnet_current_a",
        1.0,
    )

    clock.now = 4.0

    with pytest.raises(
        CommandCadenceClockError,
        match="backwards",
    ):
        controller.reserve(
            "magnet_current_a",
            1.0,
        )