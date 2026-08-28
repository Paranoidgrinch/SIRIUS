import pytest

from sirius.cup_ack import (
    CupSelectionPolicy,
    CupSelectionTimeoutError,
    InvalidCupReadbackError,
    select_cup_and_wait,
)


class FakeClock:
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


def test_command_is_sent_exactly_once():
    clock = FakeClock()

    commands = []

    readbacks = iter(
        (
            3,
            4,
            4,
        )
    )

    result = select_cup_and_wait(
        select_cup=(
            commands.append
        ),
        read_selected_cup=lambda:
            next(
                readbacks
            ),
        target_cup=4,
        policy=CupSelectionPolicy(
            timeout_s=5.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            consecutive_confirmations=2,
        ),
        monotonic=(
            clock.monotonic
        ),
        sleeper=(
            clock.sleep
        ),
    )

    assert commands == [
        4
    ]

    assert (
        result.confirmed_cup
        == 4
    )


def test_requires_consecutive_confirmations():
    clock = FakeClock()

    readbacks = iter(
        (
            4,
            3,
            4,
            4,
        )
    )

    result = select_cup_and_wait(
        select_cup=lambda cup: None,
        read_selected_cup=lambda:
            next(
                readbacks
            ),
        target_cup=4,
        policy=CupSelectionPolicy(
            timeout_s=5.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            consecutive_confirmations=2,
        ),
        monotonic=(
            clock.monotonic
        ),
        sleeper=(
            clock.sleep
        ),
    )

    assert (
        result.sample_count
        == 4
    )

    assert (
        result.samples[
            0
        ].matched_target
        is True
    )

    assert (
        result.samples[
            1
        ].matched_target
        is False
    )


def test_none_readback_resets_confirmation_counter():
    clock = FakeClock()

    readbacks = iter(
        (
            4,
            None,
            4,
            4,
        )
    )

    result = select_cup_and_wait(
        select_cup=lambda cup: None,
        read_selected_cup=lambda:
            next(
                readbacks
            ),
        target_cup=4,
        policy=CupSelectionPolicy(
            timeout_s=5.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            consecutive_confirmations=2,
        ),
        monotonic=(
            clock.monotonic
        ),
        sleeper=(
            clock.sleep
        ),
    )

    assert (
        result.sample_count
        == 4
    )


def test_minimum_wait_is_enforced_even_if_target_is_immediate():
    clock = FakeClock()

    result = select_cup_and_wait(
        select_cup=lambda cup: None,
        read_selected_cup=lambda: 4,
        target_cup=4,
        policy=CupSelectionPolicy(
            timeout_s=5.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.3,
            consecutive_confirmations=2,
        ),
        monotonic=(
            clock.monotonic
        ),
        sleeper=(
            clock.sleep
        ),
    )

    assert (
        result.elapsed_s
        >= 0.3
    )

    assert (
        result.confirmation_count
        >= 2
    )


def test_wrong_cup_times_out():
    clock = FakeClock()

    with pytest.raises(
        CupSelectionTimeoutError
    ):
        select_cup_and_wait(
            select_cup=lambda cup: None,
            read_selected_cup=lambda: 2,
            target_cup=4,
            policy=CupSelectionPolicy(
                timeout_s=0.5,
                poll_interval_s=0.1,
                minimum_wait_s=0.0,
                consecutive_confirmations=2,
            ),
            monotonic=(
                clock.monotonic
            ),
            sleeper=(
                clock.sleep
            ),
        )


def test_no_readback_times_out():
    clock = FakeClock()

    with pytest.raises(
        CupSelectionTimeoutError
    ):
        select_cup_and_wait(
            select_cup=lambda cup: None,
            read_selected_cup=lambda: None,
            target_cup=4,
            policy=CupSelectionPolicy(
                timeout_s=0.5,
                poll_interval_s=0.1,
                minimum_wait_s=0.0,
                consecutive_confirmations=2,
            ),
            monotonic=(
                clock.monotonic
            ),
            sleeper=(
                clock.sleep
            ),
        )


@pytest.mark.parametrize(
    "readback",
    (
        0,
        7,
        -1,
        2.5,
        float("nan"),
        float("inf"),
        True,
        "abc",
    ),
)
def test_invalid_backend_readback_is_not_silently_accepted(
    readback,
):
    clock = FakeClock()

    with pytest.raises(
        InvalidCupReadbackError
    ):
        select_cup_and_wait(
            select_cup=lambda cup: None,
            read_selected_cup=lambda:
                readback,
            target_cup=4,
            policy=CupSelectionPolicy(
                timeout_s=1.0,
                poll_interval_s=0.1,
                minimum_wait_s=0.0,
                consecutive_confirmations=1,
            ),
            monotonic=(
                clock.monotonic
            ),
            sleeper=(
                clock.sleep
            ),
        )


@pytest.mark.parametrize(
    "target",
    (
        0,
        7,
        -1,
        2.5,
        True,
    ),
)
def test_invalid_requested_cup_is_rejected(
    target,
):
    with pytest.raises(
        ValueError
    ):
        select_cup_and_wait(
            select_cup=lambda cup: None,
            read_selected_cup=lambda: 1,
            target_cup=target,
        )


def test_readback_strings_containing_integer_are_supported():
    clock = FakeClock()

    result = select_cup_and_wait(
        select_cup=lambda cup: None,
        read_selected_cup=lambda:
            "4",
        target_cup=4,
        policy=CupSelectionPolicy(
            timeout_s=1.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            consecutive_confirmations=1,
        ),
        monotonic=(
            clock.monotonic
        ),
        sleeper=(
            clock.sleep
        ),
    )

    assert (
        result.confirmed_cup
        == 4
    )


def test_result_keeps_full_acknowledgement_history():
    clock = FakeClock()

    readbacks = iter(
        (
            None,
            2,
            3,
            4,
            4,
        )
    )

    result = select_cup_and_wait(
        select_cup=lambda cup: None,
        read_selected_cup=lambda:
            next(
                readbacks
            ),
        target_cup=4,
        policy=CupSelectionPolicy(
            timeout_s=5.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            consecutive_confirmations=2,
        ),
        monotonic=(
            clock.monotonic
        ),
        sleeper=(
            clock.sleep
        ),
    )

    assert tuple(
        sample.selected_cup
        for sample
        in result.samples
    ) == (
        None,
        2,
        3,
        4,
        4,
    )