from dataclasses import dataclass

import pytest

from sirius.settling import (
    SettlingPolicy,
    SettlingTimeoutError,
    set_and_wait,
    wait_for_stable_readback,
)


@dataclass
class FakeSnapshot:
    value: float | None
    timestamp: float | None


class FakeClock:
    def __init__(self):
        self.time = 0.0

    def monotonic(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class SequenceAdapter:
    def __init__(self, observations):
        self.observations = list(observations)
        self.index = 0
        self.set_calls = []

    def set_parameter(self, parameter, value):
        self.set_calls.append(
            (parameter, value)
        )

    def read_channel(self, channel):
        if not self.observations:
            return None

        if self.index >= len(self.observations):
            return self.observations[-1]

        result = self.observations[self.index]
        self.index += 1

        return result


def test_large_command_readback_offset_is_allowed():
    adapter = SequenceAdapter(
        [
            FakeSnapshot(17800.0, 1.0),
            FakeSnapshot(18200.0, 2.0),
            FakeSnapshot(18500.0, 3.0),
            FakeSnapshot(18600.0, 4.0),
            FakeSnapshot(18603.0, 5.0),
            FakeSnapshot(18601.0, 6.0),
            FakeSnapshot(18602.0, 7.0),
        ]
    )

    clock = FakeClock()

    result = wait_for_stable_readback(
        adapter,
        "extraction_voltage_v",
        19000.0,
        SettlingPolicy(
            max_readback_span=5.0,
            timeout_s=10.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            window_samples=4,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.settled_readback == pytest.approx(
        18601.5
    )

    assert result.command_value == 19000.0

    assert result.command_readback_delta == pytest.approx(
        -398.5
    )


def test_moving_readback_is_not_accepted_as_stable():
    adapter = SequenceAdapter(
        [
            FakeSnapshot(18000.0, 1.0),
            FakeSnapshot(18200.0, 2.0),
            FakeSnapshot(18400.0, 3.0),
            FakeSnapshot(18500.0, 4.0),
            FakeSnapshot(18580.0, 5.0),
            FakeSnapshot(18600.0, 6.0),
            FakeSnapshot(18601.0, 7.0),
            FakeSnapshot(18600.0, 8.0),
            FakeSnapshot(18602.0, 9.0),
        ]
    )

    clock = FakeClock()

    result = wait_for_stable_readback(
        adapter,
        "extraction_voltage_v",
        19000.0,
        SettlingPolicy(
            max_readback_span=5.0,
            timeout_s=10.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            window_samples=4,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.samples == 9


def test_stale_timestamp_does_not_count_twice():
    adapter = SequenceAdapter(
        [
            FakeSnapshot(18600.0, 1.0),
            FakeSnapshot(18600.0, 1.0),
            FakeSnapshot(18600.0, 1.0),
            FakeSnapshot(18601.0, 2.0),
            FakeSnapshot(18600.0, 3.0),
            FakeSnapshot(18602.0, 4.0),
        ]
    )

    clock = FakeClock()

    result = wait_for_stable_readback(
        adapter,
        "extraction_voltage_v",
        19000.0,
        SettlingPolicy(
            max_readback_span=5.0,
            timeout_s=10.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            window_samples=4,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.samples == 4


def test_relative_span_can_define_stability():
    policy = SettlingPolicy(
        max_readback_span=1.0,
        relative_readback_span=0.001,
    )

    assert policy.allowed_span_for(
        19000.0
    ) == 19.0


def test_timeout_if_readback_never_stabilizes():
    observations = []

    timestamp = 1.0

    for value in (
        18000.0,
        18100.0,
        18200.0,
        18300.0,
        18400.0,
        18500.0,
    ):
        observations.append(
            FakeSnapshot(value, timestamp)
        )
        timestamp += 1.0

    adapter = SequenceAdapter(observations)
    clock = FakeClock()

    with pytest.raises(SettlingTimeoutError):
        wait_for_stable_readback(
            adapter,
            "extraction_voltage_v",
            19000.0,
            SettlingPolicy(
                max_readback_span=5.0,
                timeout_s=0.6,
                poll_interval_s=0.1,
                minimum_wait_s=0.0,
                window_samples=4,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_set_and_wait_sends_command():
    adapter = SequenceAdapter(
        [
            FakeSnapshot(18000.0, 1.0),
            FakeSnapshot(18600.0, 2.0),
            FakeSnapshot(18601.0, 3.0),
            FakeSnapshot(18600.0, 4.0),
            FakeSnapshot(18602.0, 5.0),
        ]
    )

    clock = FakeClock()

    result = set_and_wait(
        adapter,
        "extraction_voltage_v",
        19000.0,
        SettlingPolicy(
            max_readback_span=5.0,
            timeout_s=5.0,
            poll_interval_s=0.1,
            minimum_wait_s=0.0,
            window_samples=4,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert adapter.set_calls == [
        ("extraction_voltage_v", 19000.0)
    ]

    assert result.settled_readback == pytest.approx(
        18600.75
    )


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        SettlingPolicy(
            max_readback_span=-1.0
        )

    with pytest.raises(ValueError):
        SettlingPolicy(
            max_readback_span=1.0,
            window_samples=1,
        )