import pytest

from sirius.settling import (
    SettlingPolicy,
    SettlingTimeoutError,
    is_within_tolerance,
    set_and_wait,
    wait_for_parameter,
)


class FakeClock:
    def __init__(self):
        self.time = 0.0

    def monotonic(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class SequenceAdapter:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0
        self.set_calls = []

    def set_parameter(self, parameter, target):
        self.set_calls.append(
            (parameter, target)
        )

    def read_parameter(self, parameter):
        if not self.values:
            return None

        if self.index >= len(self.values):
            return self.values[-1]

        value = self.values[self.index]
        self.index += 1
        return value


def test_absolute_tolerance():
    policy = SettlingPolicy(
        absolute_tolerance=5.0,
    )

    assert policy.tolerance_for(1000.0) == 5.0


def test_relative_tolerance_can_dominate():
    policy = SettlingPolicy(
        absolute_tolerance=1.0,
        relative_tolerance=0.01,
    )

    assert policy.tolerance_for(1000.0) == 10.0


def test_within_tolerance():
    assert is_within_tolerance(
        100.4,
        100.0,
        0.5,
    )

    assert not is_within_tolerance(
        100.6,
        100.0,
        0.5,
    )


def test_parameter_must_be_stable_for_multiple_samples():
    adapter = SequenceAdapter(
        [
            900.0,
            980.0,
            999.0,
            1001.0,
            1000.5,
        ]
    )

    clock = FakeClock()

    result = wait_for_parameter(
        adapter,
        "test_parameter",
        1000.0,
        SettlingPolicy(
            absolute_tolerance=2.0,
            timeout_s=10.0,
            poll_interval_s=0.1,
            consecutive_samples=3,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.final_readback == 1000.5
    assert result.samples == 5
    assert result.consecutive_samples == 3


def test_out_of_tolerance_sample_resets_stability_counter():
    adapter = SequenceAdapter(
        [
            1000.0,
            1001.0,
            1010.0,
            1000.5,
            999.5,
            1000.2,
        ]
    )

    clock = FakeClock()

    result = wait_for_parameter(
        adapter,
        "test_parameter",
        1000.0,
        SettlingPolicy(
            absolute_tolerance=2.0,
            timeout_s=10.0,
            poll_interval_s=0.1,
            consecutive_samples=3,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.samples == 6


def test_missing_readbacks_do_not_count_as_stable():
    adapter = SequenceAdapter(
        [
            None,
            None,
            1000.0,
            1000.0,
            1000.0,
        ]
    )

    clock = FakeClock()

    result = wait_for_parameter(
        adapter,
        "test_parameter",
        1000.0,
        SettlingPolicy(
            absolute_tolerance=1.0,
            timeout_s=10.0,
            poll_interval_s=0.1,
            consecutive_samples=3,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.samples == 3


def test_timeout_if_parameter_never_reaches_target():
    adapter = SequenceAdapter(
        [
            800.0,
            850.0,
            900.0,
        ]
    )

    clock = FakeClock()

    with pytest.raises(SettlingTimeoutError) as exc:
        wait_for_parameter(
            adapter,
            "test_parameter",
            1000.0,
            SettlingPolicy(
                absolute_tolerance=2.0,
                timeout_s=0.5,
                poll_interval_s=0.1,
                consecutive_samples=3,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert exc.value.target == 1000.0
    assert exc.value.last_readback == 900.0


def test_set_and_wait_sets_parameter_before_polling():
    adapter = SequenceAdapter(
        [
            99.0,
            100.0,
            100.0,
        ]
    )

    clock = FakeClock()

    result = set_and_wait(
        adapter,
        "steerer_x1_v",
        100.0,
        SettlingPolicy(
            absolute_tolerance=1.0,
            timeout_s=5.0,
            poll_interval_s=0.1,
            consecutive_samples=2,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert adapter.set_calls == [
        ("steerer_x1_v", 100.0)
    ]

    assert result.final_readback == 100.0


def test_policy_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SettlingPolicy(
            absolute_tolerance=-1.0,
        )

    with pytest.raises(ValueError):
        SettlingPolicy(
            absolute_tolerance=1.0,
            consecutive_samples=0,
        )