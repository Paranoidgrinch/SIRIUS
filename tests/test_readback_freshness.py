from dataclasses import dataclass

import pytest

from sirius.readback_freshness import (
    ReadbackFreshnessPolicy,
    ReadbackFreshnessTimeoutError,
    ReadbackTimestampInvalidError,
    ReadbackTimestampMissingError,
    wait_for_fresh_parameter_readback,
)


@dataclass
class Snapshot:
    value: float | None
    timestamp: float | None


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


class SequenceAdapter:
    def __init__(
        self,
        snapshots,
    ):
        self.snapshots = list(
            snapshots
        )

        self.index = 0

    def read_parameter_snapshot(
        self,
        name,
    ):
        if not self.snapshots:
            return None

        index = min(
            self.index,
            len(
                self.snapshots
            ) - 1,
        )

        result = (
            self.snapshots[
                index
            ]
        )

        self.index += 1

        return result


def policy():
    return ReadbackFreshnessPolicy(
        timeout_s=1.0,
        poll_interval_s=0.1,
    )


def test_cached_timestamp_cannot_confirm_new_command():
    adapter = SequenceAdapter(
        [
            Snapshot(
                1000.0,
                10.0,
            ),
            Snapshot(
                1000.0,
                10.0,
            ),
            Snapshot(
                1001.0,
                11.0,
            ),
        ]
    )

    clock = Clock()

    result = (
        wait_for_fresh_parameter_readback(
            adapter,
            "einzel_lens_voltage_v",
            not_before_source_timestamp=(
                10.0
            ),
            policy=policy(),
            monotonic=(
                clock.monotonic
            ),
            sleep=(
                clock.sleep
            ),
        )
    )

    assert (
        result.source_timestamp
        == pytest.approx(
            11.0
        )
    )

    assert (
        result.stale_observations
        == 2
    )


def test_timestamp_must_be_strictly_newer():
    adapter = SequenceAdapter(
        [
            Snapshot(
                1000.0,
                100.0,
            ),
            Snapshot(
                1000.0,
                100.0,
            ),
            Snapshot(
                1000.0,
                100.0001,
            ),
        ]
    )

    clock = Clock()

    result = (
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=(
                100.0
            ),
            policy=policy(),
            monotonic=(
                clock.monotonic
            ),
            sleep=(
                clock.sleep
            ),
        )
    )

    assert (
        result.source_timestamp
        > 100.0
    )


def test_out_of_order_timestamp_cannot_confirm_command():
    adapter = SequenceAdapter(
        [
            Snapshot(
                1000.0,
                49.0,
            ),
            Snapshot(
                1000.0,
                48.0,
            ),
            Snapshot(
                1000.0,
                51.0,
            ),
        ]
    )

    clock = Clock()

    result = (
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=(
                50.0
            ),
            policy=policy(),
            monotonic=(
                clock.monotonic
            ),
            sleep=(
                clock.sleep
            ),
        )
    )

    assert (
        result.source_timestamp
        == pytest.approx(
            51.0
        )
    )

    assert (
        result.stale_observations
        == 2
    )


def test_missing_timestamp_is_hard_failure():
    adapter = SequenceAdapter(
        [
            Snapshot(
                1000.0,
                None,
            )
        ]
    )

    clock = Clock()

    with pytest.raises(
        ReadbackTimestampMissingError
    ):
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=(
                10.0
            ),
            policy=policy(),
            monotonic=(
                clock.monotonic
            ),
            sleep=(
                clock.sleep
            ),
        )


@pytest.mark.parametrize(
    "timestamp",
    (
        float("nan"),
        float("inf"),
        -float("inf"),
        "abc",
    ),
)
def test_invalid_timestamp_is_hard_failure(
    timestamp,
):
    adapter = SequenceAdapter(
        [
            Snapshot(
                1000.0,
                timestamp,
            )
        ]
    )

    clock = Clock()

    with pytest.raises(
        ReadbackTimestampInvalidError
    ):
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=(
                10.0
            ),
            policy=policy(),
            monotonic=(
                clock.monotonic
            ),
            sleep=(
                clock.sleep
            ),
        )


def test_permanently_cached_readback_times_out():
    adapter = SequenceAdapter(
        [
            Snapshot(
                1000.0,
                10.0,
            )
        ]
    )

    clock = Clock()

    with pytest.raises(
        ReadbackFreshnessTimeoutError
    ):
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=(
                10.0
            ),
            policy=ReadbackFreshnessPolicy(
                timeout_s=0.5,
                poll_interval_s=0.1,
            ),
            monotonic=(
                clock.monotonic
            ),
            sleep=(
                clock.sleep
            ),
        )


def test_first_timestamped_readback_is_allowed_when_no_barrier_exists():
    adapter = SequenceAdapter(
        [
            None,
            Snapshot(
                500.0,
                1.0,
            ),
        ]
    )

    clock = Clock()

    result = (
        wait_for_fresh_parameter_readback(
            adapter,
            "guidefield1_voltage_v",
            not_before_source_timestamp=None,
            policy=policy(),
            monotonic=(
                clock.monotonic
            ),
            sleep=(
                clock.sleep
            ),
        )
    )

    assert (
        result.source_timestamp
        == pytest.approx(
            1.0
        )
    )