from dataclasses import dataclass

import pytest

from sirius.readback_freshness import (
    ReadbackFreshnessPolicy,
    wait_for_fresh_parameter_readback,
)
from sirius.readback_quality import (
    ReadbackQualityPolicy,
    RejectedReadbackQualityError,
)


@dataclass
class Snapshot:
    value: float
    timestamp: float
    quality: object | None
    source: object | None = None


class Adapter:
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
        index = min(
            self.index,
            len(
                self.snapshots
            ) - 1,
        )

        self.index += 1

        return self.snapshots[
            index
        ]


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


def freshness():
    return ReadbackFreshnessPolicy(
        timeout_s=1.0,
        poll_interval_s=0.1,
    )


def quality():
    return ReadbackQualityPolicy(
        accepted_values=(
            "test-valid",
        )
    )


def test_fresh_and_accepted_readback_passes():
    adapter = Adapter(
        [
            Snapshot(
                value=1000.0,
                timestamp=11.0,
                quality="test-valid",
            )
        ]
    )

    clock = Clock()

    result = (
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=10.0,
            policy=freshness(),
            quality_policy=quality(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    assert (
        result.source_timestamp
        == pytest.approx(
            11.0
        )
    )

    assert result.quality == "test-valid"


def test_fresh_but_bad_quality_is_rejected():
    adapter = Adapter(
        [
            Snapshot(
                value=1000.0,
                timestamp=11.0,
                quality="test-bad",
            )
        ]
    )

    clock = Clock()

    with pytest.raises(
        RejectedReadbackQualityError
    ):
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=10.0,
            policy=freshness(),
            quality_policy=quality(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_stale_bad_quality_is_ignored_until_new_readback():
    adapter = Adapter(
        [
            Snapshot(
                value=900.0,
                timestamp=10.0,
                quality="test-bad",
            ),
            Snapshot(
                value=1000.0,
                timestamp=11.0,
                quality="test-valid",
            ),
        ]
    )

    clock = Clock()

    result = (
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=10.0,
            policy=freshness(),
            quality_policy=quality(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    assert (
        result.source_timestamp
        == pytest.approx(
            11.0
        )
    )

    assert result.stale_observations == 1


def test_no_quality_policy_preserves_isolated_legacy_test_path():
    adapter = Adapter(
        [
            Snapshot(
                value=1000.0,
                timestamp=11.0,
                quality=None,
            )
        ]
    )

    clock = Clock()

    result = (
        wait_for_fresh_parameter_readback(
            adapter,
            "lens2_voltage_v",
            not_before_source_timestamp=10.0,
            policy=freshness(),
            quality_policy=None,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    )

    assert result.value == pytest.approx(
        1000.0
    )