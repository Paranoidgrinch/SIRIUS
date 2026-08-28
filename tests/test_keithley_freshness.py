from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sirius.flavia_adapter import (
    FlaviaBackendAdapter,
    KeithleyTimestampError,
)
from sirius.measurement import (
    BeamMeasurementFreshnessError,
    MeasurementPolicy,
    measure_beam_current,
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


class BarrierAdapter:
    def __init__(
        self,
        barrier,
        observations,
    ):
        self.barrier = barrier
        self.observations = list(
            observations
        )
        self.index = 0

    def capture_beam_current_freshness_barrier(
        self,
    ):
        return self.barrier

    def read_channel(
        self,
        channel_name,
    ):
        if not self.observations:
            return None

        index = min(
            self.index,
            len(self.observations) - 1,
        )

        result = self.observations[
            index
        ]

        self.index += 1

        return result


def policy():
    return MeasurementPolicy(
        min_samples=4,
        max_samples=20,
        min_duration_s=0.0,
        max_duration_s=5.0,
        poll_interval_s=0.1,
        relative_sem_target=0.01,
        absolute_sem_target_a=1e-13,
        drift_window_samples=4,
        drift_tolerance_factor=2.0,
    )


def test_flavia_adapter_captures_current_keithley_timestamp():
    channel = SimpleNamespace(
        value=8e-9,
        timestamp=123.5,
        quality="good",
        source="keithley",
    )

    backend = SimpleNamespace(
        model={
            "keithley/current_A":
                channel
        }
    )

    adapter = FlaviaBackendAdapter(
        backend
    )

    assert (
        adapter.capture_beam_current_freshness_barrier()
        == pytest.approx(
            123.5
        )
    )


def test_flavia_adapter_rejects_value_without_timestamp():
    channel = SimpleNamespace(
        value=8e-9,
        timestamp=None,
        quality="good",
        source="keithley",
    )

    backend = SimpleNamespace(
        model={
            "keithley/current_A":
                channel
        }
    )

    adapter = FlaviaBackendAdapter(
        backend
    )

    with pytest.raises(
        KeithleyTimestampError
    ):
        adapter.capture_beam_current_freshness_barrier()


def test_cached_pre_measurement_sample_is_rejected():
    adapter = BarrierAdapter(
        10.0,
        [
            FakeSnapshot(
                99e-9,
                10.0,
            ),
            FakeSnapshot(
                99e-9,
                10.0,
            ),
            FakeSnapshot(
                8e-9,
                11.0,
            ),
            FakeSnapshot(
                8e-9,
                12.0,
            ),
            FakeSnapshot(
                8e-9,
                13.0,
            ),
            FakeSnapshot(
                8e-9,
                14.0,
            ),
        ],
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        policy(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.n == 4

    assert result.mean_a == pytest.approx(
        8e-9
    )

    assert tuple(
        sample.source_timestamp
        for sample
        in result.samples
    ) == (
        11.0,
        12.0,
        13.0,
        14.0,
    )


def test_out_of_order_timestamp_is_rejected():
    adapter = BarrierAdapter(
        10.0,
        [
            FakeSnapshot(
                8e-9,
                11.0,
            ),
            FakeSnapshot(
                100e-9,
                10.5,
            ),
            FakeSnapshot(
                8e-9,
                12.0,
            ),
            FakeSnapshot(
                8e-9,
                13.0,
            ),
            FakeSnapshot(
                8e-9,
                14.0,
            ),
        ],
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        policy(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert tuple(
        sample.source_timestamp
        for sample
        in result.samples
    ) == (
        11.0,
        12.0,
        13.0,
        14.0,
    )

    assert result.mean_a == pytest.approx(
        8e-9
    )


def test_strict_measurement_rejects_missing_sample_timestamp():
    adapter = BarrierAdapter(
        10.0,
        [
            FakeSnapshot(
                8e-9,
                None,
            ),
        ],
    )

    clock = FakeClock()

    with pytest.raises(
        BeamMeasurementFreshnessError,
        match="cannot be proven",
    ):
        measure_beam_current(
            adapter,
            policy(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_explicit_barrier_works_for_generic_adapter():
    class Adapter:
        def __init__(self):
            self.observations = iter(
                [
                    FakeSnapshot(
                        100e-9,
                        20.0,
                    ),
                    FakeSnapshot(
                        8e-9,
                        21.0,
                    ),
                    FakeSnapshot(
                        8e-9,
                        22.0,
                    ),
                    FakeSnapshot(
                        8e-9,
                        23.0,
                    ),
                    FakeSnapshot(
                        8e-9,
                        24.0,
                    ),
                ]
            )

        def read_channel(
            self,
            channel_name,
        ):
            return next(
                self.observations
            )

    clock = FakeClock()

    result = measure_beam_current(
        Adapter(),
        policy(),
        not_before_source_timestamp=20.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.n == 4

    assert tuple(
        sample.source_timestamp
        for sample
        in result.samples
    ) == (
        21.0,
        22.0,
        23.0,
        24.0,
    )