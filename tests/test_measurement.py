from dataclasses import dataclass

import pytest

from sirius.measurement import (
    BeamMeasurementNoDataError,
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


class SequenceAdapter:
    def __init__(self, observations):
        self.observations = list(observations)
        self.index = 0

    def read_channel(self, channel):
        if not self.observations:
            return None

        if self.index >= len(self.observations):
            return self.observations[-1]

        result = self.observations[self.index]
        self.index += 1

        return result


def snapshots(values):
    return [
        FakeSnapshot(
            value=value,
            timestamp=float(index + 1),
        )
        for index, value in enumerate(values)
    ]


def test_stable_nanamp_signal_finishes_early():
    adapter = SequenceAdapter(
        snapshots(
            [
                8.00e-9,
                8.01e-9,
                7.99e-9,
                8.00e-9,
                8.01e-9,
                8.00e-9,
                8.00e-9,
                8.00e-9,
            ]
        )
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            max_samples=50,
            min_duration_s=0.0,
            max_duration_s=5.0,
            poll_interval_s=0.1,
            relative_sem_target=0.01,
            absolute_sem_target_a=1e-13,
            drift_window_samples=6,
            drift_tolerance_factor=2.0,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.stop_reason == "precision_reached"
    assert result.n == 6
    assert result.mean_a == pytest.approx(
        8.001666666666667e-9
    )
    assert result.sem_a < result.precision_threshold_a


def test_noisy_weak_signal_is_measured_longer():
    values = [
        2.0e-12,
        3.0e-12,
        1.0e-12,
        3.5e-12,
        0.8e-12,
        2.8e-12,
        1.2e-12,
        3.1e-12,
        0.9e-12,
        2.7e-12,
        1.1e-12,
        3.0e-12,
    ]

    adapter = SequenceAdapter(
        snapshots(values)
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            max_samples=12,
            min_duration_s=0.0,
            max_duration_s=10.0,
            poll_interval_s=0.1,
            relative_sem_target=0.01,
            absolute_sem_target_a=5e-14,
            drift_window_samples=6,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.n == 12
    assert result.stop_reason == "max_samples"


def test_picoamp_signal_can_still_reach_absolute_precision():
    adapter = SequenceAdapter(
        snapshots(
            [
                2.00e-12,
                2.02e-12,
                1.99e-12,
                2.01e-12,
                2.00e-12,
                2.01e-12,
            ]
        )
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            max_samples=50,
            min_duration_s=0.0,
            max_duration_s=5.0,
            poll_interval_s=0.1,
            relative_sem_target=0.005,
            absolute_sem_target_a=1e-13,
            drift_window_samples=6,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.stop_reason == "precision_reached"
    assert result.mean_a == pytest.approx(
        2.005e-12
    )


def test_signal_below_known_noise_floor_can_stop_early():
    adapter = SequenceAdapter(
        snapshots(
            [
                1.00e-13,
                1.10e-13,
                0.90e-13,
                1.00e-13,
                1.05e-13,
                0.95e-13,
            ]
        )
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            max_samples=50,
            min_duration_s=0.0,
            max_duration_s=5.0,
            poll_interval_s=0.1,
            relative_sem_target=0.001,
            absolute_sem_target_a=1e-15,
            drift_window_samples=6,
            below_noise_sigma=3.0,
        ),
        noise_floor_a=5e-13,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.stop_reason == "below_noise_floor"
    assert result.below_noise_floor is True


def test_stale_flavia_timestamp_is_not_counted_twice():
    observations = [
        FakeSnapshot(8.0e-9, 1.0),
        FakeSnapshot(8.0e-9, 1.0),
        FakeSnapshot(8.0e-9, 1.0),
        FakeSnapshot(8.0e-9, 2.0),
        FakeSnapshot(8.0e-9, 3.0),
        FakeSnapshot(8.0e-9, 4.0),
        FakeSnapshot(8.0e-9, 5.0),
        FakeSnapshot(8.0e-9, 6.0),
    ]

    adapter = SequenceAdapter(observations)
    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            max_samples=20,
            min_duration_s=0.0,
            max_duration_s=5.0,
            poll_interval_s=0.1,
            relative_sem_target=0.01,
            absolute_sem_target_a=1e-13,
            drift_window_samples=6,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.n == 6


def test_drifting_signal_is_not_accepted_as_precise():
    adapter = SequenceAdapter(
        snapshots(
            [
                8.00e-9,
                8.10e-9,
                8.20e-9,
                8.30e-9,
                8.40e-9,
                8.50e-9,
                8.60e-9,
                8.70e-9,
                8.80e-9,
                8.90e-9,
            ]
        )
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            max_samples=10,
            min_duration_s=0.0,
            max_duration_s=5.0,
            poll_interval_s=0.1,
            relative_sem_target=0.10,
            absolute_sem_target_a=1e-13,
            drift_window_samples=6,
            drift_tolerance_factor=0.1,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.stop_reason == "max_samples"
    assert result.n == 10
    assert result.drift_delta_a is not None


def test_current_magnitude_is_used():
    adapter = SequenceAdapter(
        snapshots(
            [
                -8.00e-9,
                -8.01e-9,
                -7.99e-9,
                -8.00e-9,
                -8.00e-9,
                -8.00e-9,
            ]
        )
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            min_duration_s=0.0,
            max_duration_s=5.0,
            poll_interval_s=0.1,
            drift_window_samples=6,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.mean_a > 0


def test_no_fresh_samples_raises_error():
    adapter = SequenceAdapter([])
    clock = FakeClock()

    with pytest.raises(
        BeamMeasurementNoDataError
    ):
        measure_beam_current(
            adapter,
            MeasurementPolicy(
                min_samples=6,
                min_duration_s=0.0,
                max_duration_s=0.5,
                poll_interval_s=0.1,
                drift_window_samples=6,
            ),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_raw_samples_are_preserved():
    adapter = SequenceAdapter(
        snapshots(
            [
                1.00e-9,
                1.01e-9,
                0.99e-9,
                1.00e-9,
                1.00e-9,
                1.00e-9,
            ]
        )
    )

    clock = FakeClock()

    result = measure_beam_current(
        adapter,
        MeasurementPolicy(
            min_samples=6,
            min_duration_s=0.0,
            max_duration_s=5.0,
            poll_interval_s=0.1,
            drift_window_samples=6,
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert len(result.samples) == result.n
    assert result.samples[0].current_a == 1.00e-9
    assert result.samples[0].source_timestamp == 1.0


def test_invalid_measurement_policy_is_rejected():
    with pytest.raises(ValueError):
        MeasurementPolicy(
            min_samples=1
        )

    with pytest.raises(ValueError):
        MeasurementPolicy(
            min_samples=10,
            max_samples=5,
        )

    with pytest.raises(ValueError):
        MeasurementPolicy(
            drift_window_samples=2
        )