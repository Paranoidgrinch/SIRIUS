import pytest

from sirius.measurement import BeamMeasurement
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
    transmission_from_reference,
)


def measurement(
    mean_a,
    sem_a,
):
    return BeamMeasurement(
        mean_a=mean_a,
        sigma_a=sem_a,
        sem_a=sem_a,
        n=10,
        duration_s=1.0,
        relative_sem=None,
        precision_threshold_a=sem_a,
        drift_delta_a=0.0,
        stop_reason="test",
        below_noise_floor=False,
        samples=(),
    )


def reference(
    current_a,
    sem_a,
    monotonic_s,
    *,
    mass_u=60.0,
    state_id="cup1-state",
):
    return SourceReference(
        measurement=measurement(
            current_a,
            sem_a,
        ),
        state_id=state_id,
        mass_u=mass_u,
        monotonic_s=monotonic_s,
        created_at_utc="2026-08-26T12:00:00+00:00",
    )


def test_reference_is_due_before_first_measurement():
    tracker = SourceReferenceTracker(
        interval_s=600.0
    )

    assert tracker.is_due(0.0) is True


def test_reference_is_not_due_before_ten_minutes():
    tracker = SourceReferenceTracker(
        interval_s=600.0
    )

    tracker.add(
        reference(
            10e-9,
            0.1e-9,
            100.0,
        )
    )

    assert tracker.is_due(699.9) is False


def test_reference_is_due_after_ten_minutes():
    tracker = SourceReferenceTracker(
        interval_s=600.0
    )

    tracker.add(
        reference(
            10e-9,
            0.1e-9,
            100.0,
        )
    )

    assert tracker.is_due(700.0) is True


def test_seconds_until_due():
    tracker = SourceReferenceTracker(
        interval_s=600.0
    )

    tracker.add(
        reference(
            10e-9,
            0.1e-9,
            100.0,
        )
    )

    assert tracker.seconds_until_due(
        400.0
    ) == pytest.approx(300.0)


def test_latest_reference_is_returned():
    tracker = SourceReferenceTracker()

    first = reference(
        10e-9,
        0.1e-9,
        0.0,
        state_id="first",
    )

    second = reference(
        9e-9,
        0.1e-9,
        600.0,
        state_id="second",
    )

    tracker.add(first)
    tracker.add(second)

    assert tracker.latest is second


def test_source_drift_is_calculated_from_first_reference():
    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9,
            0.1e-9,
            0.0,
        )
    )

    tracker.add(
        reference(
            9e-9,
            0.1e-9,
            600.0,
        )
    )

    assert tracker.relative_source_drift() == pytest.approx(
        -0.10
    )


def test_reference_tracker_rejects_mixed_masses():
    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9,
            0.1e-9,
            0.0,
            mass_u=60.0,
        )
    )

    with pytest.raises(ValueError):
        tracker.add(
            reference(
                10e-9,
                0.1e-9,
                600.0,
                mass_u=180.0,
            )
        )


def test_reference_tracker_rejects_time_reversal():
    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9,
            0.1e-9,
            600.0,
        )
    )

    with pytest.raises(ValueError):
        tracker.add(
            reference(
                10e-9,
                0.1e-9,
                500.0,
            )
        )


def test_transmission_relative_to_cup1():
    cup_measurement = measurement(
        8.0e-9,
        0.05e-9,
    )

    cup1_reference = reference(
        10.0e-9,
        0.05e-9,
        0.0,
    )

    result = transmission_from_reference(
        3,
        cup_measurement,
        cup1_reference,
    )

    assert result.transmission == pytest.approx(
        0.8
    )

    assert result.transmission_percent == pytest.approx(
        80.0
    )


def test_cup1_reference_itself_is_100_percent():
    cup1_reference = reference(
        10.0e-9,
        0.05e-9,
        0.0,
    )

    result = transmission_from_reference(
        1,
        cup1_reference.measurement,
        cup1_reference,
    )

    assert result.transmission == pytest.approx(
        1.0
    )

    assert result.transmission_percent == pytest.approx(
        100.0
    )


def test_transmission_uncertainty_is_propagated():
    cup_measurement = measurement(
        8.0e-9,
        0.08e-9,
    )

    cup1_reference = reference(
        10.0e-9,
        0.10e-9,
        0.0,
    )

    result = transmission_from_reference(
        6,
        cup_measurement,
        cup1_reference,
    )

    expected_relative_uncertainty = (
        (0.08 / 8.0) ** 2
        + (0.10 / 10.0) ** 2
    ) ** 0.5

    expected_sem = (
        0.8
        * expected_relative_uncertainty
    )

    assert result.transmission_sem == pytest.approx(
        expected_sem
    )


def test_zero_downstream_current_is_supported():
    cup_measurement = measurement(
        0.0,
        0.1e-12,
    )

    cup1_reference = reference(
        10e-9,
        0.1e-9,
        0.0,
    )

    result = transmission_from_reference(
        6,
        cup_measurement,
        cup1_reference,
    )

    assert result.transmission == 0.0
    assert result.transmission_sem > 0


def test_invalid_cup_is_rejected():
    cup1_reference = reference(
        10e-9,
        0.1e-9,
        0.0,
    )

    with pytest.raises(ValueError):
        transmission_from_reference(
            7,
            measurement(
                8e-9,
                0.1e-9,
            ),
            cup1_reference,
        )


def test_nonpositive_reference_is_rejected():
    tracker = SourceReferenceTracker()

    with pytest.raises(ValueError):
        tracker.add(
            reference(
                0.0,
                0.1e-12,
                0.0,
            )
        )


def test_invalid_reference_interval_is_rejected():
    with pytest.raises(ValueError):
        SourceReferenceTracker(
            interval_s=0.0
        )