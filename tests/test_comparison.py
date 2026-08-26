import pytest

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
    compare_measurements,
)
from sirius.measurement import BeamMeasurement


def measurement(
    mean_a,
    sem_a,
    *,
    sigma_a=None,
    below_noise_floor=False,
):
    if sigma_a is None:
        sigma_a = sem_a

    return BeamMeasurement(
        mean_a=mean_a,
        sigma_a=sigma_a,
        sem_a=sem_a,
        n=10,
        duration_s=1.0,
        relative_sem=None,
        precision_threshold_a=sem_a,
        drift_delta_a=0.0,
        stop_reason="test",
        below_noise_floor=below_noise_floor,
        samples=(),
    )


def test_clear_improvement_is_better():
    baseline = measurement(
        8.00e-9,
        0.02e-9,
    )

    candidate = measurement(
        8.40e-9,
        0.02e-9,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_relative_improvement=0.002,
        ),
    )

    assert result.decision == ComparisonDecision.BETTER
    assert result.delta_a == pytest.approx(
        0.40e-9
    )


def test_tiny_apparent_improvement_inside_noise_is_indistinguishable():
    baseline = measurement(
        8.00e-9,
        0.10e-9,
    )

    candidate = measurement(
        8.05e-9,
        0.10e-9,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert (
        result.decision
        == ComparisonDecision.INDISTINGUISHABLE
    )


def test_clear_decrease_is_worse():
    baseline = measurement(
        8.00e-9,
        0.02e-9,
    )

    candidate = measurement(
        7.50e-9,
        0.02e-9,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(),
    )

    assert result.decision == ComparisonDecision.WORSE


def test_statistically_clear_but_irrelevant_tiny_change_can_be_ignored():
    baseline = measurement(
        1.0000e-6,
        1.0e-12,
    )

    candidate = measurement(
        1.0005e-6,
        1.0e-12,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_relative_improvement=0.001,
        ),
    )

    assert (
        result.decision
        == ComparisonDecision.INDISTINGUISHABLE
    )

    assert result.practical_margin_a == pytest.approx(
        1.0e-9
    )


def test_relative_improvement_works_for_picoamp_signal():
    baseline = measurement(
        2.00e-12,
        0.01e-12,
    )

    candidate = measurement(
        2.20e-12,
        0.01e-12,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_relative_improvement=0.01,
        ),
    )

    assert result.decision == ComparisonDecision.BETTER
    assert result.relative_delta == pytest.approx(
        0.10
    )


def test_absolute_improvement_threshold_can_dominate():
    baseline = measurement(
        2.00e-12,
        0.001e-12,
    )

    candidate = measurement(
        2.04e-12,
        0.001e-12,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=1.0,
            minimum_relative_improvement=0.0,
            minimum_absolute_improvement_a=0.05e-12,
        ),
    )

    assert (
        result.decision
        == ComparisonDecision.INDISTINGUISHABLE
    )


def test_candidate_above_noise_when_baseline_is_below_noise_can_win():
    baseline = measurement(
        0.2e-12,
        0.02e-12,
        below_noise_floor=True,
    )

    candidate = measurement(
        2.0e-12,
        0.05e-12,
        below_noise_floor=False,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert result.decision == ComparisonDecision.BETTER


def test_candidate_below_noise_can_be_worse():
    baseline = measurement(
        2.0e-12,
        0.05e-12,
        below_noise_floor=False,
    )

    candidate = measurement(
        0.2e-12,
        0.02e-12,
        below_noise_floor=True,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert result.decision == ComparisonDecision.WORSE


def test_two_below_noise_measurements_are_not_ranked():
    baseline = measurement(
        0.1e-12,
        0.01e-12,
        below_noise_floor=True,
    )

    candidate = measurement(
        0.2e-12,
        0.01e-12,
        below_noise_floor=True,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert (
        result.decision
        == ComparisonDecision.INDISTINGUISHABLE
    )


def test_combined_sem_is_used():
    baseline = measurement(
        8.0e-9,
        3.0e-11,
    )

    candidate = measurement(
        8.5e-9,
        4.0e-11,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert result.combined_sem_a == pytest.approx(
        5.0e-11
    )

    assert result.uncertainty_margin_a == pytest.approx(
        1.0e-10
    )


def test_uncertainty_score_is_delta_over_combined_sem():
    baseline = measurement(
        8.0e-9,
        1.0e-10,
    )

    candidate = measurement(
        8.4e-9,
        1.0e-10,
    )

    result = compare_measurements(
        baseline,
        candidate,
        ComparisonPolicy(
            minimum_relative_improvement=0.0,
        ),
    )

    expected = (
        0.4e-9
        / (2 ** 0.5 * 1.0e-10)
    )

    assert result.uncertainty_score == pytest.approx(
        expected
    )


def test_invalid_comparison_policy_is_rejected():
    with pytest.raises(ValueError):
        ComparisonPolicy(
            uncertainty_multiple=-1.0
        )

    with pytest.raises(ValueError):
        ComparisonPolicy(
            minimum_relative_improvement=-0.01
        )

    with pytest.raises(ValueError):
        ComparisonPolicy(
            minimum_absolute_improvement_a=-1e-12
        )