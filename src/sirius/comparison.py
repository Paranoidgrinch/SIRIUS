from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from sirius.measurement import BeamMeasurement


class ComparisonDecision(str, Enum):
    BETTER = "better"
    WORSE = "worse"
    INDISTINGUISHABLE = "indistinguishable"


@dataclass(frozen=True)
class ComparisonPolicy:
    """
    Policy for deciding whether one beam-current measurement is meaningfully
    better than another.

    The uncertainty_multiple is deliberately described as a conservative
    decision margin rather than a formal statistical confidence level.
    Keithley samples may not always be perfectly independent.
    """

    uncertainty_multiple: float = 2.0

    minimum_absolute_improvement_a: float = 0.0
    minimum_relative_improvement: float = 0.002

    def __post_init__(self) -> None:
        if self.uncertainty_multiple < 0:
            raise ValueError(
                "uncertainty_multiple must be non-negative"
            )

        if self.minimum_absolute_improvement_a < 0:
            raise ValueError(
                "minimum_absolute_improvement_a must be non-negative"
            )

        if self.minimum_relative_improvement < 0:
            raise ValueError(
                "minimum_relative_improvement must be non-negative"
            )


@dataclass(frozen=True)
class MeasurementComparison:
    baseline_mean_a: float
    candidate_mean_a: float

    delta_a: float
    relative_delta: float | None

    combined_sem_a: float
    uncertainty_margin_a: float
    practical_margin_a: float
    required_margin_a: float

    decision: ComparisonDecision

    uncertainty_score: float | None

    baseline_below_noise_floor: bool
    candidate_below_noise_floor: bool


def combined_sem(
    baseline: BeamMeasurement,
    candidate: BeamMeasurement,
) -> float:
    return math.sqrt(
        baseline.sem_a ** 2
        + candidate.sem_a ** 2
    )


def relative_change(
    baseline_value: float,
    candidate_value: float,
) -> float | None:
    if baseline_value == 0:
        return None

    return (
        candidate_value - baseline_value
    ) / abs(baseline_value)


def compare_measurements(
    baseline: BeamMeasurement,
    candidate: BeamMeasurement,
    policy: ComparisonPolicy,
) -> MeasurementComparison:
    """
    Compare a candidate beam-current measurement against the current baseline.

    BETTER:
        Candidate exceeds the baseline by more than both the uncertainty
        margin and the configured minimum practical improvement.

    WORSE:
        Candidate is lower by more than the same conservative margin.

    INDISTINGUISHABLE:
        Difference is too small to separate reliably from measurement noise
        and/or is smaller than the configured practical improvement threshold.
    """

    baseline_mean = float(baseline.mean_a)
    candidate_mean = float(candidate.mean_a)

    delta = candidate_mean - baseline_mean

    sem_combined = combined_sem(
        baseline,
        candidate,
    )

    uncertainty_margin = (
        policy.uncertainty_multiple
        * sem_combined
    )

    relative_margin = (
        abs(baseline_mean)
        * policy.minimum_relative_improvement
    )

    practical_margin = max(
        policy.minimum_absolute_improvement_a,
        relative_margin,
    )

    required_margin = max(
        uncertainty_margin,
        practical_margin,
    )

    if (
        baseline.below_noise_floor
        and candidate.below_noise_floor
    ):
        decision = ComparisonDecision.INDISTINGUISHABLE

    elif (
        baseline.below_noise_floor
        and not candidate.below_noise_floor
        and delta > required_margin
    ):
        decision = ComparisonDecision.BETTER

    elif (
        not baseline.below_noise_floor
        and candidate.below_noise_floor
        and -delta > required_margin
    ):
        decision = ComparisonDecision.WORSE

    elif delta > required_margin:
        decision = ComparisonDecision.BETTER

    elif delta < -required_margin:
        decision = ComparisonDecision.WORSE

    else:
        decision = ComparisonDecision.INDISTINGUISHABLE

    if sem_combined > 0:
        score = delta / sem_combined
    else:
        score = None

    return MeasurementComparison(
        baseline_mean_a=baseline_mean,
        candidate_mean_a=candidate_mean,
        delta_a=delta,
        relative_delta=relative_change(
            baseline_mean,
            candidate_mean,
        ),
        combined_sem_a=sem_combined,
        uncertainty_margin_a=uncertainty_margin,
        practical_margin_a=practical_margin,
        required_margin_a=required_margin,
        decision=decision,
        uncertainty_score=score,
        baseline_below_noise_floor=(
            baseline.below_noise_floor
        ),
        candidate_below_noise_floor=(
            candidate.below_noise_floor
        ),
    )