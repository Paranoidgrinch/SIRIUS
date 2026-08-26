from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
)
from sirius.reference import TransmissionResult


@dataclass(frozen=True)
class ScalarEstimate:
    """
    Generic scalar estimate with a standard error.

    Used for optimization objectives that are not raw beam current,
    particularly normalized transmission.
    """

    value: float
    sem: float

    below_noise_floor: bool = False

    def validate(self) -> None:
        if not math.isfinite(
            float(self.value)
        ):
            raise ValueError(
                "Estimate value must be finite"
            )

        if not math.isfinite(
            float(self.sem)
        ):
            raise ValueError(
                "Estimate SEM must be finite"
            )

        if self.sem < 0:
            raise ValueError(
                "Estimate SEM must be non-negative"
            )


@dataclass(frozen=True)
class ScalarComparison:
    baseline_value: float
    candidate_value: float

    delta: float
    relative_delta: float | None

    combined_sem: float

    uncertainty_margin: float
    practical_margin: float
    required_margin: float

    decision: ComparisonDecision

    uncertainty_score: float | None


def estimate_from_transmission(
    transmission: TransmissionResult,
) -> ScalarEstimate:
    estimate = ScalarEstimate(
        value=float(
            transmission.transmission
        ),
        sem=float(
            transmission.transmission_sem
        ),
        below_noise_floor=False,
    )

    estimate.validate()

    return estimate


def compare_estimates(
    baseline: ScalarEstimate,
    candidate: ScalarEstimate,
    policy: ComparisonPolicy,
) -> ScalarComparison:
    baseline.validate()
    candidate.validate()

    baseline_value = float(
        baseline.value
    )

    candidate_value = float(
        candidate.value
    )

    delta = (
        candidate_value
        - baseline_value
    )

    combined_sem = math.sqrt(
        baseline.sem ** 2
        + candidate.sem ** 2
    )

    uncertainty_margin = (
        policy.uncertainty_multiple
        * combined_sem
    )

    relative_margin = (
        abs(baseline_value)
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
        decision = (
            ComparisonDecision.INDISTINGUISHABLE
        )

    elif (
        baseline.below_noise_floor
        and not candidate.below_noise_floor
        and delta > required_margin
    ):
        decision = (
            ComparisonDecision.BETTER
        )

    elif (
        not baseline.below_noise_floor
        and candidate.below_noise_floor
        and -delta > required_margin
    ):
        decision = (
            ComparisonDecision.WORSE
        )

    elif delta > required_margin:
        decision = (
            ComparisonDecision.BETTER
        )

    elif delta < -required_margin:
        decision = (
            ComparisonDecision.WORSE
        )

    else:
        decision = (
            ComparisonDecision.INDISTINGUISHABLE
        )

    if baseline_value == 0:
        relative_delta = None

    else:
        relative_delta = (
            delta
            / abs(baseline_value)
        )

    if combined_sem > 0:
        uncertainty_score = (
            delta
            / combined_sem
        )

    else:
        uncertainty_score = None

    return ScalarComparison(
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        relative_delta=relative_delta,
        combined_sem=combined_sem,
        uncertainty_margin=(
            uncertainty_margin
        ),
        practical_margin=(
            practical_margin
        ),
        required_margin=(
            required_margin
        ),
        decision=decision,
        uncertainty_score=(
            uncertainty_score
        ),
    )