from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from sirius.mass_profile import MassProfile
from sirius.measurement import BeamMeasurement
from sirius.parameters import PARAMETERS
from sirius.reference import TransmissionResult


class ObjectiveKind(str, Enum):
    CURRENT_A = "current_a"
    TRANSMISSION = "transmission"


class EvidenceClass(str, Enum):
    ACTIVE = "active"
    DEAD = "dead"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class RangeEvidencePoint:
    """
    One experimentally tested command value for range learning.

    objective_value is either:
      - measured beam current in ampere, or
      - transmission as a fraction (0.8 = 80 %).

    The learning algorithm operates relative to the best response in the
    scan, so it can work across the full pA-to-uA current range.
    """

    parameter_name: str
    command_value: float

    objective_value: float
    objective_sem: float

    objective_kind: ObjectiveKind

    below_noise_floor: bool = False

    state_id: str | None = None
    cup: int | None = None

    def validate(self) -> None:
        if self.parameter_name not in PARAMETERS:
            raise ValueError(
                f"Unknown SIRIUS parameter: {self.parameter_name}"
            )

        definition = PARAMETERS[
            self.parameter_name
        ]

        command = float(
            self.command_value
        )

        if not math.isfinite(command):
            raise ValueError(
                "Command value must be finite"
            )

        if not (
            definition.minimum
            <= command
            <= definition.maximum
        ):
            raise ValueError(
                f"{self.parameter_name}={command} outside hard bounds "
                f"{definition.minimum}..{definition.maximum}"
            )

        if not math.isfinite(
            float(self.objective_value)
        ):
            raise ValueError(
                "Objective value must be finite"
            )

        if self.objective_value < 0:
            raise ValueError(
                "Objective value must be non-negative"
            )

        if not math.isfinite(
            float(self.objective_sem)
        ):
            raise ValueError(
                "Objective SEM must be finite"
            )

        if self.objective_sem < 0:
            raise ValueError(
                "Objective SEM must be non-negative"
            )

        if (
            self.cup is not None
            and not 1 <= self.cup <= 6
        ):
            raise ValueError(
                "Cup must be between 1 and 6"
            )


@dataclass(frozen=True)
class RangeLearningPolicy:
    """
    Conservative policy for shrinking an operating range.

    A boundary is only moved if several tested points beyond the useful
    region are confidently classified as dead.
    """

    minimum_points: int = 7
    minimum_active_points: int = 2

    dead_points_per_edge: int = 2

    active_fraction_of_best: float = 0.10

    uncertainty_multiple: float = 2.0

    safety_margin_fraction: float = 0.05
    minimum_safety_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_points < 3:
            raise ValueError(
                "minimum_points must be at least 3"
            )

        if self.minimum_active_points < 1:
            raise ValueError(
                "minimum_active_points must be at least 1"
            )

        if self.dead_points_per_edge < 1:
            raise ValueError(
                "dead_points_per_edge must be at least 1"
            )

        if not (
            0 < self.active_fraction_of_best < 1
        ):
            raise ValueError(
                "active_fraction_of_best must be between 0 and 1"
            )

        if self.uncertainty_multiple < 0:
            raise ValueError(
                "uncertainty_multiple must be non-negative"
            )

        if self.safety_margin_fraction < 0:
            raise ValueError(
                "safety_margin_fraction must be non-negative"
            )

        if self.minimum_safety_margin < 0:
            raise ValueError(
                "minimum_safety_margin must be non-negative"
            )


@dataclass(frozen=True)
class ClassifiedEvidence:
    point: RangeEvidencePoint
    classification: EvidenceClass

    conservative_lower: float
    conservative_upper: float


@dataclass(frozen=True)
class RangeProposal:
    parameter_name: str

    current_minimum: float
    current_maximum: float

    proposed_minimum: float
    proposed_maximum: float

    threshold: float
    best_objective: float

    total_points: int
    active_points: int
    dead_points: int
    uncertain_points: int

    lower_edge_supported: bool
    upper_edge_supported: bool

    recommended: bool
    reason: str

    objective_kind: ObjectiveKind


def evidence_from_measurement(
    parameter_name: str,
    command_value: float,
    measurement: BeamMeasurement,
    *,
    state_id: str | None = None,
    cup: int | None = None,
) -> RangeEvidencePoint:
    point = RangeEvidencePoint(
        parameter_name=parameter_name,
        command_value=float(command_value),
        objective_value=float(
            measurement.mean_a
        ),
        objective_sem=float(
            measurement.sem_a
        ),
        objective_kind=ObjectiveKind.CURRENT_A,
        below_noise_floor=(
            measurement.below_noise_floor
        ),
        state_id=state_id,
        cup=cup,
    )

    point.validate()

    return point


def evidence_from_transmission(
    parameter_name: str,
    command_value: float,
    transmission: TransmissionResult,
    *,
    state_id: str | None = None,
) -> RangeEvidencePoint:
    point = RangeEvidencePoint(
        parameter_name=parameter_name,
        command_value=float(command_value),
        objective_value=float(
            transmission.transmission
        ),
        objective_sem=float(
            transmission.transmission_sem
        ),
        objective_kind=ObjectiveKind.TRANSMISSION,
        below_noise_floor=False,
        state_id=state_id,
        cup=transmission.cup,
    )

    point.validate()

    return point


def classify_evidence(
    point: RangeEvidencePoint,
    *,
    threshold: float,
    uncertainty_multiple: float,
) -> ClassifiedEvidence:
    point.validate()

    lower = max(
        0.0,
        point.objective_value
        - uncertainty_multiple
        * point.objective_sem,
    )

    upper = (
        point.objective_value
        + uncertainty_multiple
        * point.objective_sem
    )

    if point.below_noise_floor:
        classification = (
            EvidenceClass.DEAD
        )

    elif upper < threshold:
        classification = (
            EvidenceClass.DEAD
        )

    elif lower >= threshold:
        classification = (
            EvidenceClass.ACTIVE
        )

    else:
        classification = (
            EvidenceClass.UNCERTAIN
        )

    return ClassifiedEvidence(
        point=point,
        classification=classification,
        conservative_lower=lower,
        conservative_upper=upper,
    )


def propose_learned_range(
    profile: MassProfile,
    parameter_name: str,
    evidence: Iterable[
        RangeEvidencePoint
    ],
    policy: RangeLearningPolicy,
) -> RangeProposal:
    """
    Propose a safer, smaller search interval from experimental evidence.

    The profile is NOT modified by this function.

    A lower or upper edge is only moved if:
      - enough total evidence exists,
      - enough active points exist,
      - at least dead_points_per_edge confidently dead points exist outside
        the active region,
      - no uncertain points lie between that boundary and the active region.

    A safety margin is retained outside the observed active region.
    """

    profile.validate()

    if parameter_name not in PARAMETERS:
        raise KeyError(
            f"Unknown SIRIUS parameter: {parameter_name}"
        )

    points = list(
        evidence
    )

    for point in points:
        point.validate()

        if (
            point.parameter_name
            != parameter_name
        ):
            raise ValueError(
                "All evidence points must describe the same parameter"
            )

    current_minimum, current_maximum = (
        profile.effective_bounds(
            parameter_name
        )
    )

    if not points:
        return RangeProposal(
            parameter_name=parameter_name,
            current_minimum=current_minimum,
            current_maximum=current_maximum,
            proposed_minimum=current_minimum,
            proposed_maximum=current_maximum,
            threshold=0.0,
            best_objective=0.0,
            total_points=0,
            active_points=0,
            dead_points=0,
            uncertain_points=0,
            lower_edge_supported=False,
            upper_edge_supported=False,
            recommended=False,
            reason="no_evidence",
            objective_kind=ObjectiveKind.CURRENT_A,
        )

    objective_kinds = {
        point.objective_kind
        for point in points
    }

    if len(objective_kinds) != 1:
        raise ValueError(
            "Current and transmission evidence must not be mixed "
            "within one range-learning analysis"
        )

    objective_kind = next(
        iter(objective_kinds)
    )

    if len(points) < policy.minimum_points:
        best = max(
            point.objective_value
            for point in points
        )

        return RangeProposal(
            parameter_name=parameter_name,
            current_minimum=current_minimum,
            current_maximum=current_maximum,
            proposed_minimum=current_minimum,
            proposed_maximum=current_maximum,
            threshold=(
                best
                * policy.active_fraction_of_best
            ),
            best_objective=best,
            total_points=len(points),
            active_points=0,
            dead_points=0,
            uncertain_points=0,
            lower_edge_supported=False,
            upper_edge_supported=False,
            recommended=False,
            reason="insufficient_total_evidence",
            objective_kind=objective_kind,
        )

    usable_for_best = [
        point
        for point in points
        if not point.below_noise_floor
    ]

    if not usable_for_best:
        best = 0.0
    else:
        best = max(
            point.objective_value
            for point in usable_for_best
        )

    if best <= 0:
        return RangeProposal(
            parameter_name=parameter_name,
            current_minimum=current_minimum,
            current_maximum=current_maximum,
            proposed_minimum=current_minimum,
            proposed_maximum=current_maximum,
            threshold=0.0,
            best_objective=best,
            total_points=len(points),
            active_points=0,
            dead_points=len(points),
            uncertain_points=0,
            lower_edge_supported=False,
            upper_edge_supported=False,
            recommended=False,
            reason="no_active_signal",
            objective_kind=objective_kind,
        )

    threshold = (
        best
        * policy.active_fraction_of_best
    )

    classified = [
        classify_evidence(
            point,
            threshold=threshold,
            uncertainty_multiple=(
                policy.uncertainty_multiple
            ),
        )
        for point in points
    ]

    active = [
        item
        for item in classified
        if item.classification
        == EvidenceClass.ACTIVE
    ]

    dead = [
        item
        for item in classified
        if item.classification
        == EvidenceClass.DEAD
    ]

    uncertain = [
        item
        for item in classified
        if item.classification
        == EvidenceClass.UNCERTAIN
    ]

    if (
        len(active)
        < policy.minimum_active_points
    ):
        return RangeProposal(
            parameter_name=parameter_name,
            current_minimum=current_minimum,
            current_maximum=current_maximum,
            proposed_minimum=current_minimum,
            proposed_maximum=current_maximum,
            threshold=threshold,
            best_objective=best,
            total_points=len(points),
            active_points=len(active),
            dead_points=len(dead),
            uncertain_points=len(uncertain),
            lower_edge_supported=False,
            upper_edge_supported=False,
            recommended=False,
            reason="insufficient_active_evidence",
            objective_kind=objective_kind,
        )

    first_active = min(
        item.point.command_value
        for item in active
    )

    last_active = max(
        item.point.command_value
        for item in active
    )

    points_below = [
        item
        for item in classified
        if item.point.command_value
        < first_active
    ]

    points_above = [
        item
        for item in classified
        if item.point.command_value
        > last_active
    ]

    dead_below = [
        item
        for item in points_below
        if item.classification
        == EvidenceClass.DEAD
    ]

    dead_above = [
        item
        for item in points_above
        if item.classification
        == EvidenceClass.DEAD
    ]

    uncertain_below = [
        item
        for item in points_below
        if item.classification
        == EvidenceClass.UNCERTAIN
    ]

    uncertain_above = [
        item
        for item in points_above
        if item.classification
        == EvidenceClass.UNCERTAIN
    ]

    lower_supported = (
        len(dead_below)
        >= policy.dead_points_per_edge
        and not uncertain_below
    )

    upper_supported = (
        len(dead_above)
        >= policy.dead_points_per_edge
        and not uncertain_above
    )

    width = (
        current_maximum
        - current_minimum
    )

    safety_margin = max(
        policy.minimum_safety_margin,
        width
        * policy.safety_margin_fraction,
    )

    proposed_minimum = (
        current_minimum
    )

    proposed_maximum = (
        current_maximum
    )

    if lower_supported:
        proposed_minimum = max(
            current_minimum,
            first_active
            - safety_margin,
        )

    if upper_supported:
        proposed_maximum = min(
            current_maximum,
            last_active
            + safety_margin,
        )

    recommended = (
        proposed_minimum
        > current_minimum
        or proposed_maximum
        < current_maximum
    )

    if recommended:
        reason = "supported_range_reduction"
    else:
        reason = "no_supported_edge_reduction"

    return RangeProposal(
        parameter_name=parameter_name,
        current_minimum=current_minimum,
        current_maximum=current_maximum,
        proposed_minimum=proposed_minimum,
        proposed_maximum=proposed_maximum,
        threshold=threshold,
        best_objective=best,
        total_points=len(points),
        active_points=len(active),
        dead_points=len(dead),
        uncertain_points=len(uncertain),
        lower_edge_supported=lower_supported,
        upper_edge_supported=upper_supported,
        recommended=recommended,
        reason=reason,
        objective_kind=objective_kind,
    )


def apply_range_proposal(
    profile: MassProfile,
    proposal: RangeProposal,
) -> None:
    """
    Explicitly apply a recommended proposal to a MassProfile.

    Learning and applying are kept separate so SIRIUS can log the proposal
    before changing persistent search bounds.
    """

    if not proposal.recommended:
        raise ValueError(
            "Cannot apply a range proposal that is not recommended"
        )

    previous = profile.learned_ranges.get(
        proposal.parameter_name
    )

    previous_evidence = (
        0
        if previous is None
        else previous.evidence_points
    )

    profile.set_learned_range(
        proposal.parameter_name,
        proposal.proposed_minimum,
        proposal.proposed_maximum,
        evidence_points=(
            previous_evidence
            + proposal.total_points
        ),
        source="operating_range_evidence",
    )