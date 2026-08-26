import pytest

from sirius.mass_profile import (
    MassProfile,
)
from sirius.range_learning import (
    EvidenceClass,
    ObjectiveKind,
    RangeEvidencePoint,
    RangeLearningPolicy,
    apply_range_proposal,
    classify_evidence,
    propose_learned_range,
)


def point(
    command,
    response,
    *,
    sem=0.0,
    below_noise=False,
    kind=ObjectiveKind.CURRENT_A,
):
    return RangeEvidencePoint(
        parameter_name="sputter_voltage_v",
        command_value=command,
        objective_value=response,
        objective_sem=sem,
        objective_kind=kind,
        below_noise_floor=below_noise,
        cup=1,
    )


def policy():
    return RangeLearningPolicy(
        minimum_points=7,
        minimum_active_points=2,
        dead_points_per_edge=2,
        active_fraction_of_best=0.10,
        uncertainty_multiple=2.0,
        safety_margin_fraction=0.05,
    )


def test_evidence_must_respect_hard_bounds():
    evidence = point(
        9500.0,
        1.0,
    )

    with pytest.raises(
        ValueError
    ):
        evidence.validate()


def test_below_noise_point_is_dead():
    evidence = point(
        1000.0,
        1e-13,
        below_noise=True,
    )

    result = classify_evidence(
        evidence,
        threshold=1e-10,
        uncertainty_multiple=2.0,
    )

    assert (
        result.classification
        == EvidenceClass.DEAD
    )


def test_uncertain_measurement_is_not_called_dead():
    evidence = point(
        3000.0,
        0.11,
        sem=0.03,
    )

    result = classify_evidence(
        evidence,
        threshold=0.10,
        uncertainty_multiple=2.0,
    )

    assert (
        result.classification
        == EvidenceClass.UNCERTAIN
    )


def test_insufficient_points_do_not_shrink_range():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(0.0, 0.0),
        point(5000.0, 1.0),
        point(9000.0, 0.8),
    ]

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        evidence,
        policy(),
    )

    assert proposal.recommended is False

    assert proposal.reason == (
        "insufficient_total_evidence"
    )

    assert proposal.proposed_minimum == 0.0
    assert proposal.proposed_maximum == 9000.0


def test_supported_dead_lower_edge_shrinks_lower_bound():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(0.0, 0.001),
        point(1000.0, 0.002),
        point(5000.0, 0.80),
        point(6000.0, 1.00),
        point(7000.0, 0.90),
        point(8000.0, 0.80),
        point(9000.0, 0.70),
    ]

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        evidence,
        policy(),
    )

    assert proposal.recommended is True
    assert proposal.lower_edge_supported is True
    assert proposal.upper_edge_supported is False

    # 5 % safety margin of the 0..9000 V range = 450 V.
    assert proposal.proposed_minimum == pytest.approx(
        4550.0
    )

    assert proposal.proposed_maximum == 9000.0


def test_supported_dead_upper_edge_shrinks_upper_bound():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(0.0, 0.70),
        point(1000.0, 0.80),
        point(2000.0, 0.90),
        point(3000.0, 1.00),
        point(4000.0, 0.80),
        point(8000.0, 0.002),
        point(9000.0, 0.001),
    ]

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        evidence,
        policy(),
    )

    assert proposal.upper_edge_supported is True
    assert proposal.lower_edge_supported is False

    assert proposal.proposed_minimum == 0.0
    assert proposal.proposed_maximum == pytest.approx(
        4450.0
    )


def test_both_supported_edges_can_reduce_range():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(0.0, 0.001),
        point(1000.0, 0.002),
        point(4000.0, 0.80),
        point(5000.0, 1.00),
        point(6000.0, 0.90),
        point(8000.0, 0.002),
        point(9000.0, 0.001),
    ]

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        evidence,
        policy(),
    )

    assert proposal.lower_edge_supported is True
    assert proposal.upper_edge_supported is True

    assert proposal.proposed_minimum == pytest.approx(
        3550.0
    )

    assert proposal.proposed_maximum == pytest.approx(
        6450.0
    )


def test_uncertain_boundary_blocks_range_reduction():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(0.0, 0.001),
        point(1000.0, 0.002),
        point(
            3000.0,
            0.11,
            sem=0.03,
        ),
        point(4000.0, 0.80),
        point(5000.0, 1.00),
        point(6000.0, 0.90),
        point(7000.0, 0.80),
    ]

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        evidence,
        policy(),
    )

    assert proposal.lower_edge_supported is False
    assert proposal.proposed_minimum == 0.0


def test_mixed_current_and_transmission_evidence_is_rejected():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(
            1000.0,
            1e-9,
            kind=ObjectiveKind.CURRENT_A,
        ),
        point(
            2000.0,
            0.5,
            kind=ObjectiveKind.TRANSMISSION,
        ),
        point(3000.0, 1e-9),
        point(4000.0, 1e-9),
        point(5000.0, 1e-9),
        point(6000.0, 1e-9),
        point(7000.0, 1e-9),
    ]

    with pytest.raises(
        ValueError
    ):
        propose_learned_range(
            profile,
            "sputter_voltage_v",
            evidence,
            policy(),
        )


def test_proposal_is_not_applied_implicitly():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(0.0, 0.001),
        point(1000.0, 0.002),
        point(5000.0, 0.80),
        point(6000.0, 1.00),
        point(7000.0, 0.90),
        point(8000.0, 0.80),
        point(9000.0, 0.70),
    ]

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        evidence,
        policy(),
    )

    assert proposal.recommended is True

    assert profile.effective_bounds(
        "sputter_voltage_v"
    ) == (
        0.0,
        9000.0,
    )


def test_recommended_proposal_can_be_applied():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = [
        point(0.0, 0.001),
        point(1000.0, 0.002),
        point(5000.0, 0.80),
        point(6000.0, 1.00),
        point(7000.0, 0.90),
        point(8000.0, 0.80),
        point(9000.0, 0.70),
    ]

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        evidence,
        policy(),
    )

    apply_range_proposal(
        profile,
        proposal,
    )

    assert profile.effective_bounds(
        "sputter_voltage_v"
    ) == pytest.approx(
        (
            4550.0,
            9000.0,
        )
    )

    learned = profile.learned_ranges[
        "sputter_voltage_v"
    ]

    assert learned.evidence_points == 7

    assert learned.source == (
        "operating_range_evidence"
    )


def test_nonrecommended_proposal_cannot_be_applied():
    profile = MassProfile(
        mass_u=60.0
    )

    proposal = propose_learned_range(
        profile,
        "sputter_voltage_v",
        [
            point(1000.0, 0.5),
            point(5000.0, 1.0),
        ],
        policy(),
    )

    with pytest.raises(
        ValueError
    ):
        apply_range_proposal(
            profile,
            proposal,
        )