import pytest

from sirius.coupled_transition import (
    CoupledTransitionPolicy,
    cooler_end_transition_policy,
)
from sirius.cup3_coordinates import (
    EndElectrodeCoordinatePolicy,
)


def test_end_electrode_policy_accepts_hv1_hv4_transition_policy():
    transition = (
        cooler_end_transition_policy(
            max_step_v=100.0
        )
    )

    policy = EndElectrodeCoordinatePolicy(
        transition_policy=transition
    )

    assert (
        policy.transition_policy
        is transition
    )


def test_end_electrode_policy_rejects_wrong_parameter_group():
    wrong = CoupledTransitionPolicy(
        parameter_order=(
            "guidefield1_voltage_v",
            "guidefield2_voltage_v",
        ),
        max_step_by_parameter={
            "guidefield1_voltage_v": 1.0,
            "guidefield2_voltage_v": 1.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="End-electrode",
    ):
        EndElectrodeCoordinatePolicy(
            transition_policy=wrong
        )


def test_end_electrode_policy_keeps_offline_compatibility():
    policy = EndElectrodeCoordinatePolicy()

    assert (
        policy.transition_policy
        is None
    )