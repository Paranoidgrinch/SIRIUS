import json

import pytest

from sirius.coupled_transition import (
    CoupledTransitionPolicy,
)
from sirius.cup3_coordinates import (
    EndElectrodeCoordinatePolicy,
)
from sirius.cup_ack import (
    CupSelectionPolicy,
)
from sirius.hardware_safety import (
    HardwareSafetyConfig,
)
from sirius.qpt_scan2d import (
    QPT2DScanPolicy,
)


def safety():
    return HardwareSafetyConfig(
        cooler_end_max_step_v=100.0,
        qpt_max_step_v=125.0,
    )


def test_step_sizes_are_explicit_and_positive():
    with pytest.raises(
        ValueError
    ):
        HardwareSafetyConfig(
            cooler_end_max_step_v=0.0,
            qpt_max_step_v=100.0,
        )

    with pytest.raises(
        ValueError
    ):
        HardwareSafetyConfig(
            cooler_end_max_step_v=100.0,
            qpt_max_step_v=float("nan"),
        )


def test_cooler_end_policy_is_constructed_correctly():
    config = safety()

    policy = (
        config.cooler_end_transition_policy()
    )

    assert isinstance(
        policy,
        CoupledTransitionPolicy,
    )

    assert set(
        policy.parameter_order
    ) == {
        "deceleration_voltage_v",
        "acceleration_voltage_v",
    }

    assert (
        policy.max_step_by_parameter[
            "deceleration_voltage_v"
        ]
        == pytest.approx(
            100.0
        )
    )

    assert (
        policy.max_step_by_parameter[
            "acceleration_voltage_v"
        ]
        == pytest.approx(
            100.0
        )
    )


def test_qpt_policy_is_constructed_correctly():
    config = safety()

    policy = (
        config.qpt_transition_policy()
    )

    assert set(
        policy.parameter_order
    ) == {
        "quadrupole1_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole3_voltage_v",
    }

    for parameter_name in (
        policy.parameter_order
    ):
        assert (
            policy.max_step_by_parameter[
                parameter_name
            ]
            == pytest.approx(
                125.0
            )
        )


def test_custom_cooler_command_order_is_preserved():
    config = HardwareSafetyConfig(
        cooler_end_max_step_v=100.0,
        qpt_max_step_v=100.0,
        cooler_end_parameter_order=(
            "acceleration_voltage_v",
            "deceleration_voltage_v",
        ),
    )

    policy = (
        config.cooler_end_transition_policy()
    )

    assert policy.parameter_order == (
        "acceleration_voltage_v",
        "deceleration_voltage_v",
    )


def test_custom_qpt_command_order_is_preserved():
    config = HardwareSafetyConfig(
        cooler_end_max_step_v=100.0,
        qpt_max_step_v=100.0,
        qpt_parameter_order=(
            "quadrupole3_voltage_v",
            "quadrupole2_voltage_v",
            "quadrupole1_voltage_v",
        ),
    )

    policy = (
        config.qpt_transition_policy()
    )

    assert policy.parameter_order == (
        "quadrupole3_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole1_voltage_v",
    )


def test_invalid_cooler_order_is_rejected():
    with pytest.raises(
        ValueError
    ):
        HardwareSafetyConfig(
            cooler_end_max_step_v=100.0,
            qpt_max_step_v=100.0,
            cooler_end_parameter_order=(
                "deceleration_voltage_v",
                "deceleration_voltage_v",
            ),
        )


def test_invalid_qpt_order_is_rejected():
    with pytest.raises(
        ValueError
    ):
        HardwareSafetyConfig(
            cooler_end_max_step_v=100.0,
            qpt_max_step_v=100.0,
            qpt_parameter_order=(
                "quadrupole1_voltage_v",
                "quadrupole2_voltage_v",
            ),
        )


def test_end_electrode_policy_is_bound_without_mutating_original():
    config = safety()

    original = (
        EndElectrodeCoordinatePolicy()
    )

    assert (
        original.transition_policy
        is None
    )

    bound = (
        config.bind_end_electrode_policy(
            original
        )
    )

    assert (
        original.transition_policy
        is None
    )

    assert (
        bound.transition_policy
        is not None
    )

    assert set(
        bound.transition_policy.parameter_order
    ) == {
        "deceleration_voltage_v",
        "acceleration_voltage_v",
    }


def test_qpt_policy_is_bound_without_mutating_original():
    config = safety()

    original = (
        QPT2DScanPolicy()
    )

    assert (
        original.transition_policy
        is None
    )

    bound = (
        config.bind_qpt_scan_policy(
            original
        )
    )

    assert (
        original.transition_policy
        is None
    )

    assert (
        bound.transition_policy
        is not None
    )

    assert set(
        bound.transition_policy.parameter_order
    ) == {
        "quadrupole1_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole3_voltage_v",
    }


def test_cup_ack_policy_is_centralized():
    cup_policy = CupSelectionPolicy(
        timeout_s=7.0,
        poll_interval_s=0.05,
        minimum_wait_s=0.4,
        consecutive_confirmations=3,
    )

    config = HardwareSafetyConfig(
        cooler_end_max_step_v=100.0,
        qpt_max_step_v=100.0,
        cup_selection_policy=(
            cup_policy
        ),
    )

    assert (
        config.transition_apply_kwargs()[
            "cup_selection_policy"
        ]
        is cup_policy
    )


def test_manifest_representation_is_json_serializable():
    config = HardwareSafetyConfig(
        cooler_end_max_step_v=100.0,
        qpt_max_step_v=125.0,
        cup_selection_policy=(
            CupSelectionPolicy(
                timeout_s=7.0,
                poll_interval_s=0.05,
                minimum_wait_s=0.4,
                consecutive_confirmations=3,
            )
        ),
    )

    manifest = (
        config.to_manifest_dict()
    )

    encoded = json.dumps(
        manifest
    )

    assert isinstance(
        encoded,
        str,
    )

    assert (
        manifest[
            "cooler_end_max_step_v"
        ]
        == pytest.approx(
            100.0
        )
    )

    assert (
        manifest[
            "qpt_max_step_v"
        ]
        == pytest.approx(
            125.0
        )
    )

    assert (
        manifest[
            "cup_selection"
        ][
            "consecutive_confirmations"
        ]
        == 3
    )