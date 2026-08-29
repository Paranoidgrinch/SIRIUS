import json

import pytest

from sirius.cologneams_safety import (
    COLOGNEAMS_COMMISSIONING_MAX_STEPS,
    COLOGNEAMS_COMMISSIONING_PROFILE_NAME,
    build_cologneams_commissioning_guard,
    build_cologneams_hardware_safety,
    cologneams_commissioning_max_steps,
    cologneams_safety_manifest,
)
from sirius.cup_ack import (
    CupSelectionPolicy,
)
from sirius.hardware_guard import (
    require_complete_hardware_guard,
    required_guard_parameters,
)


def test_commissioning_profile_exactly_covers_required_parameters():
    required = set(
        required_guard_parameters()
    )

    configured = set(
        COLOGNEAMS_COMMISSIONING_MAX_STEPS
    )

    assert configured == required


def test_profile_contains_no_disabled_hv2_hv3():
    assert (
        "hv2_voltage_v"
        not in
        COLOGNEAMS_COMMISSIONING_MAX_STEPS
    )

    assert (
        "hv3_voltage_v"
        not in
        COLOGNEAMS_COMMISSIONING_MAX_STEPS
    )


def test_commissioning_guard_is_complete():
    guard = (
        build_cologneams_commissioning_guard()
    )

    coverage = (
        require_complete_hardware_guard(
            guard
        )
    )

    assert coverage.ready is True


def test_all_initial_steps_are_positive():
    assert all(
        float(
            value
        ) > 0
        for value
        in COLOGNEAMS_COMMISSIONING_MAX_STEPS.values()
    )


def test_general_hv_guard_matches_coupled_hv_limit():
    safety = (
        build_cologneams_hardware_safety()
    )

    guard = (
        safety.hardware_guard_policy
    )

    hv1 = (
        guard.parameter_rules[
            "deceleration_voltage_v"
        ].max_step
    )

    hv4 = (
        guard.parameter_rules[
            "acceleration_voltage_v"
        ].max_step
    )

    coupled = (
        safety.cooler_end_transition_policy()
    )

    assert (
        coupled.max_step_by_parameter[
            "deceleration_voltage_v"
        ]
        == pytest.approx(
            hv1
        )
    )

    assert (
        coupled.max_step_by_parameter[
            "acceleration_voltage_v"
        ]
        == pytest.approx(
            hv4
        )
    )


def test_general_qpt_guard_matches_coupled_qpt_limit():
    safety = (
        build_cologneams_hardware_safety()
    )

    guard = (
        safety.hardware_guard_policy
    )

    coupled = (
        safety.qpt_transition_policy()
    )

    for name in (
        "quadrupole1_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole3_voltage_v",
    ):
        assert (
            coupled.max_step_by_parameter[
                name
            ]
            <= (
                guard.parameter_rules[
                    name
                ].max_step
                + 1e-12
            )
        )


def test_override_propagates_to_general_guard():
    safety = (
        build_cologneams_hardware_safety(
            max_step_overrides={
                "lens4_voltage_v":
                    20.0,
            }
        )
    )

    assert (
        safety.hardware_guard_policy
        .parameter_rules[
            "lens4_voltage_v"
        ]
        .max_step
        == pytest.approx(
            20.0
        )
    )


def test_hv_override_propagates_into_coupled_transition():
    safety = (
        build_cologneams_hardware_safety(
            max_step_overrides={
                "deceleration_voltage_v":
                    20.0,

                "acceleration_voltage_v":
                    30.0,
            }
        )
    )

    coupled = (
        safety.cooler_end_transition_policy()
    )

    # Coupled path takes the stricter of the two physical-channel limits.
    assert (
        coupled.max_step_by_parameter[
            "deceleration_voltage_v"
        ]
        == pytest.approx(
            20.0
        )
    )

    assert (
        coupled.max_step_by_parameter[
            "acceleration_voltage_v"
        ]
        == pytest.approx(
            20.0
        )
    )


def test_qpt_override_uses_strictest_physical_channel():
    safety = (
        build_cologneams_hardware_safety(
            max_step_overrides={
                "quadrupole1_voltage_v":
                    30.0,

                "quadrupole2_voltage_v":
                    20.0,

                "quadrupole3_voltage_v":
                    40.0,
            }
        )
    )

    coupled = (
        safety.qpt_transition_policy()
    )

    for name in (
        coupled.parameter_order
    ):
        assert (
            coupled.max_step_by_parameter[
                name
            ]
            == pytest.approx(
                20.0
            )
        )


def test_cup_ack_policy_can_be_explicitly_replaced():
    cup = CupSelectionPolicy(
        timeout_s=12.0,
        poll_interval_s=0.1,
        minimum_wait_s=0.5,
        consecutive_confirmations=3,
    )

    safety = (
        build_cologneams_hardware_safety(
            cup_selection_policy=cup
        )
    )

    assert (
        safety.cup_selection_policy
        is cup
    )


def test_profile_copy_cannot_mutate_canonical_profile():
    copy = (
        cologneams_commissioning_max_steps()
    )

    copy[
        "lens4_voltage_v"
    ] = 999.0

    assert (
        COLOGNEAMS_COMMISSIONING_MAX_STEPS[
            "lens4_voltage_v"
        ]
        != 999.0
    )


def test_manifest_records_provisional_profile_identity():
    safety = (
        build_cologneams_hardware_safety()
    )

    manifest = (
        cologneams_safety_manifest(
            safety
        )
    )

    assert (
        manifest[
            "profile_name"
        ]
        == COLOGNEAMS_COMMISSIONING_PROFILE_NAME
    )

    assert (
        manifest[
            "profile_status"
        ]
        == "provisional"
    )

    assert isinstance(
        json.dumps(
            manifest
        ),
        str,
    )