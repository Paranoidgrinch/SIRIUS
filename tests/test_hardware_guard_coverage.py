import pytest

from sirius.hardware_guard import (
    HardwareGuardPolicy,
    HardwareSafetyViolation,
    ParameterSafetyRule,
    audit_hardware_guard_policy,
    build_strict_hardware_guard,
    require_complete_hardware_guard,
    required_guard_parameters,
)


def complete_steps(
    value=100.0,
):
    return {
        name: float(
            value
        )
        for name
        in required_guard_parameters()
    }


def test_required_guard_parameters_are_nonempty():
    required = (
        required_guard_parameters()
    )

    assert required


def test_disabled_hv2_hv3_are_not_required():
    required = set(
        required_guard_parameters()
    )

    assert (
        "hv2_voltage_v"
        not in required
    )

    assert (
        "hv3_voltage_v"
        not in required
    )


def test_known_optimizable_transport_channels_are_required():
    required = set(
        required_guard_parameters()
    )

    expected = {
        "sputter_voltage_v",
        "extraction_voltage_v",
        "einzel_lens_voltage_v",
        "magnet_current_a",
        "lens2_voltage_v",
        "steerer_x1_v",
        "steerer_y1_v",
        "ion_cooler_voltage_v",
        "deceleration_voltage_v",
        "acceleration_voltage_v",
        "guidefield1_voltage_v",
        "guidefield2_voltage_v",
        "quadrupole1_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole3_voltage_v",
        "steerer_x2_v",
        "steerer_y2_v",
        "esa_voltage_v",
        "steerer_x3_v",
        "steerer_y3_v",
        "lens4_voltage_v",
    }

    assert expected.issubset(
        required
    )


def test_complete_strict_guard_passes_audit():
    policy = (
        build_strict_hardware_guard(
            complete_steps()
        )
    )

    audit = (
        audit_hardware_guard_policy(
            policy
        )
    )

    assert audit.ready is True

    assert (
        audit.missing_parameters
        == ()
    )

    assert (
        audit.weak_readback_parameters
        == ()
    )

    assert (
        audit.weak_settling_parameters
        == ()
    )


def test_missing_single_parameter_fails_closed():
    steps = complete_steps()

    missing = next(
        iter(
            steps
        )
    )

    del steps[
        missing
    ]

    with pytest.raises(
        HardwareSafetyViolation,
        match="Explicit max_step missing",
    ):
        build_strict_hardware_guard(
            steps
        )


def test_manual_incomplete_policy_is_detected():
    required = (
        required_guard_parameters()
    )

    policy = HardwareGuardPolicy(
        parameter_rules={
            required[
                0
            ]:
                ParameterSafetyRule(
                    max_step=100.0
                )
        }
    )

    audit = (
        audit_hardware_guard_policy(
            policy
        )
    )

    assert audit.ready is False

    assert audit.missing_parameters


def test_readback_cannot_be_disabled_for_real_machine_coverage():
    steps = complete_steps()

    policy = (
        build_strict_hardware_guard(
            steps
        )
    )

    rules = dict(
        policy.parameter_rules
    )

    name = next(
        iter(
            rules
        )
    )

    rules[
        name
    ] = ParameterSafetyRule(
        max_step=100.0,
        require_readback=False,
        require_settling=False,
    )

    weak = HardwareGuardPolicy(
        parameter_rules=rules
    )

    audit = (
        audit_hardware_guard_policy(
            weak
        )
    )

    assert audit.ready is False

    assert name in (
        audit.weak_readback_parameters
    )

    with pytest.raises(
        HardwareSafetyViolation,
        match="readback not required",
    ):
        require_complete_hardware_guard(
            weak
        )


def test_settling_cannot_be_disabled_for_real_machine_coverage():
    steps = complete_steps()

    policy = (
        build_strict_hardware_guard(
            steps
        )
    )

    rules = dict(
        policy.parameter_rules
    )

    name = next(
        iter(
            rules
        )
    )

    rules[
        name
    ] = ParameterSafetyRule(
        max_step=100.0,
        require_readback=True,
        require_settling=False,
    )

    weak = HardwareGuardPolicy(
        parameter_rules=rules
    )

    audit = (
        audit_hardware_guard_policy(
            weak
        )
    )

    assert audit.ready is False

    assert name in (
        audit.weak_settling_parameters
    )


def test_unknown_step_configuration_parameter_is_rejected():
    steps = complete_steps()

    steps[
        "definitely_not_a_real_parameter"
    ] = 100.0

    with pytest.raises(
        HardwareSafetyViolation,
        match="Unknown parameters",
    ):
        build_strict_hardware_guard(
            steps
        )


def test_every_strict_rule_has_positive_explicit_step():
    steps = {
        name:
            float(index + 1)
        for index, name
        in enumerate(
            required_guard_parameters()
        )
    }

    policy = (
        build_strict_hardware_guard(
            steps
        )
    )

    for name, expected in (
        steps.items()
    ):
        assert (
            policy.parameter_rules[
                name
            ].max_step
            == pytest.approx(
                expected
            )
        )