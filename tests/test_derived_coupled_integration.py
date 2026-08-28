from types import SimpleNamespace

import pytest

import sirius.derived_scan1d as module
from sirius.coupled_transition import (
    CoupledTransitionPolicy,
    cooler_end_transition_policy,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState


def hv_state(
    hv1=1000.0,
    hv4=1000.0,
):
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "deceleration_voltage_v": hv1,
            "acceleration_voltage_v": hv4,
        },
    )


def guide_state(
    gf1=10.0,
    gf2=20.0,
):
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "guidefield1_voltage_v": gf1,
            "guidefield2_voltage_v": gf2,
        },
    )


def settling(
    *names,
):
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        name: policy
        for name
        in names
    }


def test_apply_derived_target_uses_coupled_executor_when_configured(
    monkeypatch,
):
    current = hv_state(
        1000.0,
        1000.0,
    )

    target = hv_state(
        1200.0,
        1100.0,
    )

    policy = (
        cooler_end_transition_policy(
            max_step_v=100.0
        )
    )

    calls = []

    def fake_coupled(
        adapter,
        current,
        target,
        settling_policies,
        transition_policy,
        *,
        logger=None,
    ):
        calls.append(
            (
                current,
                target,
                transition_policy,
            )
        )

        return SimpleNamespace(
            final_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_coupled_transition",
        fake_coupled,
    )

    def forbidden_direct(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Direct apply_state() must not be used"
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        forbidden_direct,
    )

    result = module._apply_derived_target(
        object(),
        current,
        target,
        settling(
            "deceleration_voltage_v",
            "acceleration_voltage_v",
        ),
        coupled_transition_policy=(
            policy
        ),
    )

    assert result is target

    assert len(
        calls
    ) == 1

    assert (
        calls[
            0
        ][
            2
        ]
        is policy
    )


def test_apply_derived_target_keeps_normal_path_without_policy(
    monkeypatch,
):
    current = guide_state()

    target = guide_state(
        11.0,
        19.0,
    )

    calls = []

    def fake_direct(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        calls.append(
            target
        )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_direct,
    )

    def forbidden_coupled(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Coupled executor must not be used"
        )

    monkeypatch.setattr(
        module,
        "apply_coupled_transition",
        forbidden_coupled,
    )

    result = module._apply_derived_target(
        object(),
        current,
        target,
        settling(
            "guidefield1_voltage_v",
            "guidefield2_voltage_v",
        ),
        coupled_transition_policy=None,
    )

    assert result is target

    assert calls == [
        target
    ]


def test_wrong_coupled_parameter_group_is_rejected_before_scan():
    current = hv_state()

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

    class Tracker:
        latest = None

    with pytest.raises(
        ValueError,
        match="affected parameters",
    ):
        module.scan_derived_coordinate_transmission_1d(
            object(),
            current,
            Tracker(),
            coordinate_name="test",
            minimum=-100.0,
            maximum=100.0,
            coordinate_reader=lambda state: 0.0,
            command_builder=lambda state, value: {
                "deceleration_voltage_v": 1000.0,
                "acceleration_voltage_v": 1000.0,
            },
            affected_parameters=(
                "deceleration_voltage_v",
                "acceleration_voltage_v",
            ),
            scan_policy=SimpleNamespace(
                steps=(
                    10.0,
                ),
                refinement_half_width_factor=1.0,
                max_points_per_level=100,
            ),
            settling_policies=settling(
                "deceleration_voltage_v",
                "acceleration_voltage_v",
            ),
            measurement_policy=SimpleNamespace(),
            comparison_policy=SimpleNamespace(),
            coupled_transition_policy=wrong,
        )


def test_cooler_policy_parameter_group_is_accepted():
    policy = (
        cooler_end_transition_policy(
            max_step_v=100.0
        )
    )

    assert set(
        policy.parameter_order
    ) == {
        "deceleration_voltage_v",
        "acceleration_voltage_v",
    }