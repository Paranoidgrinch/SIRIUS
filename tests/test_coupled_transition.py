from types import SimpleNamespace

import pytest

import sirius.coupled_transition as module
from sirius.qpt_model import (
    qpt_coordinates_from_voltages,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState


def state(
    parameters,
):
    return MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        parameters=dict(
            parameters
        ),
    )


def qpt_state(
    q1,
    q2,
    q3,
):
    return state(
        {
            "quadrupole1_voltage_v": q1,
            "quadrupole2_voltage_v": q2,
            "quadrupole3_voltage_v": q3,
        }
    )


def cooler_state(
    hv1,
    hv4,
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


def test_qpt_large_jump_is_split_into_bounded_steps():
    current = qpt_state(
        2000.0,
        3000.0,
        2000.0,
    )

    target = qpt_state(
        1000.0,
        3000.0,
        1500.0,
    )

    policy = module.qpt_transition_policy(
        max_step_v=100.0
    )

    plan = module.plan_coupled_transition(
        current,
        target,
        policy,
    )

    assert plan.step_count == 10

    for macro_step in plan.macro_steps:
        assert (
            abs(
                macro_step.deltas_from_previous[
                    "quadrupole1_voltage_v"
                ]
            )
            <= 100.0 + 1e-9
        )

        assert (
            abs(
                macro_step.deltas_from_previous[
                    "quadrupole3_voltage_v"
                ]
            )
            <= 100.0 + 1e-9
        )


def test_final_macro_state_matches_target_exactly():
    current = qpt_state(
        2000.0,
        3000.0,
        2000.0,
    )

    target = qpt_state(
        1333.0,
        3000.0,
        1777.0,
    )

    plan = module.plan_coupled_transition(
        current,
        target,
        module.qpt_transition_policy(
            max_step_v=100.0
        ),
    )

    final = (
        plan.macro_steps[
            -1
        ].state
    )

    assert (
        final.parameters
        == target.parameters
    )


def test_qpt_common_mode_remains_constant_when_endpoints_match():
    current = qpt_state(
        2000.0,
        3000.0,
        2000.0,
    )

    target = qpt_state(
        1200.0,
        3000.0,
        1600.0,
    )

    plan = module.plan_coupled_transition(
        current,
        target,
        module.qpt_transition_policy(
            max_step_v=100.0
        ),
    )

    for step in plan.macro_steps:
        coordinates = (
            qpt_coordinates_from_voltages(
                step.state.parameters[
                    "quadrupole1_voltage_v"
                ],
                step.state.parameters[
                    "quadrupole2_voltage_v"
                ],
                step.state.parameters[
                    "quadrupole3_voltage_v"
                ],
            )
        )

        assert (
            coordinates.common_v
            == pytest.approx(
                3000.0
            )
        )


def test_cooler_common_and_difference_interpolate_smoothly():
    current = cooler_state(
        1000.0,
        1000.0,
    )

    target = cooler_state(
        3000.0,
        2000.0,
    )

    plan = module.plan_coupled_transition(
        current,
        target,
        module.cooler_end_transition_policy(
            max_step_v=250.0
        ),
    )

    previous_common = 1000.0
    previous_difference = 0.0

    for step in plan.macro_steps:
        hv1 = step.state.parameters[
            "deceleration_voltage_v"
        ]

        hv4 = step.state.parameters[
            "acceleration_voltage_v"
        ]

        common = (
            hv1
            + hv4
        ) / 2.0

        difference = (
            hv1
            - hv4
        )

        assert common >= previous_common
        assert difference >= previous_difference

        previous_common = common
        previous_difference = difference


def test_unapproved_parameter_change_is_rejected():
    current = MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        parameters={
            "quadrupole1_voltage_v": 2000.0,
            "quadrupole2_voltage_v": 3000.0,
            "quadrupole3_voltage_v": 2000.0,
            "esa_voltage_v": 2500.0,
        },
    )

    target = MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        parameters={
            "quadrupole1_voltage_v": 1500.0,
            "quadrupole2_voltage_v": 3000.0,
            "quadrupole3_voltage_v": 1500.0,
            "esa_voltage_v": 2600.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="unapproved",
    ):
        module.plan_coupled_transition(
            current,
            target,
            module.qpt_transition_policy(
                max_step_v=100.0
            ),
        )


def test_each_hardware_channel_is_set_sequentially(
    monkeypatch,
):
    current = qpt_state(
        2000.0,
        3000.0,
        2000.0,
    )

    target = qpt_state(
        1800.0,
        3000.0,
        2200.0,
    )

    calls = []

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        changed = [
            name
            for name
            in target.parameters
            if target.parameters[
                name
            ]
            != current.parameters[
                name
            ]
        ]

        # The coupled executor must expose only one physical command
        # change to apply_state() at a time.
        assert len(
            changed
        ) == 1

        calls.append(
            changed[
                0
            ]
        )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    result = module.apply_coupled_transition(
        object(),
        current,
        target,
        settling(
            "quadrupole1_voltage_v",
            "quadrupole2_voltage_v",
            "quadrupole3_voltage_v",
        ),
        module.qpt_transition_policy(
            max_step_v=100.0
        ),
    )

    # QPT2 is unchanged and therefore produces no real command.
    assert calls == [
        "quadrupole1_voltage_v",
        "quadrupole3_voltage_v",
        "quadrupole1_voltage_v",
        "quadrupole3_voltage_v",
    ]

    assert (
        result.final_state.parameters
        == target.parameters
    )


def test_explicit_parameter_order_is_respected(
    monkeypatch,
):
    current = cooler_state(
        1000.0,
        1000.0,
    )

    target = cooler_state(
        1100.0,
        1100.0,
    )

    calls = []

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        changed = [
            name
            for name
            in target.parameters
            if target.parameters[
                name
            ]
            != current.parameters[
                name
            ]
        ]

        calls.append(
            changed[
                0
            ]
        )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    module.apply_coupled_transition(
        object(),
        current,
        target,
        settling(
            "deceleration_voltage_v",
            "acceleration_voltage_v",
        ),
        module.cooler_end_transition_policy(
            max_step_v=100.0,
            parameter_order=(
                "acceleration_voltage_v",
                "deceleration_voltage_v",
            ),
        ),
    )

    assert calls == [
        "acceleration_voltage_v",
        "deceleration_voltage_v",
    ]


def test_hard_limits_are_preserved_in_every_macro_state():
    current = qpt_state(
        100.0,
        3000.0,
        100.0,
    )

    target = qpt_state(
        5900.0,
        3000.0,
        5900.0,
    )

    plan = module.plan_coupled_transition(
        current,
        target,
        module.qpt_transition_policy(
            max_step_v=250.0
        ),
    )

    for step in plan.macro_steps:
        for parameter_name in (
            "quadrupole1_voltage_v",
            "quadrupole2_voltage_v",
            "quadrupole3_voltage_v",
        ):
            value = step.state.parameters[
                parameter_name
            ]

            assert 0.0 <= value <= 6000.0


def test_failure_preserves_last_observed_state(
    monkeypatch,
):
    current = cooler_state(
        1000.0,
        1000.0,
    )

    target = cooler_state(
        1200.0,
        1200.0,
    )

    calls = {
        "n": 0
    }

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        calls[
            "n"
        ] += 1

        if calls["n"] == 3:
            raise RuntimeError(
                "simulated hardware failure"
            )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    with pytest.raises(
        module.CoupledTransitionError
    ) as captured:
        module.apply_coupled_transition(
            object(),
            current,
            target,
            settling(
                "deceleration_voltage_v",
                "acceleration_voltage_v",
            ),
            module.cooler_end_transition_policy(
                max_step_v=100.0
            ),
        )

    error = captured.value

    assert (
        error.completed_macro_steps
        == 1
    )

    assert (
        error.completed_channel_steps
        == 2
    )

    # First 100-V macro step was fully completed.
    assert (
        error.last_state.parameters[
            "deceleration_voltage_v"
        ]
        == pytest.approx(
            1100.0
        )
    )

    assert (
        error.last_state.parameters[
            "acceleration_voltage_v"
        ]
        == pytest.approx(
            1100.0
        )
    )


def test_invalid_qpt_order_is_rejected():
    with pytest.raises(
        ValueError
    ):
        module.qpt_transition_policy(
            max_step_v=100.0,
            parameter_order=(
                "quadrupole1_voltage_v",
                "quadrupole2_voltage_v",
            ),
        )


def test_invalid_max_step_is_rejected():
    with pytest.raises(
        ValueError
    ):
        module.cooler_end_transition_policy(
            max_step_v=0.0
        )

def test_final_state_restores_requested_target_semantics(
    monkeypatch,
):
    current = qpt_state(
        2000.0,
        3000.0,
        2000.0,
    )

    target = MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        role="scan_candidate",
        parameters={
            "quadrupole1_voltage_v": 1800.0,
            "quadrupole2_voltage_v": 3000.0,
            "quadrupole3_voltage_v": 2200.0,
        },
        metadata={
            "scan_coordinate": "qpt_focus_asymmetry_2d",
            "important": 42,
        },
    )

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    result = module.apply_coupled_transition(
        object(),
        current,
        target,
        settling(
            "quadrupole1_voltage_v",
            "quadrupole2_voltage_v",
            "quadrupole3_voltage_v",
        ),
        module.qpt_transition_policy(
            max_step_v=100.0
        ),
    )

    assert (
        result.final_state.role
        == "scan_candidate"
    )

    assert (
        result.final_state.metadata
        == target.metadata
    )

    assert (
        result.final_state.parameters
        == target.parameters
    )