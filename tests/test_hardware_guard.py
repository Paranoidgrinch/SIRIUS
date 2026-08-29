import math

import pytest

from sirius.hardware_guard import (
    HardwareGuardPolicy,
    HardwareSafetyViolation,
    HardwareTransitionFailure,
    ParameterSafetyRule,
    execute_guarded_transition,
    plan_guarded_transition,
)
from sirius.state import MachineState


def state(
    *,
    sputter=8000.0,
    extraction=19600.0,
    lens=18000.0,
):
    return MachineState(
        mass_u=60.0,
        cup=2,
        stage=2,
        parameters={
            "sputter_voltage_v":
                sputter,
            "extraction_voltage_v":
                extraction,
            "einzel_lens_voltage_v":
                lens,
        },
    )


def policy():
    return HardwareGuardPolicy(
        parameter_rules={
            "sputter_voltage_v":
                ParameterSafetyRule(
                    max_step=250.0
                ),

            "extraction_voltage_v":
                ParameterSafetyRule(
                    max_step=500.0
                ),

            "einzel_lens_voltage_v":
                ParameterSafetyRule(
                    max_step=200.0
                ),
        }
    )


def test_large_requested_jump_is_split_into_safe_steps():
    current = state(
        lens=10000.0
    )

    target = state(
        lens=12000.0
    )

    plan = plan_guarded_transition(
        current,
        target,
        policy(),
    )

    lens_steps = [
        step
        for step in plan.steps
        if step.parameter_name
        == "einzel_lens_voltage_v"
    ]

    assert len(
        lens_steps
    ) == 10

    assert all(
        abs(
            step.delta
        )
        <= 200.0
        + 1e-12
        for step
        in lens_steps
    )

    assert (
        lens_steps[
            -1
        ].command_after
        == pytest.approx(
            12000.0
        )
    )


def test_ten_kv_request_never_becomes_ten_kv_command():
    current = state(
        extraction=10000.0
    )

    target = state(
        extraction=20000.0
    )

    plan = plan_guarded_transition(
        current,
        target,
        policy(),
    )

    assert len(
        plan.steps
    ) == 20

    assert max(
        abs(
            step.delta
        )
        for step
        in plan.steps
    ) <= 500.0


def test_exact_final_target_is_preserved():
    current = state(
        lens=10000.0
    )

    target = state(
        lens=10333.0
    )

    plan = plan_guarded_transition(
        current,
        target,
        policy(),
    )

    assert [
        step.command_after
        for step
        in plan.steps
    ][
        -1
    ] == pytest.approx(
        10333.0
    )

    assert all(
        abs(
            step.delta
        )
        <= 200.0
        + 1e-12
        for step
        in plan.steps
    )


def test_unconfigured_changed_parameter_is_rejected():
    current = state()

    target = state(
        sputter=8100.0
    )

    incomplete = HardwareGuardPolicy(
        parameter_rules={
            "extraction_voltage_v":
                ParameterSafetyRule(
                    max_step=500.0
                ),

            "einzel_lens_voltage_v":
                ParameterSafetyRule(
                    max_step=200.0
                ),
        }
    )

    with pytest.raises(
        HardwareSafetyViolation,
        match="No hardware-safety rule",
    ):
        plan_guarded_transition(
            current,
            target,
            incomplete,
        )


def test_hard_parameter_bounds_are_checked_before_execution():
    current = state()

    parameters = dict(
        current.parameters
    )

    parameters[
        "sputter_voltage_v"
    ] = 9500.0

    # Construct without calling validate manually; planner must reject the
    # endpoint itself.
    target = MachineState(
        mass_u=current.mass_u,
        cup=current.cup,
        stage=current.stage,
        parameters=parameters,
    )

    with pytest.raises(
        (
            HardwareSafetyViolation,
            ValueError,
        )
    ):
        plan_guarded_transition(
            current,
            target,
            policy(),
        )


def test_parameter_transition_cannot_hide_cup_change():
    current = state()

    target = MachineState(
        mass_u=current.mass_u,
        cup=3,
        stage=3,
        parameters=dict(
            current.parameters
        ),
    )

    with pytest.raises(
        HardwareSafetyViolation,
        match="Faraday-cup",
    ):
        plan_guarded_transition(
            current,
            target,
            policy(),
        )


def test_next_command_waits_for_previous_executor_return():
    current = state(
        lens=10000.0
    )

    target = state(
        lens=10600.0
    )

    safety = policy()

    plan = plan_guarded_transition(
        current,
        target,
        safety,
    )

    events = []

    def executor(
        step,
        rule,
    ):
        events.append(
            (
                "command",
                step.step_index,
            )
        )

        events.append(
            (
                "readback",
                step.step_index,
            )
        )

        events.append(
            (
                "settled",
                step.step_index,
            )
        )

        return step.target_state

    execute_guarded_transition(
        plan,
        safety,
        executor,
    )

    assert events == [
        (
            "command",
            1,
        ),
        (
            "readback",
            1,
        ),
        (
            "settled",
            1,
        ),
        (
            "command",
            2,
        ),
        (
            "readback",
            2,
        ),
        (
            "settled",
            2,
        ),
        (
            "command",
            3,
        ),
        (
            "readback",
            3,
        ),
        (
            "settled",
            3,
        ),
    ]


def test_readback_timeout_aborts_before_next_command():
    current = state(
        lens=10000.0
    )

    target = state(
        lens=10600.0
    )

    safety = policy()

    plan = plan_guarded_transition(
        current,
        target,
        safety,
    )

    issued = []

    def executor(
        step,
        rule,
    ):
        issued.append(
            step.step_index
        )

        if step.step_index == 2:
            raise TimeoutError(
                "readback did not settle"
            )

        return step.target_state

    with pytest.raises(
        HardwareTransitionFailure
    ) as captured:
        execute_guarded_transition(
            plan,
            safety,
            executor,
        )

    error = (
        captured.value
    )

    assert issued == [
        1,
        2,
    ]

    # Step 3 must never be sent.
    assert error.completed_steps == 1

    assert (
        error.last_state.parameters[
            "einzel_lens_voltage_v"
        ]
        == pytest.approx(
            10200.0
        )
    )


def test_invalid_readback_failure_stops_transition():
    current = state(
        lens=10000.0
    )

    target = state(
        lens=10400.0
    )

    safety = policy()

    plan = plan_guarded_transition(
        current,
        target,
        safety,
    )

    calls = []

    def executor(
        step,
        rule,
    ):
        calls.append(
            step.step_index
        )

        if step.step_index == 1:
            raise ValueError(
                "non-finite readback"
            )

        return step.target_state

    with pytest.raises(
        HardwareTransitionFailure
    ) as captured:
        execute_guarded_transition(
            plan,
            safety,
            executor,
        )

    assert calls == [
        1
    ]

    assert (
        captured.value.completed_steps
        == 0
    )


def test_wrong_completed_command_state_is_rejected():
    current = state(
        lens=10000.0
    )

    target = state(
        lens=10200.0
    )

    safety = policy()

    plan = plan_guarded_transition(
        current,
        target,
        safety,
    )

    def bad_executor(
        step,
        rule,
    ):
        # Pretend to have completed the step while returning the old
        # command state.
        return step.state_before

    with pytest.raises(
        HardwareTransitionFailure
    ):
        execute_guarded_transition(
            plan,
            safety,
            bad_executor,
        )


def test_multiple_parameters_are_never_changed_in_one_guard_step():
    current = state(
        sputter=8000.0,
        lens=18000.0,
    )

    target = state(
        sputter=8500.0,
        lens=18400.0,
    )

    plan = plan_guarded_transition(
        current,
        target,
        policy(),
    )

    for step in plan.steps:
        changed = [
            name
            for name
            in step.state_before.parameters
            if not math.isclose(
                step.state_before.parameters[
                    name
                ],
                step.target_state.parameters[
                    name
                ],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

        assert changed == [
            step.parameter_name
        ]


def test_too_many_microsteps_are_rejected():
    current = state(
        lens=10000.0
    )

    target = state(
        lens=12000.0
    )

    limited = HardwareGuardPolicy(
        parameter_rules={
            "sputter_voltage_v":
                ParameterSafetyRule(
                    max_step=250.0
                ),

            "extraction_voltage_v":
                ParameterSafetyRule(
                    max_step=500.0
                ),

            "einzel_lens_voltage_v":
                ParameterSafetyRule(
                    max_step=10.0
                ),
        },
        max_total_steps=50,
    )

    with pytest.raises(
        HardwareSafetyViolation,
        match="more than",
    ):
        plan_guarded_transition(
            current,
            target,
            limited,
        )


def test_settling_requires_readback():
    with pytest.raises(
        ValueError,
        match="Settling",
    ):
        ParameterSafetyRule(
            max_step=100.0,
            require_readback=False,
            require_settling=True,
        )