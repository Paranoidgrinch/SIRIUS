from types import SimpleNamespace

import pytest

import sirius.safe_transition as module
from sirius.hardware_guard import (
    HardwareGuardPolicy,
    HardwareTransitionFailure,
    ParameterSafetyRule,
)
from sirius.readback_freshness import (
    ReadbackFreshnessPolicy,
    ReadbackFreshnessTimeoutError,
)
from sirius.state import MachineState


PARAMETER = (
    "einzel_lens_voltage_v"
)


def state(
    value,
):
    return MachineState(
        mass_u=60.0,
        cup=2,
        stage=2,
        parameters={
            PARAMETER:
                float(
                    value
                ),
        },
    )


def test_no_next_microstep_without_fresh_post_command_readback(
    monkeypatch,
):
    commands = []

    class Adapter:
        hardware_guard_policy = (
            HardwareGuardPolicy(
                parameter_rules={
                    PARAMETER:
                        ParameterSafetyRule(
                            max_step=200.0
                        )
                }
            )
        )

        readback_freshness_policy = (
            ReadbackFreshnessPolicy(
                timeout_s=1.0,
                poll_interval_s=0.01,
            )
        )

        def capture_parameter_readback_freshness_barrier(
            self,
            name,
        ):
            return 10.0

    adapter = Adapter()

    def fake_raw(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup=True,
        cup_selection_policy=None,
    ):
        before = (
            current.parameters[
                PARAMETER
            ]
        )

        after = (
            target.parameters[
                PARAMETER
            ]
        )

        if before != after:
            commands.append(
                (
                    before,
                    after,
                )
            )

        return SimpleNamespace(
            observed_state=target
        )

    def no_fresh_readback(
        *args,
        **kwargs,
    ):
        raise ReadbackFreshnessTimeoutError(
            "cached forever"
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
    )

    monkeypatch.setattr(
        module,
        "wait_for_fresh_parameter_readback",
        no_fresh_readback,
    )

    with pytest.raises(
        HardwareTransitionFailure
    ):
        module.apply_state(
            adapter,
            current=state(
                1000.0
            ),
            target=state(
                1600.0
            ),
            settling_policies={
                PARAMETER:
                    object(),
            },
            select_target_cup=False,
        )

    # 1000 -> 1200 was issued.
    # 1200 -> 1400 and 1400 -> 1600 MUST NOT be issued.
    assert commands == [
        (
            1000.0,
            1200.0,
        )
    ]


def test_barrier_is_captured_before_command(
    monkeypatch,
):
    events = []

    class Adapter:
        hardware_guard_policy = (
            HardwareGuardPolicy(
                parameter_rules={
                    PARAMETER:
                        ParameterSafetyRule(
                            max_step=200.0
                        )
                }
            )
        )

        readback_freshness_policy = (
            ReadbackFreshnessPolicy()
        )

        def capture_parameter_readback_freshness_barrier(
            self,
            name,
        ):
            events.append(
                "barrier"
            )

            return 10.0

    adapter = Adapter()

    def fake_raw(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup=True,
        cup_selection_policy=None,
    ):
        if (
            current.parameters[
                PARAMETER
            ]
            != target.parameters[
                PARAMETER
            ]
        ):
            events.append(
                "command"
            )

        return SimpleNamespace(
            observed_state=target
        )

    def fake_fresh(
        adapter,
        parameter_name,
        *,
        not_before_source_timestamp,
        policy,
        quality_policy=None,
    ):
        events.append(
            "fresh"
        )

        assert (
            not_before_source_timestamp
            == 10.0
        )

        return SimpleNamespace(
            value=1195.0,
            source_timestamp=11.0,
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
    )

    monkeypatch.setattr(
        module,
        "wait_for_fresh_parameter_readback",
        fake_fresh,
    )

    module.apply_state(
        adapter,
        current=state(
            1000.0
        ),
        target=state(
            1200.0
        ),
        settling_policies={
            PARAMETER:
                object(),
        },
        select_target_cup=False,
    )

    assert events[
        :3
    ] == [
        "barrier",
        "command",
        "fresh",
    ]


def test_verified_readback_is_preserved_in_state(
    monkeypatch,
):
    class Adapter:
        hardware_guard_policy = (
            HardwareGuardPolicy(
                parameter_rules={
                    PARAMETER:
                        ParameterSafetyRule(
                            max_step=200.0
                        )
                }
            )
        )

        readback_freshness_policy = (
            ReadbackFreshnessPolicy()
        )

        def capture_parameter_readback_freshness_barrier(
            self,
            name,
        ):
            return 10.0

    target = state(
        1200.0
    )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        lambda *args, **kwargs:
            SimpleNamespace(
                observed_state=kwargs[
                    "target"
                ]
            ),
    )

    monkeypatch.setattr(
        module,
        "wait_for_fresh_parameter_readback",
        lambda *args, **kwargs:
            SimpleNamespace(
                value=1178.0,
                source_timestamp=11.0,
            ),
    )

    result = module.apply_state(
        Adapter(),
        current=state(
            1000.0
        ),
        target=target,
        settling_policies={
            PARAMETER:
                object(),
        },
        select_target_cup=False,
    )

    assert (
        result.observed_state.parameters[
            PARAMETER
        ]
        == pytest.approx(
            1200.0
        )
    )
