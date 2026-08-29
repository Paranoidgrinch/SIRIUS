from types import SimpleNamespace

import pytest

import sirius.safe_transition as module
from sirius.command_cadence import (
    CommandCadenceController,
)
from sirius.hardware_guard import (
    HardwareGuardPolicy,
    ParameterSafetyRule,
)
from sirius.readback_freshness import (
    ReadbackFreshnessPolicy,
)
from sirius.state import MachineState


PARAMETER = (
    "einzel_lens_voltage_v"
)


class Clock:
    def __init__(
        self,
    ):
        self.now = 0.0

    def monotonic(
        self,
    ):
        return self.now

    def sleep(
        self,
        seconds,
    ):
        self.now += float(
            seconds
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


def test_microsteps_are_paced_even_with_instant_readbacks(
    monkeypatch,
):
    clock = Clock()

    command_times = []

    adapter = SimpleNamespace(
        hardware_guard_policy=(
            HardwareGuardPolicy(
                parameter_rules={
                    PARAMETER:
                        ParameterSafetyRule(
                            max_step=200.0,
                            minimum_command_interval_s=0.5,
                        )
                }
            )
        ),

        readback_freshness_policy=(
            ReadbackFreshnessPolicy()
        ),

        command_cadence_controller=(
            CommandCadenceController(
                monotonic=(
                    clock.monotonic
                ),
                sleep=(
                    clock.sleep
                ),
            )
        ),

        capture_parameter_readback_freshness_barrier=(
            lambda name:
                10.0
        ),
    )

    def fake_raw(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup=True,
        cup_selection_policy=None,
    ):
        before = float(
            current.parameters[
                PARAMETER
            ]
        )

        after = float(
            target.parameters[
                PARAMETER
            ]
        )

        if before != after:
            command_times.append(
                clock.now
            )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
    )

    monkeypatch.setattr(
        module,
        "wait_for_fresh_parameter_readback",
        lambda *args, **kwargs:
            SimpleNamespace(
                value=1000.0,
                source_timestamp=11.0,
            ),
    )

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

    assert command_times == pytest.approx(
        [
            0.0,
            0.5,
            1.0,
        ]
    )


def test_cadence_persists_across_separate_transitions(
    monkeypatch,
):
    clock = Clock()

    command_times = []

    adapter = SimpleNamespace(
        hardware_guard_policy=(
            HardwareGuardPolicy(
                parameter_rules={
                    PARAMETER:
                        ParameterSafetyRule(
                            max_step=200.0,
                            minimum_command_interval_s=0.5,
                        )
                }
            )
        ),

        readback_freshness_policy=(
            ReadbackFreshnessPolicy()
        ),

        command_cadence_controller=(
            CommandCadenceController(
                monotonic=(
                    clock.monotonic
                ),
                sleep=(
                    clock.sleep
                ),
            )
        ),

        capture_parameter_readback_freshness_barrier=(
            lambda name:
                10.0
        ),
    )

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
            command_times.append(
                clock.now
            )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
    )

    monkeypatch.setattr(
        module,
        "wait_for_fresh_parameter_readback",
        lambda *args, **kwargs:
            SimpleNamespace(
                value=1000.0,
                source_timestamp=11.0,
            ),
    )

    first = state(
        1000.0
    )

    second = state(
        1200.0
    )

    third = state(
        1400.0
    )

    module.apply_state(
        adapter,
        current=first,
        target=second,
        settling_policies={
            PARAMETER:
                object(),
        },
        select_target_cup=False,
    )

    module.apply_state(
        adapter,
        current=second,
        target=third,
        settling_policies={
            PARAMETER:
                object(),
        },
        select_target_cup=False,
    )

    assert command_times == pytest.approx(
        [
            0.0,
            0.5,
        ]
    )