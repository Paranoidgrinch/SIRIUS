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
)
from sirius.state import MachineState


PARAMETER = (
    "einzel_lens_voltage_v"
)


def state(
    value,
    *,
    cup=2,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        stage=2,
        parameters={
            PARAMETER:
                float(value),
        },
    )


def guard(
    max_step=200.0,
):
    return HardwareGuardPolicy(
        parameter_rules={
            PARAMETER:
                ParameterSafetyRule(
                    max_step=max_step,
                    require_readback=True,
                    require_settling=True,
                )
        }
    )


def guarded_adapter(
    max_step=200.0,
):
    """
    Minimal freshness-capable fake adapter.

    The old safe-transition tests exercise command splitting / sequencing,
    not freshness itself. Dedicated freshness fault-injection tests live
    in test_safe_transition_freshness.py.

    Therefore this fake supplies an immediately fresh parameter readback.
    """

    return SimpleNamespace(
        hardware_guard_policy=(
            guard(
                max_step
            )
        ),
        readback_freshness_policy=(
            ReadbackFreshnessPolicy(
                timeout_s=1.0,
                poll_interval_s=0.01,
            )
        ),
        capture_parameter_readback_freshness_barrier=(
            lambda name:
                10.0
        ),
        read_parameter_snapshot=(
            lambda name:
                SimpleNamespace(
                    value=1000.0,
                    timestamp=11.0,
                    quality=None,
                    source=None,
                )
        ),
    )


def test_guarded_apply_splits_large_jump(
    monkeypatch,
):
    calls = []

    adapter = guarded_adapter(
        200.0
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
        calls.append(
            (
                float(
                    current.parameters[
                        PARAMETER
                    ]
                ),
                float(
                    target.parameters[
                        PARAMETER
                    ]
                ),
                bool(
                    select_target_cup
                ),
            )
        )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
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

    assert calls[
        0
    ][
        :2
    ] == (
        1000.0,
        1200.0,
    )

    assert calls[
        1
    ][
        :2
    ] == (
        1200.0,
        1400.0,
    )

    assert calls[
        2
    ][
        :2
    ] == (
        1400.0,
        1600.0,
    )

    # A final no-command normalization call is permitted.
    command_calls = [
        pair
        for pair
        in calls
        if pair[
            0
        ] != pair[
            1
        ]
    ]

    assert len(
        command_calls
    ) == 3

    assert all(
        abs(
            after - before
        )
        <= 200.0
        + 1e-12
        for before, after, _
        in command_calls
    )


def test_no_second_command_after_first_step_failure(
    monkeypatch,
):
    calls = []

    adapter = guarded_adapter(
        200.0
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

        calls.append(
            (
                before,
                after,
            )
        )

        if after == pytest.approx(
            1200.0
        ):
            raise TimeoutError(
                "readback did not settle"
            )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
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

    assert calls == [
        (
            1000.0,
            1200.0,
        )
    ]


def test_parameter_changes_finish_before_cup_change(
    monkeypatch,
):
    events = []

    adapter = guarded_adapter(
        200.0
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
            events.append(
                (
                    "parameter",
                    after,
                )
            )

        if (
            select_target_cup
            and target.cup
            != current.cup
        ):
            events.append(
                (
                    "cup",
                    target.cup,
                )
            )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
    )

    module.apply_state(
        adapter,
        current=state(
            1000.0,
            cup=2,
        ),
        target=state(
            1400.0,
            cup=3,
        ),
        settling_policies={
            PARAMETER:
                object(),
        },
    )

    assert events == [
        (
            "parameter",
            1200.0,
        ),
        (
            "parameter",
            1400.0,
        ),
        (
            "cup",
            3,
        ),
    ]


def test_guard_rejects_missing_settling_policy():
    adapter = guarded_adapter()

    with pytest.raises(
        Exception,
        match="settling policy",
    ):
        module.apply_state(
            adapter,
            current=state(
                1000.0
            ),
            target=state(
                1200.0
            ),
            settling_policies={},
            select_target_cup=False,
        )


def test_no_guard_preserves_legacy_offline_path(
    monkeypatch,
):
    adapter = SimpleNamespace()

    captured = {
        "count": 0
    }

    target = state(
        1200.0
    )

    def fake_raw(
        *args,
        **kwargs,
    ):
        captured[
            "count"
        ] += 1

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "_raw_apply_state",
        fake_raw,
    )

    result = module.apply_state(
        adapter,
        current=state(
            1000.0
        ),
        target=target,
        settling_policies={},
        select_target_cup=False,
    )

    assert (
        captured[
            "count"
        ]
        == 1
    )

    assert (
        result.observed_state
        is target
    )