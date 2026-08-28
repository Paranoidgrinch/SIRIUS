import pytest

import sirius.transition as transition_module
from sirius.cup_ack import (
    CupSelectionPolicy,
    CupSelectionResult,
    CupSelectionTimeoutError,
)
from sirius.flavia_adapter import FlaviaBackendAdapter
from sirius.state import MachineState


def state(
    *,
    cup,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        parameters={
            "sputter_voltage_v": 8000.0,
        },
    )


def test_flavia_adapter_returns_raw_cup_readback():
    adapter = FlaviaBackendAdapter.__new__(
        FlaviaBackendAdapter
    )

    adapter.read_channel_value = (
        lambda channel:
            float("inf")
    )

    result = adapter.read_selected_cup()

    assert result == float("inf")


def test_apply_state_uses_positive_cup_acknowledgement(
    monkeypatch,
):
    current = state(
        cup=3
    )

    target = state(
        cup=4
    )

    calls = []

    class Adapter:
        def select_cup(
            self,
            cup,
        ):
            calls.append(
                (
                    "command",
                    cup,
                )
            )

        def read_selected_cup(
            self,
        ):
            calls.append(
                (
                    "read",
                    None,
                )
            )

            return 4

    adapter = Adapter()

    monkeypatch.setattr(
        transition_module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    def fake_ack(
        *,
        select_cup,
        read_selected_cup,
        target_cup,
        policy,
        **kwargs,
    ):
        assert target_cup == 4

        select_cup(
            target_cup
        )

        assert (
            read_selected_cup()
            == 4
        )

        return CupSelectionResult(
            requested_cup=4,
            confirmed_cup=4,
            elapsed_s=0.2,
            confirmation_count=2,
            samples=(),
        )

    monkeypatch.setattr(
        transition_module,
        "select_cup_and_wait",
        fake_ack,
    )

    result = transition_module.apply_state(
        adapter,
        current=current,
        target=target,
        settling_policies={},
        cup_selection_policy=(
            CupSelectionPolicy(
                timeout_s=1.0,
                poll_interval_s=0.1,
                minimum_wait_s=0.0,
                consecutive_confirmations=1,
            )
        ),
    )

    assert (
        result.selected_cup
        == 4
    )

    assert (
        calls
        == [
            (
                "command",
                4,
            ),
            (
                "read",
                None,
            ),
        ]
    )


def test_apply_state_does_not_claim_selection_after_timeout(
    monkeypatch,
):
    current = state(
        cup=3
    )

    target = state(
        cup=4
    )

    class Adapter:
        def select_cup(
            self,
            cup,
        ):
            pass

        def read_selected_cup(
            self,
        ):
            return 3

    adapter = Adapter()

    capture_called = {
        "value": False
    }

    def fake_capture(
        adapter,
        state,
    ):
        capture_called[
            "value"
        ] = True

        return state

    monkeypatch.setattr(
        transition_module,
        "capture_readbacks",
        fake_capture,
    )

    def fake_ack(
        **kwargs,
    ):
        raise CupSelectionTimeoutError(
            "test timeout"
        )

    monkeypatch.setattr(
        transition_module,
        "select_cup_and_wait",
        fake_ack,
    )

    with pytest.raises(
        CupSelectionTimeoutError
    ):
        transition_module.apply_state(
            adapter,
            current=current,
            target=target,
            settling_policies={},
        )

    # Most importantly, apply_state never reaches the normal successful
    # post-transition readback capture.
    assert (
        capture_called[
            "value"
        ]
        is False
    )


def test_select_target_cup_false_skips_acknowledgement(
    monkeypatch,
):
    current = state(
        cup=4
    )

    target = state(
        cup=4
    )

    class Adapter:
        pass

    adapter = Adapter()

    monkeypatch.setattr(
        transition_module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    def forbidden_ack(
        **kwargs,
    ):
        raise AssertionError(
            "Cup acknowledgement must not run"
        )

    monkeypatch.setattr(
        transition_module,
        "select_cup_and_wait",
        forbidden_ack,
    )

    result = transition_module.apply_state(
        adapter,
        current=current,
        target=target,
        settling_policies={},
        select_target_cup=False,
    )

    assert (
        result.selected_cup
        is None
    )