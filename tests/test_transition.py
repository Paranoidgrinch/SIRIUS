from dataclasses import dataclass

import pytest

from sirius.state import MachineState
from sirius.transition import (
    capture_readbacks,
    plan_state_transition,
)


def make_state(
    *,
    extraction=19000.0,
    lens=18000.0,
    magnet=42.0,
    readback_extraction=None,
):
    readbacks = {}

    if readback_extraction is not None:
        readbacks[
            "extraction_voltage_v"
        ] = readback_extraction

    return MachineState(
        mass_u=60.0,
        parameters={
            "extraction_voltage_v": extraction,
            "einzel_lens_voltage_v": lens,
            "magnet_current_a": magnet,
        },
        readbacks=readbacks,
    )


@dataclass
class FakeAdapter:
    readbacks: dict[str, float | None]

    def read_parameter(self, name):
        return self.readbacks.get(
            name
        )


def test_readback_offset_does_not_trigger_command_change():
    current = make_state(
        extraction=19000.0,
        readback_extraction=18600.0,
    )

    target = make_state(
        extraction=19000.0,
    )

    plan = plan_state_transition(
        current,
        target,
    )

    assert plan.is_noop is True
    assert plan.changed_parameters == ()


def test_changed_command_is_detected():
    current = make_state(
        extraction=19000.0,
    )

    target = make_state(
        extraction=19100.0,
    )

    plan = plan_state_transition(
        current,
        target,
    )

    assert plan.changed_parameters == (
        "extraction_voltage_v",
    )

    change = plan.changes[0]

    assert change.old_command == 19000.0
    assert change.new_command == 19100.0


def test_unchanged_magnet_is_not_touched():
    current = make_state(
        lens=18000.0,
        magnet=42.0,
    )

    target = make_state(
        lens=18100.0,
        magnet=42.0,
    )

    plan = plan_state_transition(
        current,
        target,
    )

    assert (
        "einzel_lens_voltage_v"
        in plan.changed_parameters
    )

    assert (
        "magnet_current_a"
        not in plan.changed_parameters
    )


def test_multiple_changes_follow_parameter_registry_order():
    current = make_state(
        extraction=19000.0,
        lens=18000.0,
        magnet=42.0,
    )

    target = make_state(
        extraction=19100.0,
        lens=18200.0,
        magnet=43.0,
    )

    plan = plan_state_transition(
        current,
        target,
    )

    assert plan.changed_parameters == (
        "extraction_voltage_v",
        "einzel_lens_voltage_v",
        "magnet_current_a",
    )


def test_mass_change_is_not_automatically_applied():
    current = make_state()

    target = MachineState(
        mass_u=180.0,
        parameters=current.parameters.copy(),
    )

    with pytest.raises(ValueError):
        plan_state_transition(
            current,
            target,
        )


def test_target_parameter_missing_from_current_is_added():
    current = MachineState(
        mass_u=60.0,
        parameters={
            "extraction_voltage_v": 19000.0,
        },
    )

    target = MachineState(
        mass_u=60.0,
        parameters={
            "extraction_voltage_v": 19000.0,
            "lens2_voltage_v": 5000.0,
        },
    )

    plan = plan_state_transition(
        current,
        target,
    )

    assert plan.changed_parameters == (
        "lens2_voltage_v",
    )

    assert (
        plan.changes[0].old_command
        is None
    )


def test_capture_readbacks_keeps_command_values_unchanged():
    state = make_state(
        extraction=19000.0,
        lens=18000.0,
        magnet=42.0,
    )

    adapter = FakeAdapter(
        readbacks={
            "extraction_voltage_v": 18600.0,
            "einzel_lens_voltage_v": 17650.0,
            "magnet_current_a": 41.98,
        }
    )

    observed = capture_readbacks(
        adapter,
        state,
    )

    assert (
        observed.parameters[
            "extraction_voltage_v"
        ]
        == 19000.0
    )

    assert (
        observed.readbacks[
            "extraction_voltage_v"
        ]
        == 18600.0
    )

    assert (
        observed.readbacks[
            "einzel_lens_voltage_v"
        ]
        == 17650.0
    )

    assert (
        observed.readbacks[
            "magnet_current_a"
        ]
        == 41.98
    )


def test_capture_readbacks_skips_missing_values():
    state = make_state()

    adapter = FakeAdapter(
        readbacks={
            "extraction_voltage_v": 18600.0,
            "einzel_lens_voltage_v": None,
            "magnet_current_a": None,
        }
    )

    observed = capture_readbacks(
        adapter,
        state,
    )

    assert (
        observed.readbacks[
            "extraction_voltage_v"
        ]
        == 18600.0
    )

    assert (
        "einzel_lens_voltage_v"
        not in observed.readbacks
    )


def test_readback_changes_alone_never_change_transition_plan():
    current = make_state(
        extraction=19000.0,
        readback_extraction=18000.0,
    )

    target = make_state(
        extraction=19000.0,
        readback_extraction=18999.0,
    )

    plan = plan_state_transition(
        current,
        target,
    )

    assert plan.is_noop is True