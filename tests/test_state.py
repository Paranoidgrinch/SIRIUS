import json

import pytest

from sirius.state import MachineState, RFQState


def test_machine_state_can_be_created():
    state = MachineState(
        mass_u=60.0,
        cup=1,
        stage=1,
        role="cup1_reference",
        parameters={
            "sputter_voltage_v": 8000.0,
            "extraction_voltage_v": 19600.0,
            "einzel_lens_voltage_v": 18000.0,
            "magnet_current_a": 42.0,
        },
    )

    state.validate()

    assert state.mass_u == 60.0
    assert state.cup == 1
    assert state.role == "cup1_reference"
    assert state.state_id


def test_machine_state_ids_are_unique():
    first = MachineState(
        mass_u=60.0,
        parameters={},
    )

    second = MachineState(
        mass_u=60.0,
        parameters={},
    )

    assert first.state_id != second.state_id


def test_state_rejects_unknown_parameters():
    state = MachineState(
        mass_u=60.0,
        parameters={
            "made_up_voltage": 123.0,
        },
    )

    with pytest.raises(ValueError):
        state.validate()


def test_state_rejects_parameter_outside_bounds():
    state = MachineState(
        mass_u=60.0,
        parameters={
            "sputter_voltage_v": 9500.0,
        },
    )

    with pytest.raises(ValueError):
        state.validate()


def test_disabled_future_inputs_can_still_be_stored():
    state = MachineState(
        mass_u=60.0,
        parameters={
            "hv2_voltage_v": 1000.0,
            "hv3_voltage_v": 1500.0,
        },
    )

    state.validate()

    assert state.parameters["hv2_voltage_v"] == 1000.0
    assert state.parameters["hv3_voltage_v"] == 1500.0


def test_signed_steerer_coordinate_is_stored():
    state = MachineState(
        mass_u=60.0,
        parameters={
            "steerer_x1_v": -35.0,
            "steerer_y1_v": 12.0,
        },
    )

    state.validate()

    assert state.parameters["steerer_x1_v"] == -35.0
    assert state.parameters["steerer_y1_v"] == 12.0


def test_rfq_state_preserves_measured_and_nominal_information():
    state = MachineState(
        mass_u=60.0,
        parameters={},
        rfq=RFQState(
            frequency_hz=1_800_000.0,
            generator_amplitude_vpp=2.5,
            inductance_uh=25.5,
            capacitance_pf=950.0,
            rfq_vpp_measured=280.0,
            q_target=0.55,
            q_nominal=0.48,
            q_measured=0.54,
        ),
    )

    state.validate()

    assert state.rfq.q_nominal == 0.48
    assert state.rfq.q_measured == 0.54
    assert state.rfq.rfq_vpp_measured == 280.0


def test_q_target_above_sirius_limit_is_rejected():
    state = MachineState(
        mass_u=60.0,
        parameters={},
        rfq=RFQState(
            q_target=0.91,
        ),
    )

    with pytest.raises(ValueError):
        state.validate()


def test_measured_q_above_sirius_limit_is_rejected():
    state = MachineState(
        mass_u=60.0,
        parameters={},
        rfq=RFQState(
            q_measured=0.91,
        ),
    )

    with pytest.raises(ValueError):
        state.validate()


def test_machine_state_json_round_trip(tmp_path):
    original = MachineState(
        mass_u=180.0,
        cup=3,
        stage=3,
        role="stage_best",
        parameters={
            "sputter_voltage_v": 8000.0,
            "extraction_voltage_v": 19600.0,
            "ion_cooler_voltage_v": 27558.0,
            "guidefield1_voltage_v": 3.0,
            "guidefield2_voltage_v": 8.0,
        },
        rfq=RFQState(
            frequency_hz=1_800_000.0,
            generator_amplitude_vpp=2.5,
            inductance_uh=25.5,
            capacitance_pf=950.0,
            rfq_vpp_measured=280.0,
            q_target=0.55,
            q_nominal=0.48,
            q_measured=0.54,
        ),
        fixed_conditions={
            "buffer_gas": "He",
            "gas_setting_fixed": True,
        },
        metadata={
            "comment": "test state",
        },
    )

    path = tmp_path / "state.json"

    original.to_json(path)

    restored = MachineState.from_json(path)

    assert restored.state_id == original.state_id
    assert restored.mass_u == 180.0
    assert restored.cup == 3
    assert restored.parameters == original.parameters
    assert restored.rfq.q_measured == 0.54
    assert restored.fixed_conditions["buffer_gas"] == "He"


def test_state_json_contains_schema_version(tmp_path):
    state = MachineState(
        mass_u=60.0,
        parameters={},
    )

    path = tmp_path / "state.json"
    state.to_json(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["state_id"] == state.state_id
    assert data["created_at_utc"] == state.created_at_utc