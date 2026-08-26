import pytest

from sirius.cooler_model import (
    ion_cooler_energy_state,
    nominal_cooler_command_for_residual_energy,
    require_valid_cooler_prediction,
    voltage_observation,
)
from sirius.state import MachineState


def state(
    *,
    sputter=8000.0,
    extraction=19600.0,
    cooler=27558.0,
    readbacks=None,
):
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "sputter_voltage_v": sputter,
            "extraction_voltage_v": extraction,
            "ion_cooler_voltage_v": cooler,
        },
        readbacks=(
            {}
            if readbacks is None
            else readbacks
        ),
    )


def test_command_domain_example_gives_42ev():
    result = ion_cooler_energy_state(
        state()
    )

    assert (
        result.beam_energy_command_ev
        == pytest.approx(
            27600.0
        )
    )

    assert (
        result.residual_energy_command_ev
        == pytest.approx(
            42.0
        )
    )


def test_readbacks_are_preferred_for_real_energy():
    current = state(
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
            "ion_cooler_voltage_v": 26460.0,
        }
    )

    result = ion_cooler_energy_state(
        current
    )

    # Commands still describe the reproducible requested state.
    assert (
        result.beam_energy_command_ev
        == pytest.approx(
            27600.0
        )
    )

    assert (
        result.residual_energy_command_ev
        == pytest.approx(
            42.0
        )
    )

    # Physical readbacks describe the actual observed beam energy.
    assert (
        result.beam_energy_best_available_ev
        == pytest.approx(
            26500.0
        )
    )

    assert (
        result.residual_energy_best_available_ev
        == pytest.approx(
            40.0
        )
    )

    assert (
        result.all_energy_inputs_from_readback
        is True
    )


def test_mixed_readback_and_command_is_explicit():
    current = state(
        readbacks={
            "sputter_voltage_v": 7900.0,
        }
    )

    result = ion_cooler_energy_state(
        current
    )

    assert (
        result.sputter.source
        == "readback"
    )

    assert (
        result.extraction.source
        == "command"
    )

    assert (
        result.cooler.source
        == "command"
    )

    assert (
        result.all_energy_inputs_from_readback
        is False
    )


def test_voltage_observation_preserves_command_and_readback():
    current = state(
        extraction=19600.0,
        readbacks={
            "extraction_voltage_v": 18612.0,
        },
    )

    observation = voltage_observation(
        current,
        "extraction_voltage_v",
    )

    assert observation.command_v == 19600.0

    assert observation.readback_v == 18612.0

    assert observation.value_v == 18612.0

    assert observation.source == (
        "readback"
    )


def test_nominal_command_for_42ev_uses_source_readbacks():
    current = state(
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
        }
    )

    prediction = (
        nominal_cooler_command_for_residual_energy(
            current,
            42.0,
        )
    )

    assert (
        prediction.beam_energy_best_available_ev
        == pytest.approx(
            26500.0
        )
    )

    assert (
        prediction.nominal_cooler_command_v
        == pytest.approx(
            26458.0
        )
    )

    assert prediction.within_hardware_range is True


def test_nominal_prediction_does_not_compensate_old_cooler_offset():
    current = state(
        cooler=27558.0,
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
            "ion_cooler_voltage_v": 26400.0,
        },
    )

    prediction = (
        nominal_cooler_command_for_residual_energy(
            current,
            42.0,
        )
    )

    # No "command += observed offset" feedback is applied here.
    assert (
        prediction.nominal_cooler_command_v
        == pytest.approx(
            26458.0
        )
    )


def test_command_example_can_be_reconstructed_from_target_energy():
    current = state(
        readbacks={}
    )

    prediction = (
        nominal_cooler_command_for_residual_energy(
            current,
            42.0,
        )
    )

    assert (
        prediction.nominal_cooler_command_v
        == pytest.approx(
            27558.0
        )
    )


def test_zero_residual_energy_places_cooler_at_beam_energy():
    prediction = (
        nominal_cooler_command_for_residual_energy(
            state(),
            0.0,
        )
    )

    assert (
        prediction.nominal_cooler_command_v
        == pytest.approx(
            27600.0
        )
    )


def test_residual_energy_cannot_exceed_beam_energy():
    with pytest.raises(
        ValueError
    ):
        nominal_cooler_command_for_residual_energy(
            state(),
            30000.0,
        )


def test_negative_residual_energy_is_rejected():
    with pytest.raises(
        ValueError
    ):
        nominal_cooler_command_for_residual_energy(
            state(),
            -1.0,
        )


def test_prediction_outside_cooler_hard_range_is_flagged():
    high_energy_state = state(
        sputter=9000.0,
        extraction=25000.0,
        cooler=30000.0,
    )

    prediction = (
        nominal_cooler_command_for_residual_energy(
            high_energy_state,
            0.0,
        )
    )

    # 9000 + 25000 = exactly the 34 kV cooler limit.
    assert (
        prediction.nominal_cooler_command_v
        == pytest.approx(
            34000.0
        )
    )

    assert prediction.within_hardware_range is True


def test_require_valid_prediction_returns_command():
    prediction = (
        nominal_cooler_command_for_residual_energy(
            state(),
            42.0,
        )
    )

    assert require_valid_cooler_prediction(
        prediction
    ) == pytest.approx(
        27558.0
    )


def test_missing_required_parameter_is_rejected():
    incomplete = MachineState(
        mass_u=60.0,
        cup=3,
        parameters={
            "sputter_voltage_v": 8000.0,
        },
    )

    with pytest.raises(
        ValueError
    ):
        ion_cooler_energy_state(
            incomplete
        )