import pytest

from sirius.esa_model import (
    DEFAULT_ESA_ENERGY_PER_VOLT,
    evaluate_esa,
    infer_energy_per_volt,
    predict_esa_voltage,
    require_valid_esa_prediction,
)
from sirius.state import MachineState


def state(
    *,
    sputter=8000.0,
    extraction=19600.0,
    esa=2760.0,
    readbacks=None,
):
    return MachineState(
        mass_u=60.0,
        cup=5,
        stage=5,
        parameters={
            "sputter_voltage_v": sputter,
            "extraction_voltage_v": extraction,
            "esa_voltage_v": esa,
        },
        readbacks=(
            {}
            if readbacks is None
            else readbacks
        ),
    )


def test_default_calibration_is_ten_energy_per_volt():
    assert (
        DEFAULT_ESA_ENERGY_PER_VOLT
        == pytest.approx(
            10.0
        )
    )


def test_27_6kev_predicts_2760v():
    prediction = (
        predict_esa_voltage(
            state()
        )
    )

    assert (
        prediction.beam_energy_best_available_ev
        == pytest.approx(
            27600.0
        )
    )

    assert (
        prediction.nominal_esa_command_v
        == pytest.approx(
            2760.0
        )
    )

    assert (
        prediction.within_hardware_range
        is True
    )


def test_source_readbacks_are_used_for_prediction():
    current = state(
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
        }
    )

    prediction = (
        predict_esa_voltage(
            current
        )
    )

    assert (
        prediction.beam_energy_best_available_ev
        == pytest.approx(
            26500.0
        )
    )

    assert (
        prediction.nominal_esa_command_v
        == pytest.approx(
            2650.0
        )
    )


def test_prediction_does_not_compensate_existing_esa_offset():
    current = state(
        esa=2760.0,
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
            "esa_voltage_v": 2500.0,
        },
    )

    prediction = (
        predict_esa_voltage(
            current
        )
    )

    # The old command/readback difference is deliberately ignored.
    assert (
        prediction.nominal_esa_command_v
        == pytest.approx(
            2650.0
        )
    )


def test_evaluation_separates_command_and_observed_ratio():
    current = state(
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
            "esa_voltage_v": 2600.0,
        }
    )

    result = evaluate_esa(
        current
    )

    # Command domain:
    # 27600 / 2760 = 10
    assert (
        result.energy_per_volt_command
        == pytest.approx(
            10.0
        )
    )

    # Observed:
    # 26500 / 2600 ~= 10.1923
    assert (
        result.energy_per_volt_best_available
        == pytest.approx(
            26500.0 / 2600.0
        )
    )

    assert (
        result.all_inputs_from_readback
        is True
    )


def test_partial_readbacks_are_explicit():
    result = evaluate_esa(
        state(
            readbacks={
                "sputter_voltage_v": 7900.0,
            }
        )
    )

    assert result.sputter.source == (
        "readback"
    )

    assert result.extraction.source == (
        "command"
    )

    assert result.esa.source == (
        "command"
    )

    assert (
        result.all_inputs_from_readback
        is False
    )


def test_custom_calibration_changes_seed():
    prediction = (
        predict_esa_voltage(
            state(),
            energy_per_volt=12.0,
        )
    )

    assert (
        prediction.nominal_esa_command_v
        == pytest.approx(
            2300.0
        )
    )


def test_nonpositive_calibration_is_rejected():
    with pytest.raises(
        ValueError
    ):
        predict_esa_voltage(
            state(),
            energy_per_volt=0.0,
        )

    with pytest.raises(
        ValueError
    ):
        predict_esa_voltage(
            state(),
            energy_per_volt=-1.0,
        )


def test_esa_seed_above_3kv_is_flagged():
    high_energy = state(
        sputter=9000.0,
        extraction=25000.0,
        esa=3000.0,
    )

    prediction = (
        predict_esa_voltage(
            high_energy
        )
    )

    # 34 keV / 10 = 3.4 kV, above the ESA hard limit.
    assert (
        prediction.nominal_esa_command_v
        == pytest.approx(
            3400.0
        )
    )

    assert (
        prediction.within_hardware_range
        is False
    )


def test_invalid_prediction_is_not_silently_clipped():
    prediction = (
        predict_esa_voltage(
            state(
                sputter=9000.0,
                extraction=25000.0,
                esa=3000.0,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="outside SIRIUS limits",
    ):
        require_valid_esa_prediction(
            prediction
        )


def test_valid_prediction_returns_command():
    prediction = (
        predict_esa_voltage(
            state()
        )
    )

    assert (
        require_valid_esa_prediction(
            prediction
        )
        == pytest.approx(
            2760.0
        )
    )


def test_effective_calibration_can_be_inferred_from_readbacks():
    current = state(
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
            "esa_voltage_v": 2655.0,
        }
    )

    assert (
        infer_energy_per_volt(
            current
        )
        == pytest.approx(
            26500.0 / 2655.0
        )
    )


def test_zero_esa_voltage_has_no_defined_ratio():
    current = state(
        esa=0.0
    )

    result = evaluate_esa(
        current
    )

    assert (
        result.energy_per_volt_command
        is None
    )

    with pytest.raises(
        ValueError
    ):
        infer_energy_per_volt(
            current
        )