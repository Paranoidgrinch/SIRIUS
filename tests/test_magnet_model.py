import pytest

from sirius.magnet_model import (
    MAGNET_MAX_CURRENT_A,
    beam_energy_for_magnet_current_ev,
    extraction_voltage_for_fixed_magnet,
    magnetic_field_kg_for_current,
    predict_magnet,
)


def test_flavia_prediction_for_mass60_at_27_6kev():
    prediction = predict_magnet(
        mass_u=60.0,
        sputter_voltage_v=8000.0,
        extraction_voltage_v=19600.0,
    )

    assert prediction.beam_energy_ev == pytest.approx(
        27600.0
    )

    assert prediction.magnetic_field_kg == pytest.approx(
        3.7054774736,
        rel=1e-9,
    )

    assert prediction.calculated_current_a == pytest.approx(
        34.3593034505,
        rel=1e-9,
    )

    assert prediction.command_current_a == pytest.approx(
        34.3593034505,
        rel=1e-9,
    )

    assert prediction.current_clamped is False


def test_prediction_changes_with_mass():
    mass60 = predict_magnet(
        60.0,
        8000.0,
        19600.0,
    )

    mass180 = predict_magnet(
        180.0,
        8000.0,
        19600.0,
    )

    assert mass180.command_current_a > (
        mass60.command_current_a
    )


def test_prediction_changes_with_beam_energy():
    low = predict_magnet(
        60.0,
        5000.0,
        12000.0,
    )

    high = predict_magnet(
        60.0,
        8000.0,
        19600.0,
    )

    assert high.command_current_a > (
        low.command_current_a
    )


def test_magnet_prediction_never_commands_above_120a():
    prediction = predict_magnet(
        mass_u=500.0,
        sputter_voltage_v=9000.0,
        extraction_voltage_v=25000.0,
    )

    assert prediction.command_current_a <= (
        MAGNET_MAX_CURRENT_A
    )

    if (
        prediction.calculated_current_a
        > MAGNET_MAX_CURRENT_A
    ):
        assert prediction.current_clamped is True


def test_inverse_current_recovers_beam_energy():
    prediction = predict_magnet(
        60.0,
        8000.0,
        19600.0,
    )

    recovered_energy = (
        beam_energy_for_magnet_current_ev(
            60.0,
            prediction.calculated_current_a,
        )
    )

    assert recovered_energy == pytest.approx(
        27600.0,
        rel=1e-10,
    )


def test_inverse_calculation_recovers_extraction_voltage():
    prediction = predict_magnet(
        60.0,
        8000.0,
        19600.0,
    )

    extraction = (
        extraction_voltage_for_fixed_magnet(
            mass_u=60.0,
            magnet_current_a=(
                prediction.calculated_current_a
            ),
            sputter_voltage_v=8000.0,
        )
    )

    assert extraction == pytest.approx(
        19600.0,
        rel=1e-10,
    )


def test_fixed_magnet_exposes_equivalent_energy_relation():
    prediction = predict_magnet(
        180.0,
        7000.0,
        20000.0,
    )

    energy = beam_energy_for_magnet_current_ev(
        180.0,
        prediction.calculated_current_a,
    )

    assert energy == pytest.approx(
        27000.0,
        rel=1e-10,
    )


def test_current_to_field_uses_flavia_calibration():
    field = magnetic_field_kg_for_current(
        34.3593034505
    )

    assert field == pytest.approx(
        3.7054774736,
        rel=1e-9,
    )


def test_invalid_mass_is_rejected():
    with pytest.raises(
        ValueError
    ):
        predict_magnet(
            0.0,
            8000.0,
            19600.0,
        )


def test_negative_current_is_rejected():
    with pytest.raises(
        ValueError
    ):
        magnetic_field_kg_for_current(
            -1.0
        )


def test_impossible_fixed_magnet_extraction_is_rejected():
    with pytest.raises(
        ValueError
    ):
        extraction_voltage_for_fixed_magnet(
            mass_u=60.0,
            magnet_current_a=1.0,
            sputter_voltage_v=8000.0,
        )