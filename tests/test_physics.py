import pytest

from sirius.physics import (
    beam_energy_ev,
    cooler_residual_energy_ev,
    guidefield_command_difference_v,
    ion_cooler_voltage_for_energy,
)


def test_beam_energy_is_sum_of_source_voltage_magnitudes():
    assert beam_energy_ev(8000.0, 19000.0) == 27000.0


def test_realistic_beam_energy():
    assert beam_energy_ev(8000.0, 19600.0) == 27600.0


def test_cooler_residual_energy():
    assert cooler_residual_energy_ev(
        8000.0,
        19600.0,
        27558.0,
    ) == 42.0


def test_cooler_voltage_from_requested_residual_energy():
    assert ion_cooler_voltage_for_energy(
        8000.0,
        19600.0,
        42.0,
    ) == 27558.0


def test_negative_voltage_magnitude_is_rejected():
    with pytest.raises(ValueError):
        beam_energy_ev(-8000.0, 19600.0)


def test_impossible_residual_energy_is_rejected():
    with pytest.raises(ValueError):
        ion_cooler_voltage_for_energy(
            8000.0,
            19600.0,
            30000.0,
        )


def test_guidefield_difference_does_not_assume_hardware_polarity():
    assert guidefield_command_difference_v(3.0, 8.0) == -5.0
    assert guidefield_command_difference_v(8.0, 3.0) == 5.0