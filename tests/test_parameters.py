import pytest

from sirius.parameters import (
    PARAMETERS,
    hardware_steerer_to_sirius,
    sirius_steerer_to_hardware,
    validate_parameter,
)


def test_sirius_voltage_limits():
    assert PARAMETERS["sputter_voltage_v"].maximum == 9000.0
    assert PARAMETERS["extraction_voltage_v"].maximum == 25000.0
    assert PARAMETERS["einzel_lens_voltage_v"].maximum == 25000.0
    assert PARAMETERS["ion_cooler_voltage_v"].maximum == 34000.0


def test_future_hv_inputs_are_modelled_but_disabled():
    assert PARAMETERS["hv2_voltage_v"].enabled is False
    assert PARAMETERS["hv3_voltage_v"].enabled is False


def test_steerer_coordinate_conversion():
    assert sirius_steerer_to_hardware(-250.0) == 0.0
    assert sirius_steerer_to_hardware(0.0) == 250.0
    assert sirius_steerer_to_hardware(250.0) == 500.0

    assert hardware_steerer_to_sirius(250.0) == 0.0


def test_parameter_validation():
    assert validate_parameter("sputter_voltage_v", 8000.0) == 8000.0

    with pytest.raises(ValueError):
        validate_parameter("sputter_voltage_v", 9500.0)
