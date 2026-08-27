import pytest

from sirius.cooler_electrodes import (
    cooler_end_pair_from_commands,
    cooler_end_pair_from_common_difference,
    electrode_voltage_observation,
    evaluate_cooler_end_electrodes,
    validate_cooler_bias_context,
)
from sirius.state import (
    MachineState,
)


def state(
    *,
    cooler=26460.0,
    hv1=1200.0,
    hv4=800.0,
    readbacks=None,
):
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "ion_cooler_voltage_v": cooler,
            "deceleration_voltage_v": hv1,
            "acceleration_voltage_v": hv4,
        },
        readbacks=(
            {}
            if readbacks is None
            else readbacks
        ),
    )


def test_command_difference_is_hv1_minus_hv4():
    result = (
        evaluate_cooler_end_electrodes(
            state(
                hv1=1200.0,
                hv4=800.0,
            )
        )
    )

    assert (
        result.command_end_difference_v
        == pytest.approx(
            400.0
        )
    )


def test_common_bias_is_mean_of_hv1_and_hv4():
    result = (
        evaluate_cooler_end_electrodes(
            state(
                hv1=1200.0,
                hv4=800.0,
            )
        )
    )

    assert (
        result.command_common_bias_v
        == pytest.approx(
            1000.0
        )
    )


def test_readbacks_are_preferred_for_observed_coordinates():
    current = state(
        hv1=1200.0,
        hv4=800.0,
        readbacks={
            "ion_cooler_voltage_v": 26410.0,
            "deceleration_voltage_v": 1180.0,
            "acceleration_voltage_v": 810.0,
        },
    )

    result = (
        evaluate_cooler_end_electrodes(
            current
        )
    )

    assert (
        result.command_end_difference_v
        == pytest.approx(
            400.0
        )
    )

    assert (
        result.best_available_end_difference_v
        == pytest.approx(
            370.0
        )
    )

    assert (
        result.best_available_common_bias_v
        == pytest.approx(
            995.0
        )
    )

    assert (
        result.all_inputs_from_readback
        is True
    )


def test_partial_readbacks_are_explicit():
    current = state(
        readbacks={
            "deceleration_voltage_v": 1180.0,
        }
    )

    result = (
        evaluate_cooler_end_electrodes(
            current
        )
    )

    assert (
        result.entrance.source
        == "readback"
    )

    assert (
        result.exit.source
        == "command"
    )

    assert (
        result.cooler.source
        == "command"
    )

    assert (
        result.all_inputs_from_readback
        is False
    )


def test_observation_preserves_command_and_readback():
    current = state(
        hv1=1200.0,
        readbacks={
            "deceleration_voltage_v": 1175.0,
        },
    )

    observation = (
        electrode_voltage_observation(
            current,
            "deceleration_voltage_v",
        )
    )

    assert observation.command_v == 1200.0

    assert observation.readback_v == 1175.0

    assert observation.value_v == 1175.0

    assert observation.source == (
        "readback"
    )


def test_pair_from_commands():
    pair = cooler_end_pair_from_commands(
        1200.0,
        800.0,
    )

    assert (
        pair.entrance_voltage_v
        == 1200.0
    )

    assert (
        pair.exit_voltage_v
        == 800.0
    )

    assert (
        pair.difference_v
        == 400.0
    )

    assert (
        pair.common_bias_v
        == 1000.0
    )


def test_common_difference_roundtrip():
    pair = (
        cooler_end_pair_from_common_difference(
            common_bias_v=1000.0,
            difference_v=400.0,
        )
    )

    assert (
        pair.entrance_voltage_v
        == pytest.approx(
            1200.0
        )
    )

    assert (
        pair.exit_voltage_v
        == pytest.approx(
            800.0
        )
    )


def test_negative_derived_voltage_is_rejected():
    with pytest.raises(
        ValueError
    ):
        cooler_end_pair_from_common_difference(
            common_bias_v=100.0,
            difference_v=500.0,
        )


def test_hv1_above_6_5kv_is_rejected():
    with pytest.raises(
        ValueError
    ):
        cooler_end_pair_from_commands(
            6600.0,
            1000.0,
        )


def test_hv4_above_6_5kv_is_rejected():
    with pytest.raises(
        ValueError
    ):
        cooler_end_pair_from_commands(
            1000.0,
            6600.0,
        )


def test_context_requires_cooler_and_both_end_electrodes():
    incomplete = MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "ion_cooler_voltage_v": 26460.0,
            "deceleration_voltage_v": 1000.0,
        },
    )

    with pytest.raises(
        ValueError
    ):
        validate_cooler_bias_context(
            incomplete
        )


def test_valid_context_passes():
    validate_cooler_bias_context(
        state()
    )