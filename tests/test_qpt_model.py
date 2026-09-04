import pytest

from sirius.qpt_model import (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
    QPT_POLE_PAIR_ASSIGNMENTS,
    evaluate_qpt,
    qpt_asymmetry_bounds_for_common_focus,
    qpt_asymmetry_builder,
    qpt_cfa_is_feasible,
    qpt_commands_from_cfa,
    qpt_commands_from_om,
    qpt_coordinates_from_voltages,
    qpt_focus_bounds_for_common_asymmetry,
    qpt_global_focus_builder,
)
from sirius.state import MachineState


def state(
    *,
    qpt1=1800.0,
    qpt2=3000.0,
    qpt3=2200.0,
    readbacks=None,
):
    return MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        parameters={
            QPT1_PARAMETER: qpt1,
            QPT2_PARAMETER: qpt2,
            QPT3_PARAMETER: qpt3,
        },
        readbacks=(
            {}
            if readbacks is None
            else readbacks
        ),
    )


def test_all_twelve_poles_are_represented_as_six_pairs():
    assert len(
        QPT_POLE_PAIR_ASSIGNMENTS
    ) == 6

    assert {
        (
            assignment.quadrupole,
            assignment.axis,
            assignment.parameter_name,
        )
        for assignment
        in QPT_POLE_PAIR_ASSIGNMENTS
    } == {
        (
            1,
            "x",
            QPT3_PARAMETER,
        ),
        (
            1,
            "y",
            QPT2_PARAMETER,
        ),
        (
            2,
            "x",
            QPT2_PARAMETER,
        ),
        (
            2,
            "y",
            QPT1_PARAMETER,
        ),
        (
            3,
            "x",
            QPT3_PARAMETER,
        ),
        (
            3,
            "y",
            QPT2_PARAMETER,
        ),
    }


def test_voltage_to_reduced_coordinates():
    result = (
        qpt_coordinates_from_voltages(
            1800.0,
            3000.0,
            2200.0,
        )
    )

    # M = V2 - V1 = 1200
    assert (
        result.middle_strength_v
        == pytest.approx(
            1200.0
        )
    )

    # O = V2 - V3 = 800
    assert (
        result.outer_strength_v
        == pytest.approx(
            800.0
        )
    )

    # F = (M + O) / 2 = 1000
    assert (
        result.global_focus_v
        == pytest.approx(
            1000.0
        )
    )

    # A = (M - O) / 2 = 200
    assert (
        result.asymmetry_v
        == pytest.approx(
            200.0
        )
    )

    assert (
        result.common_v
        == pytest.approx(
            3000.0
        )
    )


def test_cfa_inverse_reconstructs_psu_commands():
    result = qpt_commands_from_cfa(
        common_v=3000.0,
        global_focus_v=1000.0,
        asymmetry_v=200.0,
    )

    assert (
        result.qpt1_voltage_v
        == pytest.approx(
            1800.0
        )
    )

    assert (
        result.qpt2_voltage_v
        == pytest.approx(
            3000.0
        )
    )

    assert (
        result.qpt3_voltage_v
        == pytest.approx(
            2200.0
        )
    )


def test_cfa_roundtrip():
    original = (
        qpt_coordinates_from_voltages(
            1234.0,
            3456.0,
            2345.0,
        )
    )

    commands = qpt_commands_from_cfa(
        original.common_v,
        original.global_focus_v,
        original.asymmetry_v,
    )

    recovered = (
        commands.coordinates
    )

    assert (
        recovered.common_v
        == pytest.approx(
            original.common_v
        )
    )

    assert (
        recovered.global_focus_v
        == pytest.approx(
            original.global_focus_v
        )
    )

    assert (
        recovered.asymmetry_v
        == pytest.approx(
            original.asymmetry_v
        )
    )


def test_om_representation_is_equivalent():
    result = qpt_commands_from_om(
        common_v=3000.0,
        outer_strength_v=800.0,
        middle_strength_v=1200.0,
    )

    assert (
        result.qpt1_voltage_v
        == pytest.approx(
            1800.0
        )
    )

    assert (
        result.qpt2_voltage_v
        == pytest.approx(
            3000.0
        )
    )

    assert (
        result.qpt3_voltage_v
        == pytest.approx(
            2200.0
        )
    )


def test_increasing_global_focus_increases_outer_and_middle_equally():
    first = qpt_commands_from_cfa(
        3000.0,
        800.0,
        100.0,
    )

    second = qpt_commands_from_cfa(
        3000.0,
        1000.0,
        100.0,
    )

    delta_middle = (
        second.coordinates.middle_strength_v
        - first.coordinates.middle_strength_v
    )

    delta_outer = (
        second.coordinates.outer_strength_v
        - first.coordinates.outer_strength_v
    )

    assert delta_middle == pytest.approx(
        200.0
    )

    assert delta_outer == pytest.approx(
        200.0
    )


def test_increasing_asymmetry_strengthens_middle_and_weakens_outer():
    first = qpt_commands_from_cfa(
        3000.0,
        1000.0,
        -100.0,
    )

    second = qpt_commands_from_cfa(
        3000.0,
        1000.0,
        100.0,
    )

    assert (
        second.coordinates.middle_strength_v
        - first.coordinates.middle_strength_v
        == pytest.approx(
            200.0
        )
    )

    assert (
        second.coordinates.outer_strength_v
        - first.coordinates.outer_strength_v
        == pytest.approx(
            -200.0
        )
    )


def test_cfa_hardware_limits_are_enforced():
    with pytest.raises(
        ValueError
    ):
        qpt_commands_from_cfa(
            common_v=3000.0,
            global_focus_v=4000.0,
            asymmetry_v=0.0,
        )


def test_exact_zero_voltage_is_allowed():
    result = qpt_commands_from_cfa(
        common_v=3000.0,
        global_focus_v=3000.0,
        asymmetry_v=0.0,
    )

    assert (
        result.qpt1_voltage_v
        == pytest.approx(
            0.0
        )
    )

    assert (
        result.qpt3_voltage_v
        == pytest.approx(
            0.0
        )
    )


def test_exact_6000v_is_allowed():
    result = qpt_commands_from_cfa(
        common_v=6000.0,
        global_focus_v=0.0,
        asymmetry_v=0.0,
    )

    assert (
        result.qpt1_voltage_v
        == pytest.approx(
            6000.0
        )
    )

    assert (
        result.qpt2_voltage_v
        == pytest.approx(
            6000.0
        )
    )

    assert (
        result.qpt3_voltage_v
        == pytest.approx(
            6000.0
        )
    )


def test_feasibility_helper():
    assert qpt_cfa_is_feasible(
        3000.0,
        1000.0,
        200.0,
    ) is True

    assert qpt_cfa_is_feasible(
        3000.0,
        5000.0,
        0.0,
    ) is False


def test_focus_bounds_for_c3000_a0():
    minimum, maximum = (
        qpt_focus_bounds_for_common_asymmetry(
            3000.0,
            0.0,
        )
    )

    assert minimum == pytest.approx(
        -3000.0
    )

    assert maximum == pytest.approx(
        3000.0
    )


def test_asymmetry_bounds_for_c3000_f1000():
    minimum, maximum = (
        qpt_asymmetry_bounds_for_common_focus(
            3000.0,
            1000.0,
        )
    )

    assert minimum == pytest.approx(
        -2000.0
    )

    assert maximum == pytest.approx(
        2000.0
    )


def test_readbacks_are_preferred_for_observed_coordinates():
    current = state(
        readbacks={
            QPT1_PARAMETER: 1750.0,
            QPT2_PARAMETER: 2975.0,
            QPT3_PARAMETER: 2175.0,
        }
    )

    result = evaluate_qpt(
        current
    )

    assert (
        result.command_coordinates.global_focus_v
        == pytest.approx(
            1000.0
        )
    )

    observed = (
        result.best_available_coordinates
    )

    # M = 2975 - 1750 = 1225
    # O = 2975 - 2175 = 800
    # F = 1012.5
    # A = 212.5
    assert (
        observed.middle_strength_v
        == pytest.approx(
            1225.0
        )
    )

    assert (
        observed.outer_strength_v
        == pytest.approx(
            800.0
        )
    )

    assert (
        observed.global_focus_v
        == pytest.approx(
            1012.5
        )
    )

    assert (
        observed.asymmetry_v
        == pytest.approx(
            212.5
        )
    )

    assert (
        result.all_inputs_from_readback
        is True
    )


def test_partial_readback_state_remains_explicit():
    current = state(
        readbacks={
            QPT1_PARAMETER: 1750.0,
        }
    )

    result = evaluate_qpt(
        current
    )

    assert (
        result.qpt1.source
        == "readback"
    )

    assert (
        result.qpt2.source
        == "command"
    )

    assert (
        result.qpt3.source
        == "command"
    )

    assert (
        result.all_inputs_from_readback
        is False
    )


def test_global_focus_builder_preserves_c_and_a():
    current = state()

    before = evaluate_qpt(
        current
    ).command_coordinates

    commands = (
        qpt_global_focus_builder(
            current,
            1200.0,
        )
    )

    after = (
        qpt_coordinates_from_voltages(
            commands[
                QPT1_PARAMETER
            ],
            commands[
                QPT2_PARAMETER
            ],
            commands[
                QPT3_PARAMETER
            ],
        )
    )

    assert (
        after.common_v
        == pytest.approx(
            before.common_v
        )
    )

    assert (
        after.asymmetry_v
        == pytest.approx(
            before.asymmetry_v
        )
    )

    assert (
        after.global_focus_v
        == pytest.approx(
            1200.0
        )
    )


def test_asymmetry_builder_preserves_c_and_f():
    current = state()

    before = evaluate_qpt(
        current
    ).command_coordinates

    commands = (
        qpt_asymmetry_builder(
            current,
            -300.0,
        )
    )

    after = (
        qpt_coordinates_from_voltages(
            commands[
                QPT1_PARAMETER
            ],
            commands[
                QPT2_PARAMETER
            ],
            commands[
                QPT3_PARAMETER
            ],
        )
    )

    assert (
        after.common_v
        == pytest.approx(
            before.common_v
        )
    )

    assert (
        after.global_focus_v
        == pytest.approx(
            before.global_focus_v
        )
    )

    assert (
        after.asymmetry_v
        == pytest.approx(
            -300.0
        )
    )


def test_focus_astigmatism_controls_have_expected_anchor_points():
    from sirius.qpt_model import (
        qpt_commands_from_focus_astigmatism,
    )

    convert = (
        qpt_commands_from_focus_astigmatism
    )

    assert (
        convert(
            0.0,
            50.0,
        )
        == pytest.approx(
            (
                0.0,
                0.0,
                0.0,
            )
        )
    )

    assert (
        convert(
            100.0,
            50.0,
        )
        == pytest.approx(
            (
                6000.0,
                6000.0,
                6000.0,
            )
        )
    )

    assert (
        convert(
            50.0,
            50.0,
        )
        == pytest.approx(
            (
                3000.0,
                3000.0,
                3000.0,
            )
        )
    )

    # Positive A deviation:
    # QPT1 up, QPT3 down.
    assert (
        convert(
            50.0,
            100.0,
        )
        == pytest.approx(
            (
                6000.0,
                3000.0,
                0.0,
            )
        )
    )

    # Negative A deviation:
    # QPT1 down, QPT3 up.
    assert (
        convert(
            50.0,
            0.0,
        )
        == pytest.approx(
            (
                0.0,
                3000.0,
                6000.0,
            )
        )
    )


def test_focus_astigmatism_uses_available_voltage_headroom():
    from sirius.qpt_model import (
        qpt_commands_from_focus_astigmatism,
    )

    convert = (
        qpt_commands_from_focus_astigmatism
    )

    # F=25 -> common focus voltage = 1500 V.
    assert (
        convert(
            25.0,
            100.0,
        )
        == pytest.approx(
            (
                3000.0,
                1500.0,
                0.0,
            )
        )
    )

    assert (
        convert(
            25.0,
            0.0,
        )
        == pytest.approx(
            (
                0.0,
                1500.0,
                3000.0,
            )
        )
    )

    # F=75 -> common focus voltage = 4500 V.
    # Only 1500 V of symmetric headroom remains.
    assert (
        convert(
            75.0,
            100.0,
        )
        == pytest.approx(
            (
                6000.0,
                4500.0,
                3000.0,
            )
        )
    )

    assert (
        convert(
            75.0,
            0.0,
        )
        == pytest.approx(
            (
                3000.0,
                4500.0,
                6000.0,
            )
        )
    )

    # At F=0 and F=100 there is no symmetric headroom,
    # therefore A cannot change the physical commands.
    for astigmatism in (
        0.0,
        50.0,
        100.0,
    ):
        assert (
            convert(
                0.0,
                astigmatism,
            )
            == pytest.approx(
                (
                    0.0,
                    0.0,
                    0.0,
                )
            )
        )

        assert (
            convert(
                100.0,
                astigmatism,
            )
            == pytest.approx(
                (
                    6000.0,
                    6000.0,
                    6000.0,
                )
            )
        )


def test_focus_astigmatism_rejects_invalid_controls():
    from sirius.qpt_model import (
        qpt_commands_from_focus_astigmatism,
    )

    convert = (
        qpt_commands_from_focus_astigmatism
    )

    for focus in (
        -0.001,
        100.001,
        float("nan"),
        float("inf"),
    ):
        with pytest.raises(
            ValueError
        ):
            convert(
                focus,
                50.0,
            )

    for astigmatism in (
        -0.001,
        100.001,
        float("nan"),
        float("inf"),
    ):
        with pytest.raises(
            ValueError
        ):
            convert(
                50.0,
                astigmatism,
            )

    for maximum_voltage_v in (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ):
        with pytest.raises(
            ValueError
        ):
            convert(
                50.0,
                50.0,
                maximum_voltage_v=(
                    maximum_voltage_v
                ),
            )

    assert (
        convert(
            50.0,
            50.0,
            maximum_voltage_v=4000.0,
        )
        == pytest.approx(
            (
                2000.0,
                2000.0,
                2000.0,
            )
        )
    )
