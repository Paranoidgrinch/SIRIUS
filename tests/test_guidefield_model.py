import pytest

from sirius.guidefield_model import (
    evaluate_guidefield,
    guidefield_field_equivalent_v_per_m,
    guidefield_pair_from_commands,
    guidefield_pair_from_difference,
    infer_forward_sign,
)
from sirius.mass_profile import (
    MassProfile,
)
from sirius.state import (
    MachineState,
)


def state(
    *,
    gf1=10.0,
    gf2=20.0,
    readbacks=None,
):
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "guidefield1_voltage_v": gf1,
            "guidefield2_voltage_v": gf2,
        },
        readbacks=(
            {}
            if readbacks is None
            else readbacks
        ),
    )


def test_command_difference_is_gf1_minus_gf2():
    result = evaluate_guidefield(
        state(
            gf1=10.0,
            gf2=20.0,
        )
    )

    assert (
        result.command_difference_v
        == pytest.approx(
            -10.0
        )
    )


def test_paper_example_difference_gives_expected_field_equivalent():
    # DeltaU = +20 V:
    #
    # E = 0.65 * 20 / 0.720
    #   = 18.055... V/m

    field = (
        guidefield_field_equivalent_v_per_m(
            20.0
        )
    )

    assert field == pytest.approx(
        18.0555555556,
        rel=1e-9,
    )


def test_field_equivalent_preserves_command_space_sign():
    positive = (
        guidefield_field_equivalent_v_per_m(
            20.0
        )
    )

    negative = (
        guidefield_field_equivalent_v_per_m(
            -20.0
        )
    )

    assert negative == pytest.approx(
        -positive
    )


def test_readbacks_are_preferred_for_observed_difference():
    current = state(
        gf1=10.0,
        gf2=20.0,
        readbacks={
            "guidefield1_voltage_v": 9.8,
            "guidefield2_voltage_v": 19.1,
        },
    )

    result = evaluate_guidefield(
        current
    )

    assert (
        result.command_difference_v
        == pytest.approx(
            -10.0
        )
    )

    assert (
        result.best_available_difference_v
        == pytest.approx(
            -9.3
        )
    )

    assert (
        result.all_inputs_from_readback
        is True
    )


def test_mixed_command_and_readback_is_explicit():
    current = state(
        readbacks={
            "guidefield1_voltage_v": 9.8,
        }
    )

    result = evaluate_guidefield(
        current
    )

    assert result.gf1.source == (
        "readback"
    )

    assert result.gf2.source == (
        "command"
    )

    assert (
        result.all_inputs_from_readback
        is False
    )


def test_no_forward_direction_is_assumed_without_profile_learning():
    result = evaluate_guidefield(
        state()
    )

    assert (
        result.learned_forward_sign
        is None
    )

    assert (
        result.forward_drive_command_v
        is None
    )

    assert (
        result.forward_drive_best_available_v
        is None
    )


def test_learned_positive_direction_maps_positive_difference_to_forward():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_guidefield_forward_sign(
        1
    )

    result = evaluate_guidefield(
        state(
            gf1=20.0,
            gf2=10.0,
        ),
        profile=profile,
    )

    assert (
        result.command_difference_v
        == 10.0
    )

    assert (
        result.forward_drive_command_v
        == 10.0
    )


def test_learned_negative_direction_maps_negative_difference_to_forward():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_guidefield_forward_sign(
        -1
    )

    result = evaluate_guidefield(
        state(
            gf1=10.0,
            gf2=20.0,
        ),
        profile=profile,
    )

    assert (
        result.command_difference_v
        == -10.0
    )

    assert (
        result.forward_drive_command_v
        == 10.0
    )


def test_profile_mass_must_match():
    profile = MassProfile(
        mass_u=180.0
    )

    with pytest.raises(
        ValueError
    ):
        evaluate_guidefield(
            state(),
            profile=profile,
        )


def test_pair_from_difference_preserves_common_mode():
    pair = (
        guidefield_pair_from_difference(
            difference_v=10.0,
            common_mode_v=15.0,
        )
    )

    assert (
        pair.gf1_voltage_v
        == pytest.approx(
            20.0
        )
    )

    assert (
        pair.gf2_voltage_v
        == pytest.approx(
            10.0
        )
    )

    assert (
        pair.difference_v
        == pytest.approx(
            10.0
        )
    )

    assert (
        pair.common_mode_v
        == pytest.approx(
            15.0
        )
    )


def test_pair_from_difference_respects_individual_supply_limits():
    # GF1 would become 35 V and exceed its 30 V supply.
    with pytest.raises(
        ValueError
    ):
        guidefield_pair_from_difference(
            difference_v=20.0,
            common_mode_v=25.0,
        )


def test_pair_from_commands_calculates_difference_and_common_mode():
    pair = (
        guidefield_pair_from_commands(
            10.0,
            30.0,
        )
    )

    assert (
        pair.difference_v
        == pytest.approx(
            -20.0
        )
    )

    assert (
        pair.common_mode_v
        == pytest.approx(
            20.0
        )
    )


def test_positive_difference_can_be_identified_as_forward():
    sign = infer_forward_sign(
        positive_difference_response=0.80,
        negative_difference_response=0.40,
    )

    assert sign == 1


def test_negative_difference_can_be_identified_as_forward():
    sign = infer_forward_sign(
        positive_difference_response=0.40,
        negative_difference_response=0.80,
    )

    assert sign == -1


def test_similar_responses_do_not_force_direction_learning():
    sign = infer_forward_sign(
        positive_difference_response=0.80,
        negative_difference_response=0.78,
        minimum_relative_advantage=0.05,
    )

    assert sign is None


def test_no_signal_does_not_learn_direction():
    assert infer_forward_sign(
        0.0,
        0.0,
    ) is None