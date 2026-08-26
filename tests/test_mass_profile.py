import json

import pytest

from sirius.mass_profile import (
    MassProfile,
    MassProfileStore,
    mass_filename,
)


def test_new_profile_uses_hard_bounds():
    profile = MassProfile(
        mass_u=60.0
    )

    assert profile.effective_bounds(
        "sputter_voltage_v"
    ) == (
        0.0,
        9000.0,
    )

    assert profile.effective_bounds(
        "extraction_voltage_v"
    ) == (
        0.0,
        25000.0,
    )


def test_learned_range_reduces_search_space():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "sputter_voltage_v",
        5000.0,
        9000.0,
        evidence_points=42,
    )

    assert profile.effective_bounds(
        "sputter_voltage_v"
    ) == (
        5000.0,
        9000.0,
    )

    assert (
        profile.learned_ranges[
            "sputter_voltage_v"
        ].evidence_points
        == 42
    )


def test_learned_range_cannot_exceed_hard_bounds():
    profile = MassProfile(
        mass_u=60.0
    )

    with pytest.raises(
        ValueError
    ):
        profile.set_learned_range(
            "sputter_voltage_v",
            -1.0,
            9000.0,
        )

    with pytest.raises(
        ValueError
    ):
        profile.set_learned_range(
            "sputter_voltage_v",
            5000.0,
            9500.0,
        )


def test_reversed_learned_range_is_rejected():
    profile = MassProfile(
        mass_u=60.0
    )

    with pytest.raises(
        ValueError
    ):
        profile.set_learned_range(
            "extraction_voltage_v",
            20000.0,
            12000.0,
        )


def test_learned_range_can_be_cleared():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "extraction_voltage_v",
        12000.0,
        24000.0,
    )

    profile.clear_learned_range(
        "extraction_voltage_v"
    )

    assert profile.effective_bounds(
        "extraction_voltage_v"
    ) == (
        0.0,
        25000.0,
    )


def test_best_command_is_stored():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_best_command(
        "magnet_current_a",
        42.735,
    )

    assert (
        profile.best_commands[
            "magnet_current_a"
        ]
        == 42.735
    )


def test_best_command_must_respect_hard_bounds():
    profile = MassProfile(
        mass_u=60.0
    )

    with pytest.raises(
        ValueError
    ):
        profile.set_best_command(
            "ion_cooler_voltage_v",
            40000.0,
        )


def test_future_disabled_parameter_can_still_be_remembered():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_best_command(
        "hv2_voltage_v",
        1000.0,
    )

    assert (
        profile.best_commands[
            "hv2_voltage_v"
        ]
        == 1000.0
    )


def test_best_stage_states_can_be_remembered():
    profile = MassProfile(
        mass_u=180.0
    )

    profile.set_best_state(
        "cup1_reference",
        "state-a",
    )

    profile.set_best_state(
        "cup6_final",
        "state-b",
    )

    assert (
        profile.best_state_ids[
            "cup1_reference"
        ]
        == "state-a"
    )

    assert (
        profile.best_state_ids[
            "cup6_final"
        ]
        == "state-b"
    )


def test_guidefield_direction_can_be_learned():
    profile = MassProfile(
        mass_u=60.0
    )

    assert (
        profile.guidefield_forward_sign
        is None
    )

    profile.set_guidefield_forward_sign(
        -1
    )

    assert (
        profile.guidefield_forward_sign
        == -1
    )

    with pytest.raises(
        ValueError
    ):
        profile.set_guidefield_forward_sign(
            0
        )


def test_mass_profile_round_trip(
    tmp_path,
):
    store = MassProfileStore(
        tmp_path
    )

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "sputter_voltage_v",
        4800.0,
        9000.0,
        evidence_points=25,
        source="beam_scan",
    )

    profile.set_learned_range(
        "extraction_voltage_v",
        11800.0,
        23000.0,
        evidence_points=31,
        source="beam_scan",
    )

    profile.set_best_command(
        "magnet_current_a",
        42.735,
    )

    profile.set_best_state(
        "cup1_reference",
        "state-123",
    )

    profile.set_guidefield_forward_sign(
        1
    )

    path = store.save(
        profile
    )

    restored = store.load(
        60.0
    )

    assert path.exists()

    assert restored.mass_u == 60.0

    assert restored.effective_bounds(
        "sputter_voltage_v"
    ) == (
        4800.0,
        9000.0,
    )

    assert (
        restored.best_commands[
            "magnet_current_a"
        ]
        == 42.735
    )

    assert (
        restored.best_state_ids[
            "cup1_reference"
        ]
        == "state-123"
    )

    assert (
        restored.guidefield_forward_sign
        == 1
    )


def test_mass_profiles_are_separate_by_mass(
    tmp_path,
):
    store = MassProfileStore(
        tmp_path
    )

    profile_60 = MassProfile(
        mass_u=60.0
    )

    profile_180 = MassProfile(
        mass_u=180.0
    )

    profile_60.set_learned_range(
        "sputter_voltage_v",
        5000.0,
        9000.0,
    )

    profile_180.set_learned_range(
        "sputter_voltage_v",
        6000.0,
        9000.0,
    )

    store.save(
        profile_60
    )

    store.save(
        profile_180
    )

    loaded_60 = store.load(
        60.0
    )

    loaded_180 = store.load(
        180.0
    )

    assert loaded_60.effective_bounds(
        "sputter_voltage_v"
    ) == (
        5000.0,
        9000.0,
    )

    assert loaded_180.effective_bounds(
        "sputter_voltage_v"
    ) == (
        6000.0,
        9000.0,
    )


def test_load_or_create_returns_new_profile_when_missing(
    tmp_path,
):
    store = MassProfileStore(
        tmp_path
    )

    profile = store.load_or_create(
        60.0
    )

    assert profile.mass_u == 60.0
    assert profile.learned_ranges == {}


def test_mass_filename_is_deterministic():
    assert mass_filename(
        60.0
    ) == "mass_60.json"

    assert mass_filename(
        60.5
    ) == "mass_60p5.json"


def test_profile_file_contains_schema_version(
    tmp_path,
):
    store = MassProfileStore(
        tmp_path
    )

    profile = MassProfile(
        mass_u=60.0
    )

    path = store.save(
        profile
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        data["schema_version"]
        == 1
    )


def test_invalid_mass_is_rejected():
    with pytest.raises(
        ValueError
    ):
        MassProfile(
            mass_u=0.0
        ).validate()