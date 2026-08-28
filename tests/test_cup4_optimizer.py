from types import SimpleNamespace

import pytest

import sirius.cup4_optimizer as module
from sirius.comparison import ComparisonPolicy
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.qpt_model import (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.settling import SettlingPolicy
from sirius.state import (
    MachineState,
    RFQState,
)


def measurement(
    current=8e-9,
    *,
    below_noise=False,
):
    return BeamMeasurement(
        mean_a=current,
        sigma_a=1e-12,
        sem_a=1e-12,
        n=10,
        duration_s=0.5,
        relative_sem=None,
        precision_threshold_a=1e-12,
        drift_delta_a=0.0,
        stop_reason="test",
        below_noise_floor=below_noise,
        samples=(),
    )


def rfq():
    return RFQState(
        frequency_hz=1.8e6,
        generator_amplitude_vpp=10.0,
        inductance_uh=100.0,
        capacitance_pf=100.0,
        rfq_vpp_measured=500.0,
        q_target=0.45,
        q_measured=0.45,
    )


def upstream_parameters():
    return {
        "sputter_voltage_v": 8000.0,
        "extraction_voltage_v": 19600.0,
        "einzel_lens_voltage_v": 18000.0,
        "magnet_current_a": 34.0,

        "lens2_voltage_v": 6000.0,
        "steerer_x1_v": 20.0,
        "steerer_y1_v": -15.0,

        "ion_cooler_voltage_v": 26460.0,

        "deceleration_voltage_v": 1000.0,
        "acceleration_voltage_v": 1000.0,

        "guidefield1_voltage_v": 10.0,
        "guidefield2_voltage_v": 20.0,
    }


def cup3_state():
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        role="stage_best",
        parameters=(
            upstream_parameters()
        ),
        rfq=rfq(),
    )


def cup4_state():
    parameters = (
        upstream_parameters()
    )

    parameters.update(
        {
            # C = 3000
            # F = 1000
            # A = 0
            QPT1_PARAMETER: 2000.0,
            QPT2_PARAMETER: 3000.0,
            QPT3_PARAMETER: 2000.0,

            "steerer_x2_v": 10.0,
            "steerer_y2_v": -10.0,
        }
    )

    return MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        role="working",
        parameters=parameters,
        rfq=rfq(),
    )


def cup1_state():
    return MachineState(
        mass_u=60.0,
        cup=1,
        stage=1,
        role="cup1_reference",
        parameters={
            "sputter_voltage_v": 8000.0,
            "extraction_voltage_v": 19600.0,
            "einzel_lens_voltage_v": 18000.0,
            "magnet_current_a": 34.0,
        },
    )


def tracker():
    result = (
        SourceReferenceTracker()
    )

    result.add(
        SourceReference(
            measurement=measurement(
                10e-9
            ),
            state_id="cup1-reference",
            mass_u=60.0,
            monotonic_s=100.0,
            created_at_utc=(
                "2026-08-28T07:00:00+00:00"
            ),
        )
    )

    return result


def policies():
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    names = (
        *module.CUP4_PRIMARY_PARAMETERS,
        *module.CUP4_FROZEN_UPSTREAM_PARAMETERS,
    )

    return {
        name: policy
        for name in names
    }


def qpt_result(
    current_state,
):
    return SimpleNamespace(
        final_state=current_state,
        best_target_focus_v=1000.0,
        best_target_asymmetry_v=0.0,
    )


def steerer_result(
    current_state,
):
    return SimpleNamespace(
        final_state=current_state
    )


def patch_common(
    monkeypatch,
    calls=None,
):
    if calls is None:
        calls = []

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    qpt_counter = {
        "n": 0
    }

    def fake_qpt(
        adapter,
        current_state,
        *args,
        **kwargs,
    ):
        qpt_counter["n"] += 1

        calls.append(
            f"qpt_{qpt_counter['n']}"
        )

        return qpt_result(
            current_state
        )

    monkeypatch.setattr(
        module,
        "scan_qpt_focus_asymmetry_2d",
        fake_qpt,
    )

    def fake_steerer(
        adapter,
        current_state,
        profile,
        tracker,
        parameter_name,
        *args,
        **kwargs,
    ):
        calls.append(
            parameter_name
        )

        return steerer_result(
            current_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        fake_steerer,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                8e-9
            ),
    )

    return calls


def test_requires_cup4():
    invalid = cup4_state()
    invalid.cup = 3

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            invalid,
            cup3_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_upstream_must_match_cup3():
    invalid = cup4_state()

    invalid.parameters[
        "ion_cooler_voltage_v"
    ] += 100.0

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            invalid,
            cup3_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_rfq_must_match_cup3():
    invalid = cup4_state()

    invalid.rfq = RFQState(
        frequency_hz=1.7e6
    )

    with pytest.raises(
        ValueError,
        match="RFQ",
    ):
        module._validate_inputs(
            invalid,
            cup3_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_optimizer_phase_order(
    monkeypatch,
):
    calls = patch_common(
        monkeypatch
    )

    result = module.optimize_cup4(
        object(),
        cup4_state(),
        cup3_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup4OptimizationPolicy(
                steerer_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert calls == [
        "qpt_1",
        "steerer_x2_v",
        "steerer_y2_v",
        "qpt_2",
    ]

    assert (
        result.final_state.cup
        == 4
    )

    assert (
        result.final_state.stage
        == 4
    )


def test_default_two_steerer_passes(
    monkeypatch,
):
    calls = patch_common(
        monkeypatch
    )

    module.optimize_cup4(
        object(),
        cup4_state(),
        cup3_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    assert calls == [
        "qpt_1",

        "steerer_x2_v",
        "steerer_y2_v",

        "steerer_x2_v",
        "steerer_y2_v",

        "qpt_2",
    ]


def test_final_transmission_is_source_normalized(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    result = module.optimize_cup4(
        object(),
        cup4_state(),
        cup3_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    assert (
        result.final_transmission.transmission
        == pytest.approx(
            0.8
        )
    )

    assert (
        result.final_transmission.transmission_percent
        == pytest.approx(
            80.0
        )
    )


def test_qpt_common_mode_remains_frozen(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    qpt_counter = {
        "n": 0
    }

    def fake_qpt(
        adapter,
        current_state,
        *args,
        **kwargs,
    ):
        qpt_counter["n"] += 1

        parameters = dict(
            current_state.parameters
        )

        if qpt_counter["n"] == 2:
            parameters[
                QPT2_PARAMETER
            ] = 3100.0

        changed = MachineState(
            mass_u=60.0,
            cup=4,
            stage=4,
            parameters=parameters,
            rfq=current_state.rfq,
        )

        return qpt_result(
            changed
        )

    monkeypatch.setattr(
        module,
        "scan_qpt_focus_asymmetry_2d",
        fake_qpt,
    )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        lambda adapter, current_state, *args, **kwargs:
            steerer_result(
                current_state
            ),
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                8e-9
            ),
    )

    with pytest.raises(
        module.Cup4OptimizationError,
        match="common mode",
    ):
        module.optimize_cup4(
            object(),
            cup4_state(),
            cup3_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            monotonic=lambda: 100.0,
        )


def test_accidental_upstream_change_aborts(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    def bad_steerer(
        adapter,
        current_state,
        profile,
        tracker,
        parameter_name,
        *args,
        **kwargs,
    ):
        parameters = dict(
            current_state.parameters
        )

        parameters[
            "ion_cooler_voltage_v"
        ] += 100.0

        return steerer_result(
            MachineState(
                mass_u=60.0,
                cup=4,
                stage=4,
                parameters=parameters,
                rfq=current_state.rfq,
            )
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        bad_steerer,
    )

    with pytest.raises(
        module.Cup4OptimizationError
    ):
        module.optimize_cup4(
            object(),
            cup4_state(),
            cup3_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            monotonic=lambda: 100.0,
        )


def test_profile_stores_cup4_primary_commands(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    profile = MassProfile(
        mass_u=60.0
    )

    result = module.optimize_cup4(
        object(),
        cup4_state(),
        cup3_state(),
        cup1_state(),
        profile,
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    for parameter_name in (
        module.CUP4_PRIMARY_PARAMETERS
    ):
        assert (
            profile.best_commands[
                parameter_name
            ]
            == result.final_state.parameters[
                parameter_name
            ]
        )

    assert (
        profile.best_state_ids[
            "cup4_best"
        ]
        == result.final_state.state_id
    )

    qpt_meta = (
        profile.metadata[
            "cup4_qpt"
        ]
    )

    assert (
        qpt_meta[
            "common_command_v"
        ]
        == pytest.approx(
            3000.0
        )
    )

    assert (
        qpt_meta[
            "focus_command_v"
        ]
        == pytest.approx(
            1000.0
        )
    )

    assert (
        qpt_meta[
            "asymmetry_command_v"
        ]
        == pytest.approx(
            0.0
        )
    )


def test_final_state_contains_qpt_physics_metadata(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    result = module.optimize_cup4(
        object(),
        cup4_state(),
        cup3_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    metadata = (
        result.final_state.metadata
    )

    assert metadata[
        "qpt_common_command_v"
    ] == pytest.approx(
        3000.0
    )

    assert metadata[
        "qpt_focus_command_v"
    ] == pytest.approx(
        1000.0
    )

    assert metadata[
        "qpt_asymmetry_command_v"
    ] == pytest.approx(
        0.0
    )


def test_below_noise_final_current_is_rejected(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                0.1e-12,
                below_noise=True,
            ),
    )

    with pytest.raises(
        module.Cup4OptimizationNoBeamError
    ):
        module.optimize_cup4(
            object(),
            cup4_state(),
            cup3_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            monotonic=lambda: 100.0,
        )