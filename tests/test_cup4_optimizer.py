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


def test_primary_rcds_problem_uses_f_a_x2_y2_with_coupled_qpt_feasibility():
    from sirius.optimizer_api import (
        ObjectiveEvaluation,
    )

    current = cup4_state()

    profile = MassProfile(
        mass_u=60.0
    )

    before = profile.to_dict()

    optimization_policy = (
        module.Cup4OptimizationPolicy(
            steerer_half_width_v=25.0,
        )
    )

    comparison_policy = (
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_absolute_improvement_a=0.0,
            minimum_relative_improvement=0.01,
        )
    )

    problem = (
        module._build_primary_rcds_problem(
            current,
            profile,
            comparison_policy,
            optimization_policy,
        )
    )

    assert (
        problem.dimension
        == 4
    )

    assert tuple(
        axis.name
        for axis
        in problem.axes
    ) == (
        module.CUP4_RCDS_AXIS_NAMES
    )

    # Current cup4_state command coordinates:
    #
    # C = 3000
    # F = 1000
    # A = 0
    # X2 = 10
    # Y2 = -10
    assert (
        problem.initial_point
        == pytest.approx(
            (
                1000.0,
                0.0,
                10.0,
                -10.0,
            )
        )
    )

    focus_axis = (
        problem.axes[
            0
        ]
    )

    asymmetry_axis = (
        problem.axes[
            1
        ]
    )

    assert (
        focus_axis.minimum
        == pytest.approx(
            1000.0
            - optimization_policy
            .qpt_scan
            .initial_focus_half_width_v
        )
    )

    assert (
        focus_axis.maximum
        == pytest.approx(
            1000.0
            + optimization_policy
            .qpt_scan
            .initial_focus_half_width_v
        )
    )

    assert (
        asymmetry_axis.minimum
        == pytest.approx(
            -optimization_policy
            .qpt_scan
            .initial_asymmetry_half_width_v
        )
    )

    assert (
        asymmetry_axis.maximum
        == pytest.approx(
            optimization_policy
            .qpt_scan
            .initial_asymmetry_half_width_v
        )
    )

    # Steerer bounds reuse the existing Cup-4 local-profile
    # semantics.
    for axis, parameter_name in zip(
        problem.axes[
            2:
        ],
        module.CUP4_STEERER_PARAMETERS,
    ):
        center = float(
            current.parameters[
                parameter_name
            ]
        )

        expected_profile = (
            module._local_profile(
                profile,
                parameter_name,
                center,
                optimization_policy
                .steerer_half_width_v,
            )
        )

        (
            expected_minimum,
            expected_maximum,
        ) = (
            expected_profile
            .effective_bounds(
                parameter_name
            )
        )

        assert (
            axis.minimum
            == pytest.approx(
                expected_minimum
            )
        )

        assert (
            axis.maximum
            == pytest.approx(
                expected_maximum
            )
        )

    assert (
        problem.maximize
        is True
    )

    assert (
        problem.safety_predicate
        is not None
    )

    # Current command-space point must be feasible.
    assert (
        problem.is_allowed(
            problem.initial_point
        )
        is True
    )

    # This point lies inside all rectangular optimizer-axis
    # bounds but violates the coupled physical QPT constraint.
    #
    # C = 3000
    # F = 3000
    # A = 1500
    #
    # V1 = C - F - A = -1500 V
    invalid_coupled_point = (
        3000.0,
        1500.0,
        10.0,
        -10.0,
    )

    for axis, value in zip(
        problem.axes,
        invalid_coupled_point,
    ):
        assert (
            axis.minimum
            <= value
            <= axis.maximum
        )

    assert (
        problem.is_allowed(
            invalid_coupled_point
        )
        is False
    )

    valid_shifted_point = (
        1000.0,
        500.0,
        10.0,
        -10.0,
    )

    assert (
        problem.is_allowed(
            valid_shifted_point
        )
        is True
    )

    # Reuse canonical ComparisonPolicy semantics.
    incumbent = ObjectiveEvaluation(
        point=problem.initial_point,
        value=1.0,
        sem=0.0,
    )

    too_small = ObjectiveEvaluation(
        point=problem.initial_point,
        value=1.005,
        sem=0.0,
    )

    meaningful = ObjectiveEvaluation(
        point=problem.initial_point,
        value=1.02,
        sem=0.0,
    )

    assert (
        problem.is_better(
            too_small,
            incumbent,
        )
        is False
    )

    assert (
        problem.is_better(
            meaningful,
            incumbent,
        )
        is True
    )

    # Local optimizer geometry must not mutate persistent
    # MassProfile knowledge.
    assert (
        profile.to_dict()
        == before
    )
