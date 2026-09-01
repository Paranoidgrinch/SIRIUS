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


def test_primary_rcds_evaluator_uses_safe_transition_and_transmission(
    monkeypatch,
):
    current = cup4_state()
    cup3 = cup3_state()
    source_tracker = tracker()

    transition_calls = []

    def fake_apply_state(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup,
    ):
        transition_calls.append(
            (
                current,
                target,
                settling_policies,
                select_target_cup,
            )
        )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply_state,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                8e-9
            ),
    )

    evaluator = (
        module._Cup4PrimaryRCDSEvaluator(
            adapter=object(),
            working_state=current,
            cup3_reference_state=cup3,
            tracker=source_tracker,
            settling_policies=policies(),
            measurement_policy=(
                MeasurementPolicy()
            ),
            frozen_common_v=3000.0,
        )
    )

    requested_point = (
        1200.0,
        100.0,
        15.0,
        -20.0,
    )

    evaluation = evaluator(
        requested_point
    )

    assert (
        len(
            transition_calls
        )
        == 1
    )

    (
        transition_source,
        transition_target,
        transition_policies,
        select_target_cup,
    ) = transition_calls[
        0
    ]

    assert (
        transition_source.state_id
        == current.state_id
    )

    assert (
        transition_policies
        is evaluator.settling_policies
    )

    assert (
        select_target_cup
        is False
    )

    # C = 3000, F = 1200, A = 100
    #
    # V1 = C - F - A = 1700
    # V2 = C         = 3000
    # V3 = C - F + A = 1900
    assert (
        transition_target.parameters[
            QPT1_PARAMETER
        ]
        == pytest.approx(
            1700.0
        )
    )

    assert (
        transition_target.parameters[
            QPT2_PARAMETER
        ]
        == pytest.approx(
            3000.0
        )
    )

    assert (
        transition_target.parameters[
            QPT3_PARAMETER
        ]
        == pytest.approx(
            1900.0
        )
    )

    assert (
        transition_target.parameters[
            "steerer_x2_v"
        ]
        == pytest.approx(
            15.0
        )
    )

    assert (
        transition_target.parameters[
            "steerer_y2_v"
        ]
        == pytest.approx(
            -20.0
        )
    )

    # The complete upstream Cup-3 transport solution remains
    # unchanged.
    for parameter_name in (
        module.CUP4_FROZEN_UPSTREAM_PARAMETERS
    ):
        assert (
            transition_target.parameters[
                parameter_name
            ]
            == cup3.parameters[
                parameter_name
            ]
        )

    assert (
        transition_target.rfq
        == cup3.rfq
    )

    qpt = module.evaluate_qpt(
        transition_target
    )

    assert (
        qpt.command_coordinates.common_v
        == pytest.approx(
            3000.0
        )
    )

    assert (
        qpt.command_coordinates.global_focus_v
        == pytest.approx(
            1200.0
        )
    )

    assert (
        qpt.command_coordinates.asymmetry_v
        == pytest.approx(
            100.0
        )
    )

    assert (
        evaluator.working_state.state_id
        == transition_target.state_id
    )

    assert (
        evaluation.point
        == pytest.approx(
            requested_point
        )
    )

    assert (
        evaluation.value
        == pytest.approx(
            0.8
        )
    )

    assert (
        evaluation.safe
        is True
    )

    assert (
        evaluation.below_noise_floor
        is False
    )

    assert (
        evaluation.metadata[
            "requested_state_id"
        ]
        == transition_target.state_id
    )

    assert (
        evaluation.metadata[
            "observed_state_id"
        ]
        == transition_target.state_id
    )

    assert (
        evaluation.metadata[
            "reference_state_id"
        ]
        == "cup1-reference"
    )

    assert (
        evaluation.metadata[
            "qpt_common_v"
        ]
        == pytest.approx(
            3000.0
        )
    )


def test_primary_rcds_evaluator_tracks_actual_machine_state_between_calls(
    monkeypatch,
):
    current = cup4_state()

    transition_calls = []

    def fake_apply_state(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup,
    ):
        transition_calls.append(
            (
                current,
                target,
            )
        )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply_state,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                8e-9
            ),
    )

    evaluator = (
        module._Cup4PrimaryRCDSEvaluator(
            adapter=object(),
            working_state=current,
            cup3_reference_state=(
                cup3_state()
            ),
            tracker=tracker(),
            settling_policies=policies(),
            measurement_policy=(
                MeasurementPolicy()
            ),
            frozen_common_v=3000.0,
        )
    )

    first_point = (
        1100.0,
        0.0,
        12.0,
        -12.0,
    )

    second_point = (
        900.0,
        -100.0,
        5.0,
        -8.0,
    )

    first = evaluator(
        first_point
    )

    second = evaluator(
        second_point
    )

    assert (
        len(
            transition_calls
        )
        == 2
    )

    first_source, first_target = (
        transition_calls[
            0
        ]
    )

    second_source, second_target = (
        transition_calls[
            1
        ]
    )

    assert (
        first_source.state_id
        == current.state_id
    )

    # The second transition must start from the actual observed
    # state of the first transition, not from the original state.
    assert (
        second_source.state_id
        == first_target.state_id
    )

    assert (
        evaluator.working_state.state_id
        == second_target.state_id
    )

    assert (
        first.point
        == pytest.approx(
            first_point
        )
    )

    assert (
        second.point
        == pytest.approx(
            second_point
        )
    )

    first_qpt = module.evaluate_qpt(
        first_target
    )

    second_qpt = module.evaluate_qpt(
        second_target
    )

    assert (
        first_qpt.command_coordinates.common_v
        == pytest.approx(
            3000.0
        )
    )

    assert (
        second_qpt.command_coordinates.common_v
        == pytest.approx(
            3000.0
        )
    )

    for target in (
        first_target,
        second_target,
    ):
        for parameter_name in (
            module.CUP4_FROZEN_UPSTREAM_PARAMETERS
        ):
            assert (
                target.parameters[
                    parameter_name
                ]
                == cup3_state().parameters[
                    parameter_name
                ]
            )

        assert (
            target.rfq
            == cup3_state().rfq
        )


def test_primary_rcds_full_mock_integration_loop(
    monkeypatch,
):
    from sirius.rcds_optimizer import (
        RCDSPolicy,
        RobustConjugateDirectionOptimizer,
    )

    current = cup4_state()
    cup3 = cup3_state()
    source_tracker = tracker()

    profile = MassProfile(
        mass_u=60.0
    )

    optimization_policy = (
        module.Cup4OptimizationPolicy(
            steerer_half_width_v=100.0,
        )
    )

    comparison_policy = (
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_absolute_improvement_a=0.0,
            minimum_relative_improvement=0.0,
        )
    )

    # --------------------------------------------------------
    # REAL Cup-4 RCDS problem.
    # --------------------------------------------------------

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

    # Choose a deterministic optimum exactly one eighth of each
    # normalized axis away from the starting point.
    #
    # For the default QPT geometry this remains comfortably
    # inside the coupled physical QPT domain.
    signs = (
        1.0,
        1.0,
        1.0,
        -1.0,
    )

    optimum = tuple(
        float(
            initial
        )
        + sign
        * 0.125
        * float(
            axis.span
        )
        for (
            initial,
            axis,
            sign,
        )
        in zip(
            problem.initial_point,
            problem.axes,
            signs,
        )
    )

    assert (
        problem.is_allowed(
            optimum
        )
        is True
    )

    initial_value = tuple(
        float(
            value
        )
        for value
        in problem.initial_point
    )

    assert (
        optimum
        != initial_value
    )

    # --------------------------------------------------------
    # Mock only the real hardware boundary and current reading.
    #
    # The problem builder, RCDS implementation, reduced QPT
    # conversion, evaluator state tracking and transmission
    # calculation remain real production code.
    # --------------------------------------------------------

    machine = {
        "state": current,
    }

    transition_calls = []

    def fake_apply_state(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup,
    ):
        assert (
            current.state_id
            == machine[
                "state"
            ].state_id
        )

        assert (
            select_target_cup
            is False
        )

        transition_calls.append(
            (
                current,
                target,
            )
        )

        machine[
            "state"
        ] = target

        return SimpleNamespace(
            observed_state=target
        )

    def reduced_point(
        state,
    ):
        coordinates = (
            module.evaluate_qpt(
                state
            ).command_coordinates
        )

        return (
            float(
                coordinates
                .global_focus_v
            ),
            float(
                coordinates
                .asymmetry_v
            ),
            float(
                state.parameters[
                    "steerer_x2_v"
                ]
            ),
            float(
                state.parameters[
                    "steerer_y2_v"
                ]
            ),
        )

    def fake_measure_beam_current(
        adapter,
        measurement_policy,
        *,
        noise_floor_a=None,
    ):
        actual = reduced_point(
            machine[
                "state"
            ]
        )

        squared_distance = sum(
            (
                (
                    actual_value
                    - optimum_value
                )
                / float(
                    axis.span
                )
            )
            ** 2
            for (
                actual_value,
                optimum_value,
                axis,
            )
            in zip(
                actual,
                optimum,
                problem.axes,
            )
        )

        # Smooth positive deterministic transmission surface.
        transmission = max(
            0.05,
            0.95
            - 0.50
            * squared_distance,
        )

        return measurement(
            10e-9
            * transmission
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply_state,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure_beam_current,
    )

    frozen_common_v = float(
        module.evaluate_qpt(
            current
        ).command_coordinates.common_v
    )

    evaluator = (
        module._Cup4PrimaryRCDSEvaluator(
            adapter=object(),
            working_state=current,
            cup3_reference_state=cup3,
            tracker=source_tracker,
            settling_policies=policies(),
            measurement_policy=(
                MeasurementPolicy()
            ),
            frozen_common_v=(
                frozen_common_v
            ),
        )
    )

    optimizer = (
        RobustConjugateDirectionOptimizer(
            policy=RCDSPolicy(
                max_iterations=2,
                max_evaluations=80,
                line_samples=5,
                line_half_width=0.25,
                stall_iterations=2,
                parabolic_refinement=True,
                reuse_cached_evaluations=False,
            )
        )
    )

    result = optimizer.optimize(
        problem,
        evaluator,
    )

    # --------------------------------------------------------
    # REAL RCDS result contract.
    # --------------------------------------------------------

    assert (
        result.optimizer_name
        == "rcds"
    )

    assert (
        result.optimizer_version
        == "1.0"
    )

    assert (
        result.evaluations
        > 1
    )

    assert (
        result.best_evaluation.value
        > result.initial_evaluation.value
    )

    assert (
        problem.is_allowed(
            result.best_evaluation.point
        )
        is True
    )

    assert tuple(
        result.metadata[
            "axis_names"
        ]
    ) == (
        module.CUP4_RCDS_AXIS_NAMES
    )

    # Cache reuse is explicitly disabled in this integration
    # run, therefore every recorded optimizer evaluation must
    # correspond to one real evaluator/hardware transition.
    assert (
        len(
            transition_calls
        )
        == result.evaluations
    )

    assert (
        len(
            result.history
        )
        == result.evaluations
    )

    # --------------------------------------------------------
    # Every point that actually reaches the machine must satisfy
    # BOTH the rectangular optimizer bounds and the coupled QPT
    # feasibility predicate.
    # --------------------------------------------------------

    for (
        evaluation,
        transition_pair,
    ) in zip(
        result.history,
        transition_calls,
    ):
        (
            transition_source,
            transition_target,
        ) = transition_pair

        actual_point = reduced_point(
            transition_target
        )

        assert (
            evaluation.point
            == pytest.approx(
                actual_point
            )
        )

        assert (
            problem.is_allowed(
                actual_point
            )
            is True
        )

        qpt = module.evaluate_qpt(
            transition_target
        )

        assert (
            qpt.command_coordinates.common_v
            == pytest.approx(
                frozen_common_v
            )
        )

        assert (
            transition_target.cup
            == 4
        )

        assert (
            transition_target.stage
            in (
                None,
                4,
            )
        )

        for parameter_name in (
            module.CUP4_FROZEN_UPSTREAM_PARAMETERS
        ):
            assert (
                transition_target.parameters[
                    parameter_name
                ]
                == cup3.parameters[
                    parameter_name
                ]
            )

        assert (
            transition_target.rfq
            == cup3.rfq
        )

    # --------------------------------------------------------
    # Stateful physical-machine chain:
    #
    # evaluation N+1 must begin from the observed state left by
    # evaluation N.
    # --------------------------------------------------------

    assert (
        transition_calls[
            0
        ][
            0
        ].state_id
        == current.state_id
    )

    for index in range(
        1,
        len(
            transition_calls
        ),
    ):
        previous_target = (
            transition_calls[
                index - 1
            ][
                1
            ]
        )

        next_source = (
            transition_calls[
                index
            ][
                0
            ]
        )

        assert (
            next_source.state_id
            == previous_target.state_id
        )

    last_target = (
        transition_calls[
            -1
        ][
            1
        ]
    )

    assert (
        evaluator.working_state.state_id
        == last_target.state_id
    )

    assert (
        machine[
            "state"
        ].state_id
        == last_target.state_id
    )

    # --------------------------------------------------------
    # Trace contract.
    # --------------------------------------------------------

    event_types = tuple(
        event[
            "event_type"
        ]
        for event
        in result.metadata[
            "trace"
        ]
    )

    for required_event in (
        "optimizer_started",
        "evaluation",
        "line_search_started",
        "line_search_completed",
        "iteration_completed",
        "optimizer_terminated",
    ):
        assert (
            required_event
            in event_types
        )

    # If RCDS encounters any coupled-QPT-invalid point, that
    # point must appear only as candidate_skipped and must never
    # have reached fake_apply_state().
    skipped = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "candidate_skipped"
        )
    )

    for event in skipped:
        assert (
            problem.is_allowed(
                tuple(
                    event[
                        "physical_point"
                    ]
                )
            )
            is False
        )

    # ------------------------------------------------------------
    # Best-point confirmation contract.
    #
    # optimizer.optimize() leaves the physical machine at the last
    # real evaluation. The best point may be an earlier evaluation.
    # Production integration must therefore explicitly return to the
    # best point and freshly measure it.
    # ------------------------------------------------------------

    last_state_before_confirmation = (
        evaluator.working_state
    )

    transition_count_before_confirmation = len(
        transition_calls
    )

    assert (
        transition_count_before_confirmation
        == result.evaluations
    )

    confirmation = (
        module._confirm_primary_rcds_best(
            evaluator,
            result,
        )
    )

    # Exactly one additional REAL safe evaluator transition.
    assert (
        len(
            transition_calls
        )
        == transition_count_before_confirmation
        + 1
    )

    assert (
        len(
            transition_calls
        )
        == result.evaluations
        + 1
    )

    (
        confirmation_source,
        confirmation_target,
    ) = transition_calls[
        -1
    ]

    assert (
        confirmation_source.state_id
        == last_state_before_confirmation.state_id
    )

    # The fresh evaluator result must report the RCDS best point.
    assert (
        confirmation.point
        == pytest.approx(
            result.best_evaluation.point
        )
    )

    # And the ACTUAL physical command state must now represent
    # exactly the same reduced [F, A, X2, Y2] point.
    confirmed_physical_point = reduced_point(
        evaluator.working_state
    )

    assert (
        confirmed_physical_point
        == pytest.approx(
            result.best_evaluation.point
        )
    )

    assert (
        reduced_point(
            confirmation_target
        )
        == pytest.approx(
            result.best_evaluation.point
        )
    )

    assert (
        evaluator.working_state.state_id
        == confirmation_target.state_id
    )

    assert (
        machine[
            "state"
        ].state_id
        == confirmation_target.state_id
    )

    # Common mode C remains frozen after the return move.
    confirmed_qpt = module.evaluate_qpt(
        evaluator.working_state
    )

    assert (
        confirmed_qpt.command_coordinates.common_v
        == pytest.approx(
            frozen_common_v
        )
    )

    # Complete upstream Cup-3 transport remains frozen.
    for parameter_name in (
        module.CUP4_FROZEN_UPSTREAM_PARAMETERS
    ):
        assert (
            evaluator.working_state.parameters[
                parameter_name
            ]
            == cup3.parameters[
                parameter_name
            ]
        )

    assert (
        evaluator.working_state.rfq
        == cup3.rfq
    )

    # The best point is still part of the same feasible optimizer
    # geometry.
    assert (
        problem.is_allowed(
            confirmation.point
        )
        is True
    )

    # The synthetic objective is deterministic. Therefore the fresh
    # measurement reproduces the original best objective here.
    #
    # Real hardware is intentionally NOT required to reproduce the
    # old value exactly because beam/source drift may occur.
    assert (
        confirmation.value
        == pytest.approx(
            result.best_evaluation.value
        )
    )

    # Confirmation occurs outside the optimizer and must not mutate
    # OptimizationResult.history.
    assert (
        len(
            result.history
        )
        == result.evaluations
    )


def test_run_primary_rcds_orchestrates_components(
    monkeypatch,
):
    current = cup4_state()
    cup3 = cup3_state()

    profile = MassProfile(
        mass_u=60.0
    )

    source_tracker = tracker()
    settling = policies()

    measurement_policy = (
        MeasurementPolicy()
    )

    comparison_policy = (
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_absolute_improvement_a=0.0,
            minimum_relative_improvement=0.0,
        )
    )

    optimization_policy = (
        module.Cup4OptimizationPolicy()
    )

    rcds_policy = module.RCDSPolicy(
        max_iterations=1,
        max_evaluations=10,
        line_samples=3,
        line_half_width=0.25,
        stall_iterations=1,
        parabolic_refinement=False,
        reuse_cached_evaluations=False,
    )

    fake_problem = object()
    fake_result = object()
    fake_confirmation = object()

    final_state = cup4_state()

    calls = []

    def fake_builder(
        working_state,
        received_profile,
        received_comparison_policy,
        received_optimization_policy,
    ):
        calls.append(
            (
                "builder",
                working_state,
                received_profile,
                received_comparison_policy,
                received_optimization_policy,
            )
        )

        return fake_problem

    class FakeEvaluator:
        def __init__(
            self,
            **kwargs,
        ):
            calls.append(
                (
                    "evaluator",
                    kwargs,
                )
            )

            self.working_state = (
                final_state
            )

    class FakeOptimizer:
        def __init__(
            self,
            policy=None,
        ):
            calls.append(
                (
                    "optimizer_init",
                    policy,
                )
            )

        def optimize(
            self,
            problem,
            evaluator,
        ):
            calls.append(
                (
                    "optimize",
                    problem,
                    evaluator,
                )
            )

            return fake_result

    def fake_confirm(
        evaluator,
        result,
    ):
        calls.append(
            (
                "confirm",
                evaluator,
                result,
            )
        )

        return fake_confirmation

    class FakeLogger:
        def log_optimizer_trace(
            self,
            result,
            *,
            stage,
            cup,
        ):
            calls.append(
                (
                    "trace",
                    result,
                    stage,
                    cup,
                )
            )

            return ()

    logger = FakeLogger()

    maintenance_hook = (
        lambda state:
            state
    )

    monkeypatch.setattr(
        module,
        "_build_primary_rcds_problem",
        fake_builder,
    )

    monkeypatch.setattr(
        module,
        "_Cup4PrimaryRCDSEvaluator",
        FakeEvaluator,
    )

    monkeypatch.setattr(
        module,
        "RobustConjugateDirectionOptimizer",
        FakeOptimizer,
    )

    monkeypatch.setattr(
        module,
        "_confirm_primary_rcds_best",
        fake_confirm,
    )

    (
        result,
        confirmation,
        state,
    ) = module._run_primary_rcds(
        object(),
        current,
        cup3,
        profile,
        source_tracker,
        settling,
        measurement_policy,
        comparison_policy,
        optimization_policy,
        rcds_policy=(
            rcds_policy
        ),
        noise_floor_a=1e-12,
        logger=logger,
        maintenance_hook=(
            maintenance_hook
        ),
    )

    assert (
        result
        is fake_result
    )

    assert (
        confirmation
        is fake_confirmation
    )

    assert (
        state
        is final_state
    )

    assert [
        call[
            0
        ]
        for call
        in calls
    ] == [
        "builder",
        "evaluator",
        "optimizer_init",
        "optimize",
        "trace",
        "confirm",
    ]

    # --------------------------------------------------------
    # Builder contract.
    # --------------------------------------------------------

    (
        _,
        builder_state,
        builder_profile,
        builder_comparison,
        builder_policy,
    ) = calls[
        0
    ]

    assert (
        builder_state
        is current
    )

    assert (
        builder_profile
        is profile
    )

    assert (
        builder_comparison
        is comparison_policy
    )

    assert (
        builder_policy
        is optimization_policy
    )

    # --------------------------------------------------------
    # Evaluator construction contract.
    # --------------------------------------------------------

    evaluator_kwargs = calls[
        1
    ][
        1
    ]

    assert (
        evaluator_kwargs[
            "working_state"
        ]
        is current
    )

    assert (
        evaluator_kwargs[
            "cup3_reference_state"
        ]
        is cup3
    )

    assert (
        evaluator_kwargs[
            "tracker"
        ]
        is source_tracker
    )

    assert (
        evaluator_kwargs[
            "settling_policies"
        ]
        is settling
    )

    assert (
        evaluator_kwargs[
            "measurement_policy"
        ]
        is measurement_policy
    )

    assert (
        evaluator_kwargs[
            "frozen_common_v"
        ]
        == pytest.approx(
            3000.0
        )
    )

    assert (
        evaluator_kwargs[
            "noise_floor_a"
        ]
        == pytest.approx(
            1e-12
        )
    )

    assert (
        evaluator_kwargs[
            "logger"
        ]
        is logger
    )

    assert (
        evaluator_kwargs[
            "maintenance_hook"
        ]
        is maintenance_hook
    )

    # --------------------------------------------------------
    # RCDS construction + execution.
    # --------------------------------------------------------

    assert calls[
        2
    ] == (
        "optimizer_init",
        rcds_policy,
    )

    assert (
        calls[
            3
        ][
            1
        ]
        is fake_problem
    )

    # --------------------------------------------------------
    # Completed optimizer trace is persisted BEFORE the fresh
    # best-point confirmation, matching the existing Cup-2
    # production pattern.
    # --------------------------------------------------------

    assert calls[
        4
    ] == (
        "trace",
        fake_result,
        4,
        4,
    )

    assert (
        calls[
            5
        ][
            2
        ]
        is fake_result
    )


def test_opt_in_primary_rcds_replaces_legacy_scans(
    monkeypatch,
):
    current = cup4_state()
    cup3 = cup3_state()

    profile = MassProfile(
        mass_u=60.0
    )

    source_tracker = tracker()

    # Existing test helper patches:
    #   capture_readbacks
    #   both legacy scan functions
    #   final beam-current measurement
    legacy_calls = patch_common(
        monkeypatch
    )

    rcds_policy = module.RCDSPolicy(
        max_iterations=1,
        max_evaluations=10,
        line_samples=3,
        line_half_width=0.25,
        stall_iterations=1,
        parabolic_refinement=False,
        reuse_cached_evaluations=False,
    )

    fake_optimization = object()
    fake_confirmation = object()

    parameters = dict(
        current.parameters
    )

    # Preserve C=3000 while moving the RCDS point to:
    #
    # F = 1200
    # A = 100
    #
    # V1 = C - F - A = 1700
    # V2 = C         = 3000
    # V3 = C - F + A = 1900
    parameters[
        QPT1_PARAMETER
    ] = 1700.0

    parameters[
        QPT2_PARAMETER
    ] = 3000.0

    parameters[
        QPT3_PARAMETER
    ] = 1900.0

    parameters[
        "steerer_x2_v"
    ] = 25.0

    parameters[
        "steerer_y2_v"
    ] = -30.0

    rcds_state = MachineState(
        mass_u=current.mass_u,
        cup=4,
        stage=4,
        role="working",
        parameters=parameters,
        rfq=current.rfq,
    )

    rcds_calls = []

    def fake_run_primary_rcds(
        adapter,
        working_state,
        cup3_reference_state,
        received_profile,
        received_tracker,
        settling_policies,
        measurement_policy,
        comparison_policy,
        optimization_policy,
        *,
        rcds_policy,
        noise_floor_a=None,
        logger=None,
        maintenance_hook=None,
    ):
        rcds_calls.append(
            {
                "working_state":
                    working_state,
                "cup3_reference_state":
                    cup3_reference_state,
                "profile":
                    received_profile,
                "tracker":
                    received_tracker,
                "rcds_policy":
                    rcds_policy,
                "maintenance_hook":
                    maintenance_hook,
            }
        )

        return (
            fake_optimization,
            fake_confirmation,
            rcds_state,
        )

    monkeypatch.setattr(
        module,
        "_run_primary_rcds",
        fake_run_primary_rcds,
    )

    result = module.optimize_cup4(
        object(),
        current,
        cup3,
        cup1_state(),
        profile,
        source_tracker,
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        primary_rcds_policy=(
            rcds_policy
        ),
        monotonic=lambda: 100.0,
    )

    # --------------------------------------------------------
    # Opt-in RCDS path.
    # --------------------------------------------------------

    assert (
        len(rcds_calls)
        == 1
    )

    call = rcds_calls[
        0
    ]

    assert (
        call[
            "working_state"
        ].state_id
        == current.state_id
    )

    assert (
        call[
            "cup3_reference_state"
        ]
        is cup3
    )

    assert (
        call[
            "profile"
        ]
        is profile
    )

    assert (
        call[
            "tracker"
        ]
        is source_tracker
    )

    assert (
        call[
            "rcds_policy"
        ]
        is rcds_policy
    )

    assert callable(
        call[
            "maintenance_hook"
        ]
    )

    # RCDS already owns F, A, X2 and Y2. None of the legacy
    # optimization scans may execute.
    assert (
        legacy_calls
        == []
    )

    assert (
        result.initial_qpt_scan
        is None
    )

    assert (
        result.final_qpt_scan
        is None
    )

    assert (
        result.steerer_scans
        == ()
    )

    assert (
        result.primary_optimization
        is fake_optimization
    )

    assert (
        result.primary_confirmation
        is fake_confirmation
    )

    # --------------------------------------------------------
    # Shared Phase D characterizes the RCDS-returned state.
    # --------------------------------------------------------

    final_qpt = module.evaluate_qpt(
        result.final_state
    )

    assert (
        final_qpt.command_coordinates.common_v
        == pytest.approx(
            3000.0
        )
    )

    assert (
        final_qpt.command_coordinates.global_focus_v
        == pytest.approx(
            1200.0
        )
    )

    assert (
        final_qpt.command_coordinates.asymmetry_v
        == pytest.approx(
            100.0
        )
    )

    assert (
        result.final_state.parameters[
            "steerer_x2_v"
        ]
        == pytest.approx(
            25.0
        )
    )

    assert (
        result.final_state.parameters[
            "steerer_y2_v"
        ]
        == pytest.approx(
            -30.0
        )
    )

    for parameter_name in (
        module.CUP4_FROZEN_UPSTREAM_PARAMETERS
    ):
        assert (
            result.final_state.parameters[
                parameter_name
            ]
            == cup3.parameters[
                parameter_name
            ]
        )

    assert (
        result.final_state.rfq
        == cup3.rfq
    )

    assert (
        result.final_transmission.transmission
        == pytest.approx(
            0.8
        )
    )

    # Existing MassProfile finalization must use the final RCDS
    # state just as it uses the final legacy state.
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
