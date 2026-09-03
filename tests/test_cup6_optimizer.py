from types import SimpleNamespace

import pytest

import sirius.cup6_optimizer as module
from sirius.comparison import ComparisonPolicy
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState, RFQState


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


def cup5_parameters():
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

        "quadrupole1_voltage_v": 2000.0,
        "quadrupole2_voltage_v": 3000.0,
        "quadrupole3_voltage_v": 2000.0,
        "steerer_x2_v": 10.0,
        "steerer_y2_v": -10.0,

        "esa_voltage_v": 2760.0,
    }


def cup5_state():
    return MachineState(
        mass_u=60.0,
        cup=5,
        stage=5,
        role="stage_best",
        parameters=(
            cup5_parameters()
        ),
        rfq=rfq(),
    )


def cup6_state():
    parameters = (
        cup5_parameters()
    )

    parameters.update(
        {
            "lens4_voltage_v": 5000.0,
            "steerer_x3_v": 10.0,
            "steerer_y3_v": -10.0,
        }
    )

    return MachineState(
        mass_u=60.0,
        cup=6,
        stage=6,
        role="working",
        parameters=parameters,
        rfq=rfq(),
    )


def cup1_state():
    return MachineState(
        mass_u=60.0,
        cup=1,
        stage=1,
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
                "2026-08-28T10:00:00+00:00"
            ),
        )
    )

    return result


def policies():
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        parameter_name: policy
        for parameter_name
        in module.CUP6_REQUIRED_PARAMETERS
    }


def scan_result(
    state
):
    return SimpleNamespace(
        final_state=state
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

    lens_count = {
        "n": 0
    }

    def fake_scan(
        adapter,
        current_state,
        profile,
        tracker,
        parameter_name,
        *args,
        **kwargs,
    ):
        if (
            parameter_name
            == "lens4_voltage_v"
        ):
            lens_count["n"] += 1

            calls.append(
                f"lens4_{lens_count['n']}"
            )

        else:
            calls.append(
                parameter_name
            )

        return scan_result(
            current_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        fake_scan,
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


def test_requires_cup6():
    invalid = cup6_state()
    invalid.cup = 5

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            invalid,
            cup5_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_cup6_must_start_from_cup5_solution():
    invalid = cup6_state()

    invalid.parameters[
        "esa_voltage_v"
    ] += 10.0

    with pytest.raises(
        ValueError,
        match="esa_voltage_v",
    ):
        module._validate_inputs(
            invalid,
            cup5_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_qpt_is_frozen_from_cup5():
    invalid = cup6_state()

    invalid.parameters[
        "quadrupole1_voltage_v"
    ] += 10.0

    with pytest.raises(
        ValueError,
        match="quadrupole1_voltage_v",
    ):
        module._validate_inputs(
            invalid,
            cup5_state(),
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

    result = module.optimize_cup6(
        object(),
        cup6_state(),
        cup5_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup6OptimizationPolicy(
                steerer_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert calls == [
        "lens4_1",
        "steerer_x3_v",
        "steerer_y3_v",
        "lens4_2",
    ]

    assert (
        result.final_state.cup
        == 6
    )

    assert (
        result.final_state.stage
        == 6
    )


def test_default_two_steerer_passes(
    monkeypatch,
):
    calls = patch_common(
        monkeypatch
    )

    module.optimize_cup6(
        object(),
        cup6_state(),
        cup5_state(),
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
        "lens4_1",

        "steerer_x3_v",
        "steerer_y3_v",

        "steerer_x3_v",
        "steerer_y3_v",

        "lens4_2",
    ]


def test_final_transmission_is_source_normalized(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    result = module.optimize_cup6(
        object(),
        cup6_state(),
        cup5_state(),
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


def test_accidental_esa_change_aborts(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    def bad_scan(
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
            "esa_voltage_v"
        ] += 10.0

        return scan_result(
            MachineState(
                mass_u=60.0,
                cup=6,
                stage=6,
                parameters=parameters,
                rfq=current_state.rfq,
            )
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        bad_scan,
    )

    with pytest.raises(
        module.Cup6OptimizationError,
        match="esa_voltage_v",
    ):
        module.optimize_cup6(
            object(),
            cup6_state(),
            cup5_state(),
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


def test_accidental_cooler_change_aborts(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    def bad_scan(
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
        ] += 10.0

        return scan_result(
            MachineState(
                mass_u=60.0,
                cup=6,
                stage=6,
                parameters=parameters,
                rfq=current_state.rfq,
            )
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        bad_scan,
    )

    with pytest.raises(
        module.Cup6OptimizationError,
        match="ion_cooler_voltage_v",
    ):
        module.optimize_cup6(
            object(),
            cup6_state(),
            cup5_state(),
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


def test_profile_stores_only_cup6_primary_commands(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_best_command(
        "esa_voltage_v",
        2711.0,
    )

    result = module.optimize_cup6(
        object(),
        cup6_state(),
        cup5_state(),
        cup1_state(),
        profile,
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    for parameter_name in (
        module.CUP6_PRIMARY_PARAMETERS
    ):
        assert (
            profile.best_commands[
                parameter_name
            ]
            == result.final_state.parameters[
                parameter_name
            ]
        )

    # Cup-5 optimum remains a separate stage result.
    assert (
        profile.best_commands[
            "esa_voltage_v"
        ]
        == pytest.approx(
            2711.0
        )
    )

    assert (
        profile.best_state_ids[
            "cup6_best"
        ]
        == result.final_state.state_id
    )


def test_final_state_contains_transport_metadata(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    result = module.optimize_cup6(
        object(),
        cup6_state(),
        cup5_state(),
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
        "lens4_command_v"
    ] == pytest.approx(
        5000.0
    )

    assert metadata[
        "steerer_x3_v"
    ] == pytest.approx(
        10.0
    )

    assert metadata[
        "steerer_y3_v"
    ] == pytest.approx(
        -10.0
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
        module.Cup6OptimizationNoBeamError
    ):
        module.optimize_cup6(
            object(),
            cup6_state(),
            cup5_state(),
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


def test_rfq_change_aborts():
    invalid = cup6_state()

    invalid.rfq = RFQState(
        frequency_hz=1.7e6
    )

    with pytest.raises(
        ValueError,
        match="RFQ",
    ):
        module._validate_inputs(
            invalid,
            cup5_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_primary_rcds_problem_uses_lens4_x3_y3_local_bounds():
    from sirius.optimizer_api import (
        ObjectiveEvaluation,
    )

    current = cup6_state()

    profile = MassProfile(
        mass_u=60.0
    )

    before = profile.to_dict()

    optimization_policy = (
        module.Cup6OptimizationPolicy(
            initial_lens4_half_width_v=1000.0,
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
        == 3
    )

    assert tuple(
        axis.name
        for axis
        in problem.axes
    ) == (
        module.CUP6_PRIMARY_PARAMETERS
    )

    assert (
        problem.initial_point
        == pytest.approx(
            (
                5000.0,
                10.0,
                -10.0,
            )
        )
    )

    expected_half_widths = {
        module.LENS4_PARAMETER: (
            optimization_policy
            .initial_lens4_half_width_v
        ),
        module.CUP6_STEERER_PARAMETERS[
            0
        ]: (
            optimization_policy
            .steerer_half_width_v
        ),
        module.CUP6_STEERER_PARAMETERS[
            1
        ]: (
            optimization_policy
            .steerer_half_width_v
        ),
    }

    # Every RCDS axis reuses the existing Cup-6 local-profile
    # semantics, including hard and learned MassProfile bounds.
    for axis, parameter_name in zip(
        problem.axes,
        module.CUP6_PRIMARY_PARAMETERS,
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
                expected_half_widths[
                    parameter_name
                ],
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

    # Cup 6 has no coupled reduced-coordinate constraint like
    # Cup 4. The physical parameter hard/learned bounds are the
    # complete a-priori optimizer geometry.
    assert (
        problem.safety_predicate
        is None
    )

    assert (
        problem.is_allowed(
            problem.initial_point
        )
        is True
    )

    # Canonical ComparisonPolicy semantics are reused.
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
    current = cup6_state()
    cup5 = cup5_state()
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
        module._Cup6PrimaryRCDSEvaluator(
            adapter=object(),
            working_state=current,
            cup5_reference_state=cup5,
            tracker=source_tracker,
            settling_policies=policies(),
            measurement_policy=(
                MeasurementPolicy()
            ),
        )
    )

    requested_point = (
        5200.0,
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

    assert (
        transition_target.parameters[
            module.LENS4_PARAMETER
        ]
        == pytest.approx(
            5200.0
        )
    )

    assert (
        transition_target.parameters[
            "steerer_x3_v"
        ]
        == pytest.approx(
            15.0
        )
    )

    assert (
        transition_target.parameters[
            "steerer_y3_v"
        ]
        == pytest.approx(
            -20.0
        )
    )

    # The complete upstream Cup-5 transport solution remains
    # unchanged, including ESA and the RFQ state.
    for parameter_name in (
        module.CUP6_FROZEN_UPSTREAM_PARAMETERS
    ):
        assert (
            transition_target.parameters[
                parameter_name
            ]
            == cup5.parameters[
                parameter_name
            ]
        )

    assert (
        transition_target.rfq
        == cup5.rfq
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


def test_primary_rcds_evaluator_tracks_actual_machine_state_between_calls(
    monkeypatch,
):
    current = cup6_state()
    cup5 = cup5_state()

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
        module._Cup6PrimaryRCDSEvaluator(
            adapter=object(),
            working_state=current,
            cup5_reference_state=cup5,
            tracker=tracker(),
            settling_policies=policies(),
            measurement_policy=(
                MeasurementPolicy()
            ),
        )
    )

    first_point = (
        5100.0,
        12.0,
        -12.0,
    )

    second_point = (
        4900.0,
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

    # Evaluation N+1 must begin from the actual observed state
    # left by evaluation N.
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

    assert (
        second_target.parameters[
            module.LENS4_PARAMETER
        ]
        == pytest.approx(
            second_point[
                0
            ]
        )
    )

    assert (
        second_target.parameters[
            "steerer_x3_v"
        ]
        == pytest.approx(
            second_point[
                1
            ]
        )
    )

    assert (
        second_target.parameters[
            "steerer_y3_v"
        ]
        == pytest.approx(
            second_point[
                2
            ]
        )
    )

    for target in (
        first_target,
        second_target,
    ):
        for parameter_name in (
            module.CUP6_FROZEN_UPSTREAM_PARAMETERS
        ):
            assert (
                target.parameters[
                    parameter_name
                ]
                == cup5.parameters[
                    parameter_name
                ]
            )

        assert (
            target.rfq
            == cup5.rfq
        )
