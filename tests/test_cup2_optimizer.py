from types import SimpleNamespace

import pytest

import sirius.cup2_optimizer as module
from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.mass_profile import (
    MassProfile,
)
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.settling import (
    SettlingPolicy,
)
from sirius.state import (
    MachineState,
)


class FakeAdapter:
    pass


def measurement(
    current,
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


def reference(
    current=10e-9,
    *,
    time=0.0,
    state_id="cup1-state",
):
    return SourceReference(
        measurement=measurement(
            current
        ),
        state_id=state_id,
        mass_u=60.0,
        monotonic_s=time,
        created_at_utc=(
            "2026-08-26T18:00:00+00:00"
        ),
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
            "lens2_voltage_v": 5000.0,
            "steerer_x1_v": 0.0,
            "steerer_y1_v": 0.0,
        },
    )


def cup2_state(
    *,
    magnet=34.0,
):
    return MachineState(
        mass_u=60.0,
        cup=2,
        stage=2,
        role="working",
        parameters={
            "sputter_voltage_v": 8000.0,
            "extraction_voltage_v": 19600.0,
            "einzel_lens_voltage_v": 18000.0,
            "magnet_current_a": magnet,
            "lens2_voltage_v": 5000.0,
            "steerer_x1_v": 0.0,
            "steerer_y1_v": 0.0,
        },
    )


def policies():
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        "lens2_voltage_v": policy,
        "steerer_x1_v": policy,
        "steerer_y1_v": policy,
        "einzel_lens_voltage_v": policy,
        "sputter_voltage_v": policy,
        "extraction_voltage_v": policy,
        "magnet_current_a": policy,
    }


def fake_scan_result(
    state,
):
    return SimpleNamespace(
        final_state=state
    )


def test_requires_cup2():
    invalid = cup2_state()
    invalid.cup = 3

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            invalid,
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            policies(),
        )


def test_frozen_magnet_must_match_cup1_reference():
    current = cup2_state(
        magnet=35.0
    )

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            current,
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            policies(),
        )


def test_scan_order_is_lens2_x_y_then_einzel(
    monkeypatch,
):
    current = cup2_state()

    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            time=0.0
        )
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    calls = []

    def fake_scan(
        adapter,
        current_state,
        profile,
        tracker,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        calls.append(
            parameter_name
        )

        return fake_scan_result(
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

    module.optimize_cup2(
        FakeAdapter(),
        current,
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker,
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup2OptimizationPolicy(
                coordinate_passes=2
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert calls == [
        "lens2_voltage_v",
        "steerer_x1_v",
        "steerer_y1_v",
        "einzel_lens_voltage_v",
        "lens2_voltage_v",
        "steerer_x1_v",
        "steerer_y1_v",
        "einzel_lens_voltage_v",
    ]


def test_source_parameters_remain_frozen(
    monkeypatch,
):
    current = cup2_state()

    tracker = SourceReferenceTracker()

    tracker.add(
        reference()
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fake_scan(
        adapter,
        current_state,
        profile,
        tracker,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        parameters = dict(
            current_state.parameters
        )

        if parameter_name == (
            "lens2_voltage_v"
        ):
            parameters[
                parameter_name
            ] += 100.0

        elif parameter_name == (
            "steerer_x1_v"
        ):
            parameters[
                parameter_name
            ] += 5.0

        elif parameter_name == (
            "steerer_y1_v"
        ):
            parameters[
                parameter_name
            ] -= 5.0

        elif parameter_name == (
            "einzel_lens_voltage_v"
        ):
            parameters[
                parameter_name
            ] += 50.0

        new_state = MachineState(
            mass_u=60.0,
            cup=2,
            stage=2,
            parameters=parameters,
        )

        return fake_scan_result(
            new_state
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

    result = module.optimize_cup2(
        FakeAdapter(),
        current,
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker,
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup2OptimizationPolicy(
                coordinate_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert (
        result.final_state.parameters[
            "sputter_voltage_v"
        ]
        == 8000.0
    )

    assert (
        result.final_state.parameters[
            "extraction_voltage_v"
        ]
        == 19600.0
    )

    assert (
        result.final_state.parameters[
            "magnet_current_a"
        ]
        == 34.0
    )


def test_accidental_magnet_change_aborts(
    monkeypatch,
):
    tracker = SourceReferenceTracker()

    tracker.add(
        reference()
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def bad_scan(
        adapter,
        current_state,
        profile,
        tracker,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        parameters = dict(
            current_state.parameters
        )

        parameters[
            "magnet_current_a"
        ] = 34.5

        return fake_scan_result(
            MachineState(
                mass_u=60.0,
                cup=2,
                stage=2,
                parameters=parameters,
            )
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        bad_scan,
    )

    with pytest.raises(
        module.Cup2OptimizationError
    ):
        module.optimize_cup2(
            FakeAdapter(),
            cup2_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker,
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            optimization_policy=(
                module.Cup2OptimizationPolicy(
                    coordinate_passes=1
                )
            ),
            monotonic=lambda: 100.0,
        )


def test_due_reference_is_refreshed(
    monkeypatch,
):
    tracker = SourceReferenceTracker(
        interval_s=600.0
    )

    tracker.add(
        reference(
            current=10e-9,
            time=0.0,
            state_id="old-ref",
        )
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    checks = []

    def fake_reference_check(
        adapter,
        working_state,
        reference_state,
        tracker,
        settling_policies,
        measurement_policy,
        **kwargs,
    ):
        new_reference = reference(
            current=9e-9,
            time=700.0,
            state_id=(
                reference_state.state_id
            ),
        )

        tracker.add(
            new_reference
        )

        result = SimpleNamespace(
            working_state_after=(
                working_state
            ),
            reference=(
                new_reference
            ),
        )

        checks.append(
            result
        )

        return result

    monkeypatch.setattr(
        module,
        "perform_source_reference_check",
        fake_reference_check,
    )

    # Avoid logger-specific fields on the lightweight fake result.
    monkeypatch.setattr(
        module,
        "_log_reference_check",
        lambda logger, result: None,
    )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        lambda adapter, current_state, *args, **kwargs:
            fake_scan_result(
                current_state
            ),
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                7.2e-9
            ),
    )

    result = module.optimize_cup2(
        FakeAdapter(),
        cup2_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker,
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup2OptimizationPolicy(
                coordinate_passes=1
            )
        ),
        monotonic=lambda: 700.0,
    )

    assert len(checks) == 1

    assert (
        tracker.latest.measurement.mean_a
        == pytest.approx(
            9e-9
        )
    )

    assert len(
        result.reference_checks
    ) == 1


def test_final_transmission_uses_latest_reference(
    monkeypatch,
):
    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            current=10e-9,
            time=100.0,
        )
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        lambda adapter, current_state, *args, **kwargs:
            fake_scan_result(
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

    result = module.optimize_cup2(
        FakeAdapter(),
        cup2_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker,
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup2OptimizationPolicy(
                coordinate_passes=1
            )
        ),
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


def test_profile_stores_primary_cup2_parameters_but_not_retuned_einzel(
    monkeypatch,
):
    profile = MassProfile(
        mass_u=60.0
    )

    # Preserve the Cup-1 einzel optimum.
    profile.set_best_command(
        "einzel_lens_voltage_v",
        18000.0,
    )

    tracker = SourceReferenceTracker()

    tracker.add(
        reference()
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fake_scan(
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

        if parameter_name == (
            "lens2_voltage_v"
        ):
            parameters[
                parameter_name
            ] = 6000.0

        elif parameter_name == (
            "steerer_x1_v"
        ):
            parameters[
                parameter_name
            ] = 20.0

        elif parameter_name == (
            "steerer_y1_v"
        ):
            parameters[
                parameter_name
            ] = -15.0

        elif parameter_name == (
            "einzel_lens_voltage_v"
        ):
            parameters[
                parameter_name
            ] = 18200.0

        return fake_scan_result(
            MachineState(
                mass_u=60.0,
                cup=2,
                stage=2,
                parameters=parameters,
            )
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

    result = module.optimize_cup2(
        FakeAdapter(),
        cup2_state(),
        cup1_state(),
        profile,
        tracker,
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup2OptimizationPolicy(
                coordinate_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert (
        profile.best_commands[
            "lens2_voltage_v"
        ]
        == 6000.0
    )

    assert (
        profile.best_commands[
            "steerer_x1_v"
        ]
        == 20.0
    )

    assert (
        profile.best_commands[
            "steerer_y1_v"
        ]
        == -15.0
    )

    # Cup-1 global starting value is deliberately preserved.
    assert (
        profile.best_commands[
            "einzel_lens_voltage_v"
        ]
        == 18000.0
    )

    # But the complete Cup-2 state retains the local retune.
    assert (
        result.final_state.parameters[
            "einzel_lens_voltage_v"
        ]
        == 18200.0
    )

    assert (
        profile.best_state_ids[
            "cup2_best"
        ]
        == result.final_state.state_id
    )


def test_below_noise_final_beam_is_rejected(
    monkeypatch,
):
    tracker = SourceReferenceTracker()

    tracker.add(
        reference()
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        lambda adapter, current_state, *args, **kwargs:
            fake_scan_result(
                current_state
            ),
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
        module.Cup2OptimizationNoBeamError
    ):
        module.optimize_cup2(
            FakeAdapter(),
            cup2_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker,
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            optimization_policy=(
                module.Cup2OptimizationPolicy(
                    coordinate_passes=1
                )
            ),
            monotonic=lambda: 100.0,
        )


def test_primary_rcds_problem_uses_existing_cup2_local_windows_and_comparison_policy():
    from sirius.optimizer_api import (
        ObjectiveEvaluation,
    )

    current = cup2_state()

    profile = MassProfile(
        mass_u=60.0
    )

    optimization_policy = (
        module.Cup2OptimizationPolicy(
            lens2_half_width_v=400.0,
            steerer_half_width_v=25.0,
        )
    )

    comparison_policy = ComparisonPolicy(
        uncertainty_multiple=0.0,
        minimum_absolute_improvement_a=0.0,
        minimum_relative_improvement=0.01,
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
    ) == module.CUP2_PRIMARY_PARAMETERS

    assert problem.initial_point == pytest.approx(
        (
            current.parameters[
                "lens2_voltage_v"
            ],
            current.parameters[
                "steerer_x1_v"
            ],
            current.parameters[
                "steerer_y1_v"
            ],
        )
    )

    assert (
        "einzel_lens_voltage_v"
        not in tuple(
            axis.name
            for axis
            in problem.axes
        )
    )

    half_widths = {
        "lens2_voltage_v": (
            optimization_policy.lens2_half_width_v
        ),
        "steerer_x1_v": (
            optimization_policy.steerer_half_width_v
        ),
        "steerer_y1_v": (
            optimization_policy.steerer_half_width_v
        ),
    }

    for axis in problem.axes:
        center = float(
            current.parameters[
                axis.name
            ]
        )

        expected_profile = (
            module._local_profile(
                profile,
                axis.name,
                center,
                half_widths[
                    axis.name
                ],
            )
        )

        (
            expected_minimum,
            expected_maximum,
        ) = expected_profile.effective_bounds(
            axis.name
        )

        assert axis.minimum == pytest.approx(
            expected_minimum
        )

        assert axis.maximum == pytest.approx(
            expected_maximum
        )

        assert (
            axis.minimum
            <= center
            <= axis.maximum
        )

    assert problem.maximize is True

    # This builder only defines the local optimizer geometry.
    # Authoritative machine safety remains in the later
    # evaluator/transition path.
    assert problem.safety_predicate is None

    assert problem.comparison is not None

    incumbent = ObjectiveEvaluation(
        point=problem.initial_point,
        value=1.0,
        sem=0.0,
    )

    statistically_too_small = (
        ObjectiveEvaluation(
            point=problem.initial_point,
            value=1.005,
            sem=0.0,
        )
    )

    meaningful_improvement = (
        ObjectiveEvaluation(
            point=problem.initial_point,
            value=1.02,
            sem=0.0,
        )
    )

    assert (
        problem.is_better(
            statistically_too_small,
            incumbent,
        )
        is False
    )

    assert (
        problem.is_better(
            meaningful_improvement,
            incumbent,
        )
        is True
    )


def test_primary_rcds_problem_does_not_modify_mass_profile():
    current = cup2_state()

    profile = MassProfile(
        mass_u=60.0
    )

    before = profile.to_dict()

    module._build_primary_rcds_problem(
        current,
        profile,
        ComparisonPolicy(),
        module.Cup2OptimizationPolicy(),
    )

    assert (
        profile.to_dict()
        == before
    )


def test_primary_rcds_evaluator_uses_safe_transition_and_transmission(
    monkeypatch,
):
    current = cup2_state()

    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            current=10e-9,
            state_id="current-cup1-reference",
        )
    )

    transition_calls = []
    log_calls = []
    maintenance_calls = []

    def maintenance_hook(
        state,
    ):
        maintenance_calls.append(
            state.state_id
        )

        return state

    def fake_apply_state(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup,
    ):
        transition_calls.append(
            {
                "current": current,
                "target": target,
                "settling_policies": (
                    settling_policies
                ),
                "select_target_cup": (
                    select_target_cup
                ),
            }
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

    logger = SimpleNamespace(
        log_state_transition=lambda transition:
            log_calls.append(
                (
                    "transition",
                    transition,
                )
            ),
        log_measurement=lambda value, **kwargs:
            log_calls.append(
                (
                    "measurement",
                    value,
                    kwargs,
                )
            ),
        log_transmission=lambda value:
            log_calls.append(
                (
                    "transmission",
                    value,
                )
            ),
    )

    evaluator = (
        module._Cup2PrimaryRCDSEvaluator(
            adapter=FakeAdapter(),
            working_state=current,
            cup1_reference_state=(
                cup1_state()
            ),
            tracker=tracker,
            settling_policies=policies(),
            measurement_policy=(
                MeasurementPolicy()
            ),
            logger=logger,
            maintenance_hook=(
                maintenance_hook
            ),
        )
    )

    requested = (
        5500.0,
        20.0,
        -15.0,
    )

    evaluation = evaluator(
        requested
    )

    assert (
        len(
            transition_calls
        )
        == 1
    )

    transition = transition_calls[
        0
    ]

    assert (
        transition[
            "current"
        ].state_id
        == current.state_id
    )

    assert (
        transition[
            "select_target_cup"
        ]
        is False
    )

    target = transition[
        "target"
    ]

    assert (
        target.cup
        == 2
    )

    assert (
        target.parameters[
            "lens2_voltage_v"
        ]
        == 5500.0
    )

    assert (
        target.parameters[
            "steerer_x1_v"
        ]
        == 20.0
    )

    assert (
        target.parameters[
            "steerer_y1_v"
        ]
        == -15.0
    )

    # Upstream correction is not an RCDS axis in this phase.
    assert (
        target.parameters[
            "einzel_lens_voltage_v"
        ]
        == current.parameters[
            "einzel_lens_voltage_v"
        ]
    )

    # Cup-1 source/analyser commands remain frozen.
    for parameter_name in (
        module.CUP2_FROZEN_CUP1_PARAMETERS
    ):
        assert (
            target.parameters[
                parameter_name
            ]
            == current.parameters[
                parameter_name
            ]
        )

    assert (
        evaluation.point
        == pytest.approx(
            requested
        )
    )

    assert (
        evaluation.value
        == pytest.approx(
            0.8
        )
    )

    assert (
        evaluation.sem
        > 0.0
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
            "reference_state_id"
        ]
        == "current-cup1-reference"
    )

    assert (
        evaluation.metadata[
            "observed_state_id"
        ]
        == evaluator.working_state.state_id
    )

    assert (
        evaluation.metadata[
            "transmission"
        ]
        == pytest.approx(
            0.8
        )
    )

    assert maintenance_calls == [
        current.state_id
    ]

    assert [
        entry[
            0
        ]
        for entry
        in log_calls
    ] == [
        "transition",
        "measurement",
        "transmission",
    ]

    measurement_log = log_calls[
        1
    ]

    assert (
        measurement_log[
            2
        ][
            "cup"
        ]
        == 2
    )

    assert (
        measurement_log[
            2
        ][
            "purpose"
        ]
        == "cup2_rcds_candidate"
    )


def test_primary_rcds_evaluator_tracks_actual_machine_state_between_calls(
    monkeypatch,
):
    current = cup2_state()

    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            current=10e-9
        )
    )

    transition_pairs = []

    def fake_apply_state(
        adapter,
        *,
        current,
        target,
        settling_policies,
        select_target_cup,
    ):
        transition_pairs.append(
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
        module._Cup2PrimaryRCDSEvaluator(
            adapter=FakeAdapter(),
            working_state=current,
            cup1_reference_state=(
                cup1_state()
            ),
            tracker=tracker,
            settling_policies=policies(),
            measurement_policy=(
                MeasurementPolicy()
            ),
        )
    )

    first_point = (
        5250.0,
        10.0,
        -5.0,
    )

    second_point = (
        5400.0,
        15.0,
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
            transition_pairs
        )
        == 2
    )

    first_current, first_target = (
        transition_pairs[
            0
        ]
    )

    second_current, second_target = (
        transition_pairs[
            1
        ]
    )

    assert (
        first_current.state_id
        == current.state_id
    )

    # The second safe transition starts from the actually
    # observed state of the first evaluation, not from the
    # optimizer's abstract incumbent.
    assert (
        second_current.state_id
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
