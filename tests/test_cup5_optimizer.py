from types import SimpleNamespace

import pytest

import sirius.cup5_optimizer as module
from sirius.comparison import ComparisonPolicy
from sirius.esa_model import predict_esa_voltage
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


def upstream():
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


def cup4_parameters():
    result = upstream()

    result.update(
        {
            "quadrupole1_voltage_v": 2000.0,
            "quadrupole2_voltage_v": 3000.0,
            "quadrupole3_voltage_v": 2000.0,

            "steerer_x2_v": 10.0,
            "steerer_y2_v": -10.0,
        }
    )

    return result


def cup4_state():
    return MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        role="stage_best",
        parameters=cup4_parameters(),
        rfq=rfq(),
    )


def cup5_state():
    parameters = cup4_parameters()

    parameters[
        "esa_voltage_v"
    ] = 2500.0

    return MachineState(
        mass_u=60.0,
        cup=5,
        stage=5,
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
    result = SourceReferenceTracker()

    result.add(
        SourceReference(
            measurement=measurement(
                10e-9
            ),
            state_id="cup1",
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

    return {
        name: policy
        for name in (
            *module.CUP5_REQUIRED_PARAMETERS,
        )
    }


def scan_result(state):
    return SimpleNamespace(
        final_state=state
    )


def qpt_result(state):
    return SimpleNamespace(
        final_state=state
    )


def patch_common(monkeypatch, calls=None):
    if calls is None:
        calls = []

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fake_seed(
        adapter,
        state,
        settling_policies,
        *,
        energy_per_volt,
        logger=None,
    ):
        calls.append("esa_seed")

        prediction = predict_esa_voltage(
            state,
            energy_per_volt=energy_per_volt,
        )

        parameters = dict(
            state.parameters
        )

        parameters[
            "esa_voltage_v"
        ] = prediction.nominal_esa_command_v

        after = MachineState(
            mass_u=state.mass_u,
            cup=5,
            stage=5,
            parameters=parameters,
            rfq=state.rfq,
        )

        return module.ESASeedApplication(
            prediction=prediction,
            requested_voltage_v=(
                prediction.nominal_esa_command_v
            ),
            state_before=state,
            transition=SimpleNamespace(),
            state_after=after,
        )

    monkeypatch.setattr(
        module,
        "_apply_esa_seed",
        fake_seed,
    )

    esa_count = {"n": 0}

    def fake_scan(
        adapter,
        current_state,
        profile,
        tracker,
        parameter_name,
        *args,
        **kwargs,
    ):
        if parameter_name == "esa_voltage_v":
            esa_count["n"] += 1
            calls.append(
                f"esa_scan_{esa_count['n']}"
            )
        else:
            calls.append(parameter_name)

        return scan_result(
            current_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        fake_scan,
    )

    def fake_qpt(
        adapter,
        current_state,
        *args,
        **kwargs,
    ):
        calls.append("qpt")
        return qpt_result(
            current_state
        )

    monkeypatch.setattr(
        module,
        "scan_qpt_focus_asymmetry_2d",
        fake_qpt,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(8e-9),
    )

    return calls


def test_requires_cup5():
    invalid = cup5_state()
    invalid.cup = 4

    with pytest.raises(ValueError):
        module._validate_inputs(
            invalid,
            cup4_state(),
            MassProfile(mass_u=60.0),
            tracker(),
            policies(),
        )


def test_cup5_must_start_from_cup4_solution():
    invalid = cup5_state()

    invalid.parameters[
        "steerer_x2_v"
    ] += 5.0

    with pytest.raises(
        ValueError,
        match="steerer_x2_v",
    ):
        module._validate_inputs(
            invalid,
            cup4_state(),
            MassProfile(mass_u=60.0),
            tracker(),
            policies(),
        )


def test_upstream_must_start_from_cup4_solution():
    invalid = cup5_state()

    invalid.parameters[
        "ion_cooler_voltage_v"
    ] += 10.0

    with pytest.raises(
        ValueError,
        match="ion_cooler_voltage_v",
    ):
        module._validate_inputs(
            invalid,
            cup4_state(),
            MassProfile(mass_u=60.0),
            tracker(),
            policies(),
        )


def test_optimizer_phase_order(monkeypatch):
    calls = patch_common(
        monkeypatch
    )

    result = module.optimize_cup5(
        object(),
        cup5_state(),
        cup4_state(),
        cup1_state(),
        MassProfile(mass_u=60.0),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup5OptimizationPolicy(
                steerer_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert calls == [
        "esa_seed",
        "esa_scan_1",
        "steerer_x2_v",
        "steerer_y2_v",
        "qpt",
        "esa_scan_2",
    ]

    assert result.final_state.cup == 5
    assert result.final_state.stage == 5


def test_physics_seed_is_2760v(monkeypatch):
    patch_common(monkeypatch)

    result = module.optimize_cup5(
        object(),
        cup5_state(),
        cup4_state(),
        cup1_state(),
        MassProfile(mass_u=60.0),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    assert (
        result.esa_seed.requested_voltage_v
        == pytest.approx(2760.0)
    )


def test_final_transmission_is_source_normalized(monkeypatch):
    patch_common(monkeypatch)

    result = module.optimize_cup5(
        object(),
        cup5_state(),
        cup4_state(),
        cup1_state(),
        MassProfile(mass_u=60.0),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    assert (
        result.final_transmission.transmission
        == pytest.approx(0.8)
    )

    assert (
        result.final_transmission.transmission_percent
        == pytest.approx(80.0)
    )


def test_accidental_upstream_change_aborts(monkeypatch):
    patch_common(monkeypatch)

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
        ] += 100.0

        return scan_result(
            MachineState(
                mass_u=60.0,
                cup=5,
                stage=5,
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
        module.Cup5OptimizationError
    ):
        module.optimize_cup5(
            object(),
            cup5_state(),
            cup4_state(),
            cup1_state(),
            MassProfile(mass_u=60.0),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            monotonic=lambda: 100.0,
        )


def test_qpt_common_mode_may_not_change(monkeypatch):
    patch_common(monkeypatch)

    def bad_qpt(
        adapter,
        current_state,
        *args,
        **kwargs,
    ):
        parameters = dict(
            current_state.parameters
        )

        parameters[
            "quadrupole2_voltage_v"
        ] += 100.0

        return qpt_result(
            MachineState(
                mass_u=60.0,
                cup=5,
                stage=5,
                parameters=parameters,
                rfq=current_state.rfq,
            )
        )

    monkeypatch.setattr(
        module,
        "scan_qpt_focus_asymmetry_2d",
        bad_qpt,
    )

    with pytest.raises(
        module.Cup5OptimizationError,
        match="common mode",
    ):
        module.optimize_cup5(
            object(),
            cup5_state(),
            cup4_state(),
            cup1_state(),
            MassProfile(mass_u=60.0),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            monotonic=lambda: 100.0,
        )


def test_profile_only_promotes_esa_as_new_primary_command(
    monkeypatch,
):
    patch_common(monkeypatch)

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_best_command(
        "quadrupole1_voltage_v",
        1111.0,
    )

    result = module.optimize_cup5(
        object(),
        cup5_state(),
        cup4_state(),
        cup1_state(),
        profile,
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    assert (
        profile.best_commands[
            "esa_voltage_v"
        ]
        == result.final_state.parameters[
            "esa_voltage_v"
        ]
    )

    # Cup-4 stage-specific QPT optimum is not overwritten by a local
    # Cup-5 transport correction.
    assert (
        profile.best_commands[
            "quadrupole1_voltage_v"
        ]
        == 1111.0
    )

    assert (
        profile.best_state_ids[
            "cup5_best"
        ]
        == result.final_state.state_id
    )


def test_final_state_contains_esa_metadata(monkeypatch):
    patch_common(monkeypatch)

    result = module.optimize_cup5(
        object(),
        cup5_state(),
        cup4_state(),
        cup1_state(),
        MassProfile(mass_u=60.0),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        monotonic=lambda: 100.0,
    )

    assert (
        result.final_state.metadata[
            "esa_command_v"
        ]
        == pytest.approx(2760.0)
    )

    assert (
        result.final_state.metadata[
            "esa_energy_per_volt_command"
        ]
        == pytest.approx(10.0)
    )


def test_below_noise_final_current_is_rejected(
    monkeypatch,
):
    patch_common(monkeypatch)

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
        module.Cup5OptimizationNoBeamError
    ):
        module.optimize_cup5(
            object(),
            cup5_state(),
            cup4_state(),
            cup1_state(),
            MassProfile(mass_u=60.0),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            monotonic=lambda: 100.0,
        )


def test_local_transport_rcds_problem_uses_f_a_x2_y2_with_coupled_qpt_feasibility():
    from sirius.optimizer_api import (
        ObjectiveEvaluation,
    )

    current = cup5_state()

    profile = MassProfile(
        mass_u=60.0
    )

    before = profile.to_dict()

    # Deliberately use a wide mathematical F/A rectangle here so
    # the test proves that the coupled physical QPT constraint is
    # enforced independently of rectangular optimizer bounds.
    optimization_policy = (
        module.Cup5OptimizationPolicy(
            steerer_half_width_v=25.0,
            local_qpt_scan=(
                module.QPT2DScanPolicy(
                    initial_focus_half_width_v=2500.0,
                    initial_asymmetry_half_width_v=2000.0,
                    levels=(
                        module.QPTScanLevel(
                            focus_step_v=500.0,
                            asymmetry_step_v=500.0,
                        ),
                        module.QPTScanLevel(
                            focus_step_v=100.0,
                            asymmetry_step_v=100.0,
                        ),
                    ),
                    refinement_half_width_factor=2.0,
                    max_points_per_level=500,
                )
            ),
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
        module._build_local_transport_rcds_problem(
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
        module.CUP5_LOCAL_TRANSPORT_RCDS_AXIS_NAMES
    )

    assert (
        module.ESA_PARAMETER
        not in tuple(
            axis.name
            for axis
            in problem.axes
        )
    )

    # Current QPT commands:
    #
    #   V1 = 2000
    #   V2 = 3000
    #   V3 = 2000
    #
    # therefore:
    #
    #   C = 3000
    #   F = 1000
    #   A = 0
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

    # F/A rectangular geometry comes directly from the existing
    # Cup-5 local_qpt_scan policy.
    assert (
        problem.axes[
            0
        ].minimum
        == pytest.approx(
            -1500.0
        )
    )

    assert (
        problem.axes[
            0
        ].maximum
        == pytest.approx(
            3500.0
        )
    )

    assert (
        problem.axes[
            1
        ].minimum
        == pytest.approx(
            -2000.0
        )
    )

    assert (
        problem.axes[
            1
        ].maximum
        == pytest.approx(
            2000.0
        )
    )

    # X2/Y2 bounds reuse the existing local-profile semantics.
    for (
        axis,
        parameter_name,
    ) in zip(
        problem.axes[
            2:
        ],
        (
            "steerer_x2_v",
            "steerer_y2_v",
        ),
    ):
        center = float(
            current.parameters[
                parameter_name
            ]
        )

        local = module._local_profile(
            profile,
            parameter_name,
            center,
            optimization_policy
            .steerer_half_width_v,
        )

        (
            expected_minimum,
            expected_maximum,
        ) = (
            local.effective_bounds(
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

    assert (
        problem.is_allowed(
            problem.initial_point
        )
        is True
    )

    # This point is inside all four rectangular optimizer bounds,
    # but with frozen C=3000 it implies:
    #
    #   V1 = C - F - A = -1500 V
    #
    # which violates the verified physical QPT range 0..6000 V.
    coupled_invalid = (
        3000.0,
        1500.0,
        10.0,
        -10.0,
    )

    for (
        axis,
        value,
    ) in zip(
        problem.axes,
        coupled_invalid,
    ):
        assert (
            axis.minimum
            <= value
            <= axis.maximum
        )

    assert (
        problem.is_allowed(
            coupled_invalid
        )
        is False
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

    # Building local optimizer geometry must not mutate persistent
    # learned MassProfile knowledge.
    assert (
        profile.to_dict()
        == before
    )
