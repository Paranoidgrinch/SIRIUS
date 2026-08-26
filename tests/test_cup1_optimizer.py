from types import SimpleNamespace

import pytest

import sirius.cup1_optimizer as module
from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.magnet_model import (
    MagnetPrediction,
)
from sirius.mass_profile import (
    MassProfile,
)
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.reference import (
    SourceReferenceTracker,
)
from sirius.settling import (
    SettlingPolicy,
)
from sirius.state import (
    MachineState,
)
from sirius.transition import (
    AppliedStateResult,
    StateTransitionPlan,
)


class FakeAdapter:
    pass


def measurement(
    current_a,
    *,
    below_noise=False,
):
    return BeamMeasurement(
        mean_a=current_a,
        sigma_a=1e-12,
        sem_a=1e-13,
        n=10,
        duration_s=0.5,
        relative_sem=None,
        precision_threshold_a=1e-12,
        drift_delta_a=0.0,
        stop_reason="test",
        below_noise_floor=below_noise,
        samples=(),
    )


def state(
    *,
    magnet=30.0,
    sputter=8000.0,
    extraction=19600.0,
    einzel=18000.0,
    readbacks=None,
    cup=1,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        stage=1,
        role="working",
        parameters={
            "sputter_voltage_v": sputter,
            "extraction_voltage_v": extraction,
            "einzel_lens_voltage_v": einzel,
            "magnet_current_a": magnet,
        },
        readbacks=(
            {}
            if readbacks is None
            else readbacks
        ),
    )


def policies():
    policy = SettlingPolicy(
        max_readback_span=5.0,
    )

    return {
        "sputter_voltage_v": policy,
        "extraction_voltage_v": policy,
        "einzel_lens_voltage_v": policy,
        "magnet_current_a": policy,
    }


def applied(
    source,
    target,
):
    return AppliedStateResult(
        requested_state=target,
        observed_state=target,
        plan=StateTransitionPlan(
            source_state_id=source.state_id,
            target_state_id=target.state_id,
            changes=(),
        ),
        settling_results=(),
        selected_cup=None,
    )


def prediction(
    current=34.0,
):
    return MagnetPrediction(
        mass_u=60.0,
        beam_energy_ev=26500.0,
        magnetic_field_t=0.35,
        magnetic_field_kg=3.5,
        calculated_current_a=current,
        command_current_a=current,
        current_clamped=False,
    )


def test_cup1_optimizer_requires_cup1():
    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            state(
                cup=2
            ),
            MassProfile(
                mass_u=60.0
            ),
            policies(),
        )


def test_cup1_optimizer_requires_all_four_parameters():
    incomplete = MachineState(
        mass_u=60.0,
        cup=1,
        stage=1,
        parameters={
            "sputter_voltage_v": 8000.0,
        },
    )

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            incomplete,
            MassProfile(
                mass_u=60.0
            ),
            policies(),
        )


def test_physics_calculation_prefers_readbacks(
    monkeypatch,
):
    current = state(
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
            "einzel_lens_voltage_v": 17800.0,
            "magnet_current_a": 30.0,
        }
    )

    captured = {}

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fake_predict(
        mass_u,
        sputter_voltage_v,
        extraction_voltage_v,
    ):
        captured["mass"] = mass_u
        captured["sputter"] = (
            sputter_voltage_v
        )
        captured["extraction"] = (
            extraction_voltage_v
        )

        return prediction(
            34.0
        )

    monkeypatch.setattr(
        module,
        "predict_magnet",
        fake_predict,
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            applied(current, target),
    )

    measurement_values = iter(
        [
            measurement(1e-9),
            measurement(2e-9),
            measurement(3e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurement_values
        ),
    )

    def fake_scan(
        adapter,
        current_state,
        profile,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        return SimpleNamespace(
            final_state=current_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_1d",
        fake_scan,
    )

    module.optimize_cup1(
        FakeAdapter(),
        current,
        MassProfile(
            mass_u=60.0
        ),
        SourceReferenceTracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            minimum_relative_improvement=0.0
        ),
        optimization_policy=(
            module.Cup1OptimizationPolicy(
                source_passes=1
            )
        ),
        monotonic=lambda: 100.0,
        utc_now=lambda: (
            "2026-08-26T14:00:00+00:00"
        ),
    )

    assert captured["mass"] == 60.0
    assert captured["sputter"] == 7900.0
    assert captured["extraction"] == 18600.0


def test_better_physics_prediction_becomes_magnet_seed(
    monkeypatch,
):
    current = state(
        magnet=30.0
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "predict_magnet",
        lambda *args, **kwargs: (
            prediction(
                34.0
            )
        ),
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            applied(current, target),
    )

    measurements = iter(
        [
            measurement(1e-9),
            measurement(5e-9),
            measurement(6e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    first_scan_state = {}

    def fake_scan(
        adapter,
        current_state,
        profile,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        if not first_scan_state:
            first_scan_state[
                "state"
            ] = current_state

        return SimpleNamespace(
            final_state=current_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_1d",
        fake_scan,
    )

    result = module.optimize_cup1(
        FakeAdapter(),
        current,
        MassProfile(
            mass_u=60.0
        ),
        SourceReferenceTracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            minimum_relative_improvement=0.0
        ),
        optimization_policy=(
            module.Cup1OptimizationPolicy(
                source_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert (
        result.magnet_seed_source
        == "physics_prediction"
    )

    assert (
        first_scan_state[
            "state"
        ].parameters[
            "magnet_current_a"
        ]
        == 34.0
    )


def test_good_existing_magnet_is_kept_when_prediction_is_worse(
    monkeypatch,
):
    current = state(
        magnet=30.0
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "predict_magnet",
        lambda *args, **kwargs: (
            prediction(
                34.0
            )
        ),
    )

    transitions = []

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        transitions.append(
            target.parameters[
                "magnet_current_a"
            ]
        )

        return applied(
            current,
            target,
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    measurements = iter(
        [
            measurement(10e-9),
            measurement(5e-9),
            measurement(10e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    scan_starts = []

    def fake_scan(
        adapter,
        current_state,
        profile,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        scan_starts.append(
            (
                parameter_name,
                current_state.parameters[
                    "magnet_current_a"
                ],
            )
        )

        return SimpleNamespace(
            final_state=current_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_1d",
        fake_scan,
    )

    result = module.optimize_cup1(
        FakeAdapter(),
        current,
        MassProfile(
            mass_u=60.0
        ),
        SourceReferenceTracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            minimum_relative_improvement=0.0
        ),
        optimization_policy=(
            module.Cup1OptimizationPolicy(
                source_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert (
        result.magnet_seed_source
        == "existing_state"
    )

    # Move to prediction, then restore 30 A.
    assert transitions[:2] == [
        34.0,
        30.0,
    ]

    assert scan_starts[0] == (
        "magnet_current_a",
        30.0,
    )


def test_no_beam_existing_state_uses_physics_seed_even_if_prediction_is_weak(
    monkeypatch,
):
    current = state(
        magnet=30.0
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "predict_magnet",
        lambda *args, **kwargs: (
            prediction(
                34.0
            )
        ),
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            applied(current, target),
    )

    measurements = iter(
        [
            measurement(
                0.1e-12,
                below_noise=True,
            ),
            measurement(
                0.2e-12,
                below_noise=True,
            ),
            measurement(2e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    first_scan = {}

    def fake_scan(
        adapter,
        current_state,
        profile,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        if not first_scan:
            first_scan[
                "magnet"
            ] = current_state.parameters[
                "magnet_current_a"
            ]

        return SimpleNamespace(
            final_state=current_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_1d",
        fake_scan,
    )

    result = module.optimize_cup1(
        FakeAdapter(),
        current,
        MassProfile(
            mass_u=60.0
        ),
        SourceReferenceTracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup1OptimizationPolicy(
                source_passes=1
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert (
        result.magnet_seed_source
        == "physics_prediction"
    )

    assert first_scan["magnet"] == 34.0


def test_magnet_remains_frozen_during_source_scans(
    monkeypatch,
):
    current = state(
        magnet=34.0
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "predict_magnet",
        lambda *args, **kwargs: (
            prediction(
                34.0
            )
        ),
    )

    measurements = iter(
        [
            measurement(5e-9),
            measurement(8e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    scan_calls = []

    def fake_scan(
        adapter,
        current_state,
        profile,
        parameter_name,
        scan_policy,
        settling_policies,
        measurement_policy,
        comparison_policy,
        **kwargs,
    ):
        scan_calls.append(
            (
                parameter_name,
                current_state.parameters[
                    "magnet_current_a"
                ],
            )
        )

        parameters = dict(
            current_state.parameters
        )

        if parameter_name == (
            "magnet_current_a"
        ):
            parameters[
                parameter_name
            ] = 34.1

        elif parameter_name == (
            "einzel_lens_voltage_v"
        ):
            parameters[
                parameter_name
            ] += 100.0

        elif parameter_name == (
            "sputter_voltage_v"
        ):
            parameters[
                parameter_name
            ] += 100.0

        elif parameter_name == (
            "extraction_voltage_v"
        ):
            parameters[
                parameter_name
            ] += 100.0

        new_state = MachineState(
            mass_u=current_state.mass_u,
            cup=current_state.cup,
            stage=current_state.stage,
            parameters=parameters,
        )

        return SimpleNamespace(
            final_state=new_state
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_1d",
        fake_scan,
    )

    result = module.optimize_cup1(
        FakeAdapter(),
        current,
        MassProfile(
            mass_u=60.0
        ),
        SourceReferenceTracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup1OptimizationPolicy(
                source_passes=2
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert scan_calls[0][0] == (
        "magnet_current_a"
    )

    # Every later source scan sees exactly the frozen 34.1 A.
    assert all(
        magnet == pytest.approx(
            34.1
        )
        for (
            parameter,
            magnet,
        ) in scan_calls[1:]
    )

    assert (
        result.final_state.parameters[
            "magnet_current_a"
        ]
        == pytest.approx(
            34.1
        )
    )


def test_success_creates_reference_and_updates_profile(
    monkeypatch,
):
    current = state(
        magnet=34.0
    )

    profile = MassProfile(
        mass_u=60.0
    )

    tracker = (
        SourceReferenceTracker()
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "predict_magnet",
        lambda *args, **kwargs: (
            prediction(
                34.0
            )
        ),
    )

    measurements = iter(
        [
            measurement(5e-9),
            measurement(9e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    monkeypatch.setattr(
        module,
        "scan_parameter_1d",
        lambda adapter, current_state, *args, **kwargs:
            SimpleNamespace(
                final_state=current_state
            ),
    )

    result = module.optimize_cup1(
        FakeAdapter(),
        current,
        profile,
        tracker,
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        optimization_policy=(
            module.Cup1OptimizationPolicy(
                source_passes=1
            )
        ),
        monotonic=lambda: 1234.0,
        utc_now=lambda: (
            "2026-08-26T14:30:00+00:00"
        ),
    )

    assert (
        result.final_state.role
        == "cup1_reference"
    )

    assert (
        tracker.latest
        is result.reference
    )

    assert (
        result.reference.measurement.mean_a
        == pytest.approx(
            9e-9
        )
    )

    assert (
        result.reference.state_id
        == result.final_state.state_id
    )

    assert (
        profile.best_state_ids[
            "cup1_reference"
        ]
        == result.final_state.state_id
    )

    for parameter in (
        module.CUP1_REQUIRED_PARAMETERS
    ):
        assert (
            profile.best_commands[
                parameter
            ]
            == result.final_state.parameters[
                parameter
            ]
        )


def test_final_below_noise_measurement_is_not_accepted_as_reference(
    monkeypatch,
):
    current = state(
        magnet=34.0
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "predict_magnet",
        lambda *args, **kwargs: (
            prediction(
                34.0
            )
        ),
    )

    measurements = iter(
        [
            measurement(5e-9),
            measurement(
                0.1e-12,
                below_noise=True,
            ),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    monkeypatch.setattr(
        module,
        "scan_parameter_1d",
        lambda adapter, current_state, *args, **kwargs:
            SimpleNamespace(
                final_state=current_state
            ),
    )

    tracker = (
        SourceReferenceTracker()
    )

    with pytest.raises(
        module.Cup1OptimizationNoBeamError
    ):
        module.optimize_cup1(
            FakeAdapter(),
            current,
            MassProfile(
                mass_u=60.0
            ),
            tracker,
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            optimization_policy=(
                module.Cup1OptimizationPolicy(
                    source_passes=1
                )
            ),
        )

    assert tracker.latest is None