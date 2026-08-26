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