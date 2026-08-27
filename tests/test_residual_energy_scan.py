import pytest

import sirius.residual_energy_scan as module
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
from sirius.transition import (
    AppliedStateResult,
    StateTransitionPlan,
)


class FakeAdapter:
    def __init__(self):
        self.commands = []


def measurement(
    current,
    *,
    sem=1e-12,
    below_noise=False,
):
    return BeamMeasurement(
        mean_a=current,
        sigma_a=sem,
        sem_a=sem,
        n=10,
        duration_s=0.5,
        relative_sem=None,
        precision_threshold_a=sem,
        drift_delta_a=0.0,
        stop_reason="test",
        below_noise_floor=below_noise,
        samples=(),
    )


def reference(
    current=10e-9,
):
    return SourceReference(
        measurement=measurement(
            current
        ),
        state_id="cup1-ref",
        mass_u=60.0,
        monotonic_s=0.0,
        created_at_utc=(
            "2026-08-26T19:00:00+00:00"
        ),
    )


def state(
    cooler=26460.0,
    *,
    cup=3,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        stage=3,
        role="working",
        parameters={
            "sputter_voltage_v": 8000.0,
            "extraction_voltage_v": 19600.0,
            "ion_cooler_voltage_v": cooler,
        },
        readbacks={
            "sputter_voltage_v": 7900.0,
            "extraction_voltage_v": 18600.0,
            "ion_cooler_voltage_v": 26460.0,
        },
    )


def policies():
    return {
        "ion_cooler_voltage_v": (
            SettlingPolicy(
                max_readback_span=5.0
            )
        )
    }


def tracker():
    result = SourceReferenceTracker()

    result.add(
        reference()
    )

    return result


def test_policy_requires_decreasing_steps():
    with pytest.raises(
        ValueError
    ):
        module.ResidualEnergyScanPolicy(
            minimum_ev=10.0,
            maximum_ev=100.0,
            steps_ev=(
                1.0,
                10.0,
            ),
        )


def test_energy_grid_includes_upper_edge():
    grid = module._energy_grid(
        10.0,
        45.0,
        10.0,
        max_points=20,
    )

    assert grid == (
        10.0,
        20.0,
        30.0,
        40.0,
        45.0,
    )


def test_scan_requires_cup3():
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
            tracker(),
            policies(),
        )


def test_nominal_commands_use_source_readbacks_and_not_old_cooler_offset(
    monkeypatch,
):
    adapter = FakeAdapter()

    current = state(
        cooler=27000.0
    )

    # Old cooler readback intentionally has a large offset.
    current.readbacks[
        "ion_cooler_voltage_v"
    ] = 26000.0

    calls = []

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        command = target.parameters[
            "ion_cooler_voltage_v"
        ]

        calls.append(
            command
        )

        # Simulate a persistent 60 V command/readback offset.
        observed = MachineState(
            mass_u=60.0,
            cup=3,
            stage=3,
            parameters=(
                target.parameters.copy()
            ),
            readbacks={
                "sputter_voltage_v": 7900.0,
                "extraction_voltage_v": 18600.0,
                "ion_cooler_voltage_v": (
                    command
                    - 60.0
                ),
            },
        )

        return AppliedStateResult(
            requested_state=target,
            observed_state=observed,
            plan=StateTransitionPlan(
                source_state_id=(
                    current.state_id
                ),
                target_state_id=(
                    target.state_id
                ),
                changes=(),
            ),
            settling_results=(),
            selected_cup=None,
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    measurements = iter(
        [
            measurement(5e-9),
            measurement(6e-9),
            measurement(7e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    module.scan_residual_energy(
        adapter,
        current,
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        module.ResidualEnergyScanPolicy(
            minimum_ev=30.0,
            maximum_ev=40.0,
            steps_ev=(10.0,),
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_relative_improvement=0.0,
        ),
    )

    # Source readbacks:
    # 7900 + 18600 = 26500 eV.
    #
    # 30 eV target -> 26470 V
    # 40 eV target -> 26460 V
    #
    # The old ~1000 V cooler offset is NOT compensated.
    assert calls == pytest.approx(
        [
            26470.0,
            26460.0,
            26460.0,
        ]
    )


def test_observed_residual_energy_comes_from_settled_readback(
    monkeypatch,
):
    adapter = FakeAdapter()

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        command = target.parameters[
            "ion_cooler_voltage_v"
        ]

        observed = MachineState(
            mass_u=60.0,
            cup=3,
            stage=3,
            parameters=(
                target.parameters.copy()
            ),
            readbacks={
                "sputter_voltage_v": 7900.0,
                "extraction_voltage_v": 18600.0,
                "ion_cooler_voltage_v": (
                    command
                    - 55.0
                ),
            },
        )

        return AppliedStateResult(
            requested_state=target,
            observed_state=observed,
            plan=StateTransitionPlan(
                source_state_id=(
                    current.state_id
                ),
                target_state_id=(
                    target.state_id
                ),
                changes=(),
            ),
            settling_results=(),
            selected_cup=None,
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    measurements = iter(
        [
            measurement(5e-9),
            measurement(8e-9),
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

    result = module.scan_residual_energy(
        adapter,
        state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        module.ResidualEnergyScanPolicy(
            minimum_ev=40.0,
            maximum_ev=50.0,
            steps_ev=(10.0,),
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_relative_improvement=0.0,
        ),
    )

    first = result.points[0]

    assert (
        first.target_residual_energy_ev
        == 40.0
    )

    assert (
        first.nominal_cooler_command_v
        == pytest.approx(
            26460.0
        )
    )

    # Readback = 26405 V.
    # Observed Eres = 26500 - 26405 = 95 eV.
    assert (
        first.energy_state.residual_energy_best_available_ev
        == pytest.approx(
            95.0
        )
    )


def test_best_point_is_chosen_by_transmission(
    monkeypatch,
):
    adapter = FakeAdapter()

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        return AppliedStateResult(
            requested_state=target,
            observed_state=target,
            plan=StateTransitionPlan(
                source_state_id=(
                    current.state_id
                ),
                target_state_id=(
                    target.state_id
                ),
                changes=(),
            ),
            settling_results=(),
            selected_cup=None,
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    measurements = iter(
        [
            # Baseline
            measurement(5e-9),

            # 20 eV
            measurement(6e-9),

            # 30 eV
            measurement(9e-9),

            # 40 eV
            measurement(7e-9),
        ]
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: next(
            measurements
        ),
    )

    result = module.scan_residual_energy(
        adapter,
        state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        module.ResidualEnergyScanPolicy(
            minimum_ev=20.0,
            maximum_ev=40.0,
            steps_ev=(10.0,),
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert (
        result.best_target_residual_energy_ev
        == pytest.approx(
            30.0
        )
    )

    assert (
        result.best_transmission.transmission
        == pytest.approx(
            0.9
        )
    )


def test_coarse_to_fine_refines_around_best_target(
    monkeypatch,
):
    adapter = FakeAdapter()

    commanded_targets = []

    original_prediction = (
        module.nominal_cooler_command_for_residual_energy
    )

    def wrapped_prediction(
        current_state,
        target_energy,
    ):
        commanded_targets.append(
            target_energy
        )

        return original_prediction(
            current_state,
            target_energy,
        )

    monkeypatch.setattr(
        module,
        "nominal_cooler_command_for_residual_energy",
        wrapped_prediction,
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            AppliedStateResult(
                requested_state=target,
                observed_state=target,
                plan=StateTransitionPlan(
                    source_state_id=current.state_id,
                    target_state_id=target.state_id,
                    changes=(),
                ),
                settling_results=(),
                selected_cup=None,
            ),
    )

    def response_for_target(
        target_energy,
    ):
        return max(
            1e-12,
            10e-9
            - (
                target_energy
                - 42.0
            ) ** 2
            * 0.02e-9,
        )

    measurement_counter = {
        "index": 0
    }

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        index = measurement_counter[
            "index"
        ]

        measurement_counter[
            "index"
        ] += 1

        if index == 0:
            return measurement(
                1e-9
            )

        target = commanded_targets[
            index - 1
        ]

        return measurement(
            response_for_target(
                target
            )
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    result = module.scan_residual_energy(
        adapter,
        state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        module.ResidualEnergyScanPolicy(
            minimum_ev=20.0,
            maximum_ev=70.0,
            steps_ev=(
                10.0,
                1.0,
            ),
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert (
        result.best_target_residual_energy_ev
        == pytest.approx(
            42.0
        )
    )


def test_all_tested_points_are_preserved(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            AppliedStateResult(
                requested_state=target,
                observed_state=target,
                plan=StateTransitionPlan(
                    source_state_id=current.state_id,
                    target_state_id=target.state_id,
                    changes=(),
                ),
                settling_results=(),
                selected_cup=None,
            ),
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    result = module.scan_residual_energy(
        FakeAdapter(),
        state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        module.ResidualEnergyScanPolicy(
            minimum_ev=20.0,
            maximum_ev=40.0,
            steps_ev=(10.0,),
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
    )

    assert [
        point.target_residual_energy_ev
        for point in result.points
    ] == [
        20.0,
        30.0,
        40.0,
    ]


def test_below_noise_candidate_is_not_selected_over_real_beam(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            AppliedStateResult(
                requested_state=target,
                observed_state=target,
                plan=StateTransitionPlan(
                    source_state_id=current.state_id,
                    target_state_id=target.state_id,
                    changes=(),
                ),
                settling_results=(),
                selected_cup=None,
            ),
    )

    measurements = iter(
        [
            measurement(
                5e-9
            ),
            measurement(
                0.1e-12,
                below_noise=True,
            ),
            measurement(
                0.2e-12,
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

    result = module.scan_residual_energy(
        FakeAdapter(),
        state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        module.ResidualEnergyScanPolicy(
            minimum_ev=30.0,
            maximum_ev=40.0,
            steps_ev=(10.0,),
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(
            uncertainty_multiple=0.0,
            minimum_relative_improvement=0.0,
        ),
    )

    assert (
        result.best_measurement.mean_a
        == pytest.approx(
            5e-9
        )
    )


def test_maintenance_hook_must_restore_cup3(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    def bad_hook(
        current_state
    ):
        changed = state(
            cup=2
        )

        return changed

    with pytest.raises(
        ValueError
    ):
        module.scan_residual_energy(
            FakeAdapter(),
            state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            module.ResidualEnergyScanPolicy(
                minimum_ev=30.0,
                maximum_ev=40.0,
                steps_ev=(10.0,),
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            maintenance_hook=(
                bad_hook
            ),
        )