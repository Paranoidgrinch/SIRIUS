import pytest

import sirius.derived_scan1d as scan_module
from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.coupled_coordinates import (
    end_electrode_common_builder,
    end_electrode_common_command,
    end_electrode_difference_builder,
    end_electrode_difference_command,
    guidefield_common_builder,
    guidefield_common_bounds,
    guidefield_common_command,
    guidefield_difference_bounds,
    guidefield_difference_builder,
    guidefield_difference_command,
)
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.scan1d import ScanPolicy
from sirius.settling import (
    SettlingPolicy,
)
from sirius.state import MachineState
from sirius.transition import (
    AppliedStateResult,
    StateTransitionPlan,
)


class FakeAdapter:
    def __init__(self):
        self.state = None


def measurement(
    value,
):
    return BeamMeasurement(
        mean_a=value,
        sigma_a=1e-12,
        sem_a=1e-12,
        n=10,
        duration_s=0.5,
        relative_sem=None,
        precision_threshold_a=1e-12,
        drift_delta_a=0.0,
        stop_reason="test",
        below_noise_floor=False,
        samples=(),
    )


def tracker():
    result = SourceReferenceTracker()

    result.add(
        SourceReference(
            measurement=measurement(
                10e-9
            ),
            state_id="cup1-reference",
            mass_u=60.0,
            monotonic_s=0.0,
            created_at_utc=(
                "2026-08-27T14:00:00+00:00"
            ),
        )
    )

    return result


def guidefield_state(
    *,
    gf1=10.0,
    gf2=20.0,
):
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "guidefield1_voltage_v": gf1,
            "guidefield2_voltage_v": gf2,
        },
    )


def electrode_state(
    *,
    hv1=1200.0,
    hv4=800.0,
):
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "deceleration_voltage_v": hv1,
            "acceleration_voltage_v": hv4,
        },
    )


def policies(
    *names,
):
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        name: policy
        for name in names
    }


def applied(
    source,
    target,
):
    return AppliedStateResult(
        requested_state=target,
        observed_state=target,
        plan=StateTransitionPlan(
            source_state_id=(
                source.state_id
            ),
            target_state_id=(
                target.state_id
            ),
            changes=(),
        ),
        settling_results=(),
        selected_cup=None,
    )


def test_guidefield_difference_builder_preserves_common_mode():
    current = guidefield_state(
        gf1=10.0,
        gf2=20.0,
    )

    commands = (
        guidefield_difference_builder(
            current,
            10.0,
        )
    )

    assert commands[
        "guidefield1_voltage_v"
    ] == pytest.approx(
        20.0
    )

    assert commands[
        "guidefield2_voltage_v"
    ] == pytest.approx(
        10.0
    )

    assert (
        (
            commands[
                "guidefield1_voltage_v"
            ]
            + commands[
                "guidefield2_voltage_v"
            ]
        )
        / 2.0
        == pytest.approx(
            15.0
        )
    )


def test_guidefield_common_builder_preserves_difference():
    current = guidefield_state(
        gf1=20.0,
        gf2=10.0,
    )

    commands = (
        guidefield_common_builder(
            current,
            20.0,
        )
    )

    assert (
        commands[
            "guidefield1_voltage_v"
        ]
        - commands[
            "guidefield2_voltage_v"
        ]
        == pytest.approx(
            10.0
        )
    )

    assert (
        (
            commands[
                "guidefield1_voltage_v"
            ]
            + commands[
                "guidefield2_voltage_v"
            ]
        )
        / 2.0
        == pytest.approx(
            20.0
        )
    )


def test_guidefield_bounds_respect_30v_gf1_limit():
    current = guidefield_state(
        gf1=10.0,
        gf2=20.0,
    )

    minimum, maximum = (
        guidefield_difference_bounds(
            current
        )
    )

    assert minimum == pytest.approx(
        -30.0
    )

    assert maximum == pytest.approx(
        30.0
    )


def test_guidefield_common_bounds_are_feasible():
    current = guidefield_state(
        gf1=20.0,
        gf2=10.0,
    )

    minimum, maximum = (
        guidefield_common_bounds(
            current
        )
    )

    assert minimum == pytest.approx(
        5.0
    )

    assert maximum == pytest.approx(
        25.0
    )


def test_end_electrode_difference_builder_preserves_common():
    current = electrode_state(
        hv1=1200.0,
        hv4=800.0,
    )

    commands = (
        end_electrode_difference_builder(
            current,
            -400.0,
        )
    )

    assert commands[
        "deceleration_voltage_v"
    ] == pytest.approx(
        800.0
    )

    assert commands[
        "acceleration_voltage_v"
    ] == pytest.approx(
        1200.0
    )

    assert (
        (
            commands[
                "deceleration_voltage_v"
            ]
            + commands[
                "acceleration_voltage_v"
            ]
        )
        / 2.0
        == pytest.approx(
            1000.0
        )
    )


def test_end_electrode_common_builder_preserves_difference():
    current = electrode_state(
        hv1=1200.0,
        hv4=800.0,
    )

    commands = (
        end_electrode_common_builder(
            current,
            2000.0,
        )
    )

    assert (
        commands[
            "deceleration_voltage_v"
        ]
        - commands[
            "acceleration_voltage_v"
        ]
        == pytest.approx(
            400.0
        )
    )

    assert (
        (
            commands[
                "deceleration_voltage_v"
            ]
            + commands[
                "acceleration_voltage_v"
            ]
        )
        / 2.0
        == pytest.approx(
            2000.0
        )
    )


def test_generic_derived_scan_finds_best_difference(
    monkeypatch,
):
    adapter = FakeAdapter()

    current = guidefield_state(
        gf1=10.0,
        gf2=10.0,
    )

    adapter.state = current

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        adapter.state = target

        return applied(
            current,
            target,
        )

    monkeypatch.setattr(
        scan_module,
        "apply_state",
        fake_apply,
    )

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        current_state = (
            adapter.state
        )

        difference = (
            guidefield_difference_command(
                current_state
            )
        )

        # Artificial maximum at delta = -10 V.
        response = max(
            1e-12,
            10e-9
            - (
                difference
                + 10.0
            ) ** 2
            * 0.02e-9,
        )

        return measurement(
            response
        )

    monkeypatch.setattr(
        scan_module,
        "measure_beam_current",
        fake_measure,
    )

    result = (
        scan_module.scan_derived_coordinate_transmission_1d(
            adapter,
            current,
            tracker(),
            coordinate_name=(
                "guidefield_difference_v"
            ),
            minimum=-20.0,
            maximum=20.0,
            coordinate_reader=(
                guidefield_difference_command
            ),
            command_builder=(
                guidefield_difference_builder
            ),
            affected_parameters=(
                "guidefield1_voltage_v",
                "guidefield2_voltage_v",
            ),
            scan_policy=ScanPolicy(
                steps=(
                    10.0,
                    2.0,
                )
            ),
            settling_policies=policies(
                "guidefield1_voltage_v",
                "guidefield2_voltage_v",
            ),
            measurement_policy=(
                MeasurementPolicy()
            ),
            comparison_policy=(
                ComparisonPolicy(
                    uncertainty_multiple=0.0,
                    minimum_relative_improvement=0.0,
                )
            ),
        )
    )

    assert (
        result.best_coordinate
        == pytest.approx(
            -10.0
        )
    )

    assert (
        guidefield_difference_command(
            result.final_state
        )
        == pytest.approx(
            -10.0
        )
    )


def test_both_physical_parameters_change_together(
    monkeypatch,
):
    adapter = FakeAdapter()

    current = guidefield_state(
        gf1=10.0,
        gf2=10.0,
    )

    adapter.state = current

    targets = []

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        targets.append(
            (
                target.parameters[
                    "guidefield1_voltage_v"
                ],
                target.parameters[
                    "guidefield2_voltage_v"
                ],
            )
        )

        adapter.state = target

        return applied(
            current,
            target,
        )

    monkeypatch.setattr(
        scan_module,
        "apply_state",
        fake_apply,
    )

    monkeypatch.setattr(
        scan_module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    scan_module.scan_derived_coordinate_transmission_1d(
        adapter,
        current,
        tracker(),
        coordinate_name=(
            "guidefield_difference_v"
        ),
        minimum=-10.0,
        maximum=10.0,
        coordinate_reader=(
            guidefield_difference_command
        ),
        command_builder=(
            guidefield_difference_builder
        ),
        affected_parameters=(
            "guidefield1_voltage_v",
            "guidefield2_voltage_v",
        ),
        scan_policy=ScanPolicy(
            steps=(10.0,)
        ),
        settling_policies=policies(
            "guidefield1_voltage_v",
            "guidefield2_voltage_v",
        ),
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
    )

    assert (
        5.0,
        15.0,
    ) in targets

    assert (
        15.0,
        5.0,
    ) in targets


def test_builder_must_return_exact_affected_parameters(
    monkeypatch,
):
    current = guidefield_state()

    monkeypatch.setattr(
        scan_module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    def bad_builder(
        state,
        coordinate,
    ):
        return {
            "guidefield1_voltage_v": 10.0,
        }

    with pytest.raises(
        ValueError
    ):
        scan_module.scan_derived_coordinate_transmission_1d(
            FakeAdapter(),
            current,
            tracker(),
            coordinate_name="bad",
            minimum=-20.0,
            maximum=20.0,
            coordinate_reader=(
                guidefield_difference_command
            ),
            command_builder=(
                bad_builder
            ),
            affected_parameters=(
                "guidefield1_voltage_v",
                "guidefield2_voltage_v",
            ),
            scan_policy=ScanPolicy(
                steps=(10.0,)
            ),
            settling_policies=policies(
                "guidefield1_voltage_v",
                "guidefield2_voltage_v",
            ),
            measurement_policy=(
                MeasurementPolicy()
            ),
            comparison_policy=(
                ComparisonPolicy()
            ),
        )


def test_initial_coordinate_must_be_inside_window(
    monkeypatch,
):
    monkeypatch.setattr(
        scan_module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    with pytest.raises(
        ValueError
    ):
        scan_module.scan_derived_coordinate_transmission_1d(
            FakeAdapter(),
            guidefield_state(
                gf1=20.0,
                gf2=10.0,
            ),
            tracker(),
            coordinate_name=(
                "guidefield_difference_v"
            ),
            minimum=-5.0,
            maximum=5.0,
            coordinate_reader=(
                guidefield_difference_command
            ),
            command_builder=(
                guidefield_difference_builder
            ),
            affected_parameters=(
                "guidefield1_voltage_v",
                "guidefield2_voltage_v",
            ),
            scan_policy=ScanPolicy(
                steps=(1.0,)
            ),
            settling_policies=policies(
                "guidefield1_voltage_v",
                "guidefield2_voltage_v",
            ),
            measurement_policy=(
                MeasurementPolicy()
            ),
            comparison_policy=(
                ComparisonPolicy()
            ),
        )


def test_maintenance_runs_before_baseline_measurement(
    monkeypatch,
):
    calls = []

    current = guidefield_state(
        gf1=10.0,
        gf2=10.0,
    )

    def maintenance(
        state
    ):
        calls.append(
            "maintenance"
        )

        return state

    def fake_measure(
        *args,
        **kwargs,
    ):
        calls.append(
            "measure"
        )

        return measurement(
            5e-9
        )

    monkeypatch.setattr(
        scan_module,
        "measure_beam_current",
        fake_measure,
    )

    monkeypatch.setattr(
        scan_module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            applied(
                current,
                target
            ),
    )

    scan_module.scan_derived_coordinate_transmission_1d(
        FakeAdapter(),
        current,
        tracker(),
        coordinate_name=(
            "guidefield_difference_v"
        ),
        minimum=-10.0,
        maximum=10.0,
        coordinate_reader=(
            guidefield_difference_command
        ),
        command_builder=(
            guidefield_difference_builder
        ),
        affected_parameters=(
            "guidefield1_voltage_v",
            "guidefield2_voltage_v",
        ),
        scan_policy=ScanPolicy(
            steps=(10.0,)
        ),
        settling_policies=policies(
            "guidefield1_voltage_v",
            "guidefield2_voltage_v",
        ),
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        maintenance_hook=(
            maintenance
        ),
    )

    assert calls[0:2] == [
        "maintenance",
        "measure",
    ]


def test_coordinate_readers():
    gf = guidefield_state(
        gf1=20.0,
        gf2=10.0,
    )

    assert (
        guidefield_difference_command(
            gf
        )
        == 10.0
    )

    assert (
        guidefield_common_command(
            gf
        )
        == 15.0
    )

    hv = electrode_state(
        hv1=1200.0,
        hv4=800.0,
    )

    assert (
        end_electrode_difference_command(
            hv
        )
        == 400.0
    )

    assert (
        end_electrode_common_command(
            hv
        )
        == 1000.0
    )