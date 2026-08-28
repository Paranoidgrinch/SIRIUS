import pytest

import sirius.qpt_scan2d as module
from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.qpt_model import (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
    evaluate_qpt,
    qpt_cfa_is_feasible,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState
from sirius.transition import (
    AppliedStateResult,
    StateTransitionPlan,
)


class FakeAdapter:
    def __init__(
        self,
        state=None,
    ):
        self.state = state
        self.targets = []


def measurement(
    value,
    *,
    sem=1e-12,
    below_noise=False,
):
    return BeamMeasurement(
        mean_a=value,
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


def tracker(
    current=10e-9,
):
    result = SourceReferenceTracker()

    result.add(
        SourceReference(
            measurement=measurement(
                current
            ),
            state_id="cup1-reference",
            mass_u=60.0,
            monotonic_s=0.0,
            created_at_utc=(
                "2026-08-28T07:00:00+00:00"
            ),
        )
    )

    return result


def state(
    *,
    qpt1=2000.0,
    qpt2=3000.0,
    qpt3=2000.0,
    cup=4,
    readbacks=None,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        stage=4,
        parameters={
            QPT1_PARAMETER: qpt1,
            QPT2_PARAMETER: qpt2,
            QPT3_PARAMETER: qpt3,
        },
        readbacks=(
            {}
            if readbacks is None
            else readbacks
        ),
    )


def policies():
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        QPT1_PARAMETER: policy,
        QPT2_PARAMETER: policy,
        QPT3_PARAMETER: policy,
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


def patch_apply(
    monkeypatch,
    adapter,
):
    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        adapter.targets.append(
            target
        )

        adapter.state = (
            target
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


def small_policy():
    return module.QPT2DScanPolicy(
        initial_focus_half_width_v=500.0,
        initial_asymmetry_half_width_v=500.0,
        levels=(
            module.QPTScanLevel(
                250.0,
                250.0,
            ),
            module.QPTScanLevel(
                50.0,
                50.0,
            ),
        ),
        refinement_half_width_factor=2.0,
        max_points_per_level=500,
    )


def test_requires_cup4():
    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            state(
                cup=3
            ),
            tracker(),
            policies(),
        )


def test_axis_grid_includes_upper_boundary():
    assert module._axis_grid(
        0.0,
        550.0,
        250.0,
    ) == (
        0.0,
        250.0,
        500.0,
        550.0,
    )


def test_feasible_grid_contains_only_valid_psu_states():
    points = module._feasible_grid(
        common_v=3000.0,
        focus_minimum=-4000.0,
        focus_maximum=4000.0,
        asymmetry_minimum=-3000.0,
        asymmetry_maximum=3000.0,
        focus_step=500.0,
        asymmetry_step=500.0,
        max_points=500,
        start_focus=1000.0,
        start_asymmetry=0.0,
    )

    assert len(
        points
    ) > 0

    assert all(
        qpt_cfa_is_feasible(
            3000.0,
            focus,
            asymmetry,
        )
        for focus, asymmetry
        in points
    )


def test_qpt_common_mode_remains_frozen(
    monkeypatch,
):
    current = state()

    adapter = FakeAdapter(
        current
    )

    patch_apply(
        monkeypatch,
        adapter,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    result = (
        module.scan_qpt_focus_asymmetry_2d(
            adapter,
            current,
            tracker(),
            small_policy(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
        )
    )

    assert result.common_v == pytest.approx(
        3000.0
    )

    for target in adapter.targets:
        coordinates = (
            evaluate_qpt(
                target
            ).command_coordinates
        )

        assert coordinates.common_v == pytest.approx(
            3000.0
        )

        assert (
            target.parameters[
                QPT2_PARAMETER
            ]
            == pytest.approx(
                3000.0
            )
        )


def test_scanner_finds_two_dimensional_optimum(
    monkeypatch,
):
    current = state(
        # C = 3000
        # F = 1000
        # A = 0
        qpt1=2000.0,
        qpt2=3000.0,
        qpt3=2000.0,
    )

    adapter = FakeAdapter(
        current
    )

    patch_apply(
        monkeypatch,
        adapter,
    )

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        coordinates = (
            evaluate_qpt(
                adapter.state
            ).command_coordinates
        )

        focus = (
            coordinates.global_focus_v
        )

        asymmetry = (
            coordinates.asymmetry_v
        )

        # Artificial optimum:
        #
        # F = 1250 V
        # A = -250 V
        response = (
            10e-9
            - (
                focus
                - 1250.0
            ) ** 2
            * 2e-15
            - (
                asymmetry
                + 250.0
            ) ** 2
            * 2e-15
        )

        return measurement(
            max(
                response,
                1e-12,
            )
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    result = (
        module.scan_qpt_focus_asymmetry_2d(
            adapter,
            current,
            tracker(),
            module.QPT2DScanPolicy(
                initial_focus_half_width_v=500.0,
                initial_asymmetry_half_width_v=500.0,
                levels=(
                    module.QPTScanLevel(
                        250.0,
                        250.0,
                    ),
                    module.QPTScanLevel(
                        50.0,
                        50.0,
                    ),
                ),
                refinement_half_width_factor=2.0,
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(
                uncertainty_multiple=0.0,
                minimum_relative_improvement=0.0,
            ),
        )
    )

    assert (
        result.best_target_focus_v
        == pytest.approx(
            1250.0
        )
    )

    assert (
        result.best_target_asymmetry_v
        == pytest.approx(
            -250.0
        )
    )

    final = evaluate_qpt(
        result.final_state
    ).command_coordinates

    assert (
        final.global_focus_v
        == pytest.approx(
            1250.0
        )
    )

    assert (
        final.asymmetry_v
        == pytest.approx(
            -250.0
        )
    )


def test_invalid_points_never_reach_apply_state(
    monkeypatch,
):
    current = state()

    adapter = FakeAdapter(
        current
    )

    patch_apply(
        monkeypatch,
        adapter,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    module.scan_qpt_focus_asymmetry_2d(
        adapter,
        current,
        tracker(),
        module.QPT2DScanPolicy(
            initial_focus_half_width_v=5000.0,
            initial_asymmetry_half_width_v=5000.0,
            levels=(
                module.QPTScanLevel(
                    1000.0,
                    1000.0,
                ),
            ),
            max_points_per_level=100,
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
    )

    for target in adapter.targets:
        for parameter_name in (
            QPT1_PARAMETER,
            QPT2_PARAMETER,
            QPT3_PARAMETER,
        ):
            value = target.parameters[
                parameter_name
            ]

            assert 0.0 <= value <= 6000.0


def test_readback_coordinates_are_kept_separate_from_targets(
    monkeypatch,
):
    current = state()

    adapter = FakeAdapter(
        current
    )

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        parameters = dict(
            target.parameters
        )

        observed = MachineState(
            mass_u=60.0,
            cup=4,
            stage=4,
            parameters=parameters,
            readbacks={
                QPT1_PARAMETER: (
                    parameters[
                        QPT1_PARAMETER
                    ]
                    - 50.0
                ),
                QPT2_PARAMETER: (
                    parameters[
                        QPT2_PARAMETER
                    ]
                    - 25.0
                ),
                QPT3_PARAMETER: (
                    parameters[
                        QPT3_PARAMETER
                    ]
                    - 75.0
                ),
            },
        )

        adapter.state = (
            observed
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

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                6e-9
            ),
    )

    result = (
        module.scan_qpt_focus_asymmetry_2d(
            adapter,
            current,
            tracker(),
            module.QPT2DScanPolicy(
                initial_focus_half_width_v=250.0,
                initial_asymmetry_half_width_v=250.0,
                levels=(
                    module.QPTScanLevel(
                        250.0,
                        250.0,
                    ),
                ),
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
        )
    )

    point = result.points[0]

    assert (
        point.observed_coordinates
        != evaluate_qpt(
            point.state
        ).command_coordinates
    )


def test_all_tested_points_are_preserved(
    monkeypatch,
):
    current = state()

    adapter = FakeAdapter(
        current
    )

    patch_apply(
        monkeypatch,
        adapter,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    result = (
        module.scan_qpt_focus_asymmetry_2d(
            adapter,
            current,
            tracker(),
            small_policy(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
        )
    )

    coordinates = {
        (
            point.target_focus_v,
            point.target_asymmetry_v,
        )
        for point
        in result.points
    }

    assert len(
        coordinates
    ) == len(
        result.points
    )


def test_maintenance_occurs_before_baseline_measurement(
    monkeypatch,
):
    calls = []

    current = state()

    adapter = FakeAdapter(
        current
    )

    patch_apply(
        monkeypatch,
        adapter,
    )

    def maintenance(
        current_state
    ):
        calls.append(
            "maintenance"
        )

        return current_state

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
        module,
        "measure_beam_current",
        fake_measure,
    )

    module.scan_qpt_focus_asymmetry_2d(
        adapter,
        current,
        tracker(),
        module.QPT2DScanPolicy(
            initial_focus_half_width_v=250.0,
            initial_asymmetry_half_width_v=250.0,
            levels=(
                module.QPTScanLevel(
                    250.0,
                    250.0,
                ),
            ),
        ),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        maintenance_hook=(
            maintenance
        ),
    )

    assert calls[0:2] == [
        "maintenance",
        "measure",
    ]


def test_maintenance_may_not_change_common_mode(
    monkeypatch,
):
    current = state()

    adapter = FakeAdapter(
        current
    )

    patch_apply(
        monkeypatch,
        adapter,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    count = {
        "n": 0
    }

    def maintenance(
        current_state
    ):
        count["n"] += 1

        if count["n"] == 1:
            return current_state

        parameters = dict(
            current_state.parameters
        )

        parameters[
            QPT2_PARAMETER
        ] += 100.0

        return MachineState(
            mass_u=60.0,
            cup=4,
            stage=4,
            parameters=parameters,
        )

    with pytest.raises(
        ValueError,
        match="common mode",
    ):
        module.scan_qpt_focus_asymmetry_2d(
            adapter,
            current,
            tracker(),
            module.QPT2DScanPolicy(
                initial_focus_half_width_v=250.0,
                initial_asymmetry_half_width_v=250.0,
                levels=(
                    module.QPTScanLevel(
                        250.0,
                        250.0,
                    ),
                ),
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            maintenance_hook=(
                maintenance
            ),
        )


def test_nearest_neighbor_order_starts_near_current_point():
    points = (
        (
            -1000.0,
            -1000.0,
        ),
        (
            100.0,
            0.0,
        ),
        (
            500.0,
            500.0,
        ),
    )

    ordered = (
        module._ordered_nearest_neighbor(
            points,
            start_focus=0.0,
            start_asymmetry=0.0,
            focus_scale=100.0,
            asymmetry_scale=100.0,
        )
    )

    assert ordered[0] == (
        100.0,
        0.0,
    )

def test_feasible_grid_enforces_max_points_guard():
    with pytest.raises(
        ValueError,
        match="max_points_per_level",
    ):
        module._feasible_grid(
            common_v=3000.0,
            focus_minimum=500.0,
            focus_maximum=1500.0,
            asymmetry_minimum=-500.0,
            asymmetry_maximum=500.0,
            focus_step=50.0,
            asymmetry_step=50.0,
            max_points=100,
            start_focus=1000.0,
            start_asymmetry=0.0,
        )