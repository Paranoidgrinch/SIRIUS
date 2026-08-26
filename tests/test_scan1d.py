import math

import pytest

import sirius.scan1d as module
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
from sirius.scan1d import (
    ScanPolicy,
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
        self.command = None


def state(
    command=4000.0,
):
    return MachineState(
        mass_u=60.0,
        cup=1,
        stage=1,
        role="working",
        parameters={
            "sputter_voltage_v": command,
        },
    )


def measurement(
    value,
):
    return BeamMeasurement(
        mean_a=value,
        sigma_a=1e-12,
        sem_a=1e-13,
        n=10,
        duration_s=0.5,
        relative_sem=None,
        precision_threshold_a=1e-12,
        drift_delta_a=0.0,
        stop_reason="test",
        below_noise_floor=False,
        samples=(),
    )


def fake_applied(
    target,
):
    return AppliedStateResult(
        requested_state=target,
        observed_state=target,
        plan=StateTransitionPlan(
            source_state_id="source",
            target_state_id=target.state_id,
            changes=(),
        ),
        settling_results=(),
        selected_cup=None,
    )


def policies():
    return {
        "sputter_voltage_v": (
            SettlingPolicy(
                max_readback_span=5.0,
            )
        )
    }


def test_scan_policy_requires_decreasing_steps():
    with pytest.raises(
        ValueError
    ):
        ScanPolicy(
            steps=(
                100.0,
                200.0,
            )
        )


def test_scan_policy_rejects_zero_step():
    with pytest.raises(
        ValueError
    ):
        ScanPolicy(
            steps=(
                100.0,
                0.0,
            )
        )


def test_grid_includes_upper_boundary():
    grid = module._generate_grid(
        0.0,
        9000.0,
        2000.0,
        max_points=20,
    )

    assert grid == (
        0.0,
        2000.0,
        4000.0,
        6000.0,
        8000.0,
        9000.0,
    )


def test_grid_has_point_limit():
    with pytest.raises(
        ValueError
    ):
        module._generate_grid(
            0.0,
            9000.0,
            1.0,
            max_points=100,
        )


def test_learned_mass_profile_bounds_are_used():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "sputter_voltage_v",
        5000.0,
        9000.0,
    )

    minimum, maximum = (
        module._resolve_effective_bounds(
            profile,
            "sputter_voltage_v",
            6000.0,
        )
    )

    assert minimum == 5000.0
    assert maximum == 9000.0


def test_current_command_is_not_silently_excluded_by_old_learned_bounds():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "sputter_voltage_v",
        5000.0,
        9000.0,
    )

    minimum, maximum = (
        module._resolve_effective_bounds(
            profile,
            "sputter_voltage_v",
            4500.0,
        )
    )

    assert minimum == 4500.0
    assert maximum == 9000.0


def test_coarse_to_fine_scan_finds_peak(
    monkeypatch,
):
    adapter = FakeAdapter()

    current = state(
        4000.0
    )

    profile = MassProfile(
        mass_u=60.0
    )

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        adapter.command = (
            target.parameters[
                "sputter_voltage_v"
            ]
        )

        return fake_applied(
            target
        )

    def response(command):
        # Smooth maximum at 6500 V.
        return max(
            1e-12,
            10e-9
            - (
                (command - 6500.0)
                / 2000.0
            ) ** 2
            * 4e-9,
        )

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        command = (
            4000.0
            if adapter.command is None
            else adapter.command
        )

        return measurement(
            response(command)
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    result = (
        module.scan_parameter_1d(
            adapter,
            current,
            profile,
            "sputter_voltage_v",
            ScanPolicy(
                steps=(
                    2000.0,
                    500.0,
                )
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(
                uncertainty_multiple=2.0,
                minimum_relative_improvement=0.0,
            ),
        )
    )

    assert result.best_command == pytest.approx(
        6500.0
    )

    assert result.improved is True

    assert (
        result.final_state.parameters[
            "sputter_voltage_v"
        ]
        == pytest.approx(
            6500.0
        )
    )


def test_flat_response_keeps_initial_best(
    monkeypatch,
):
    adapter = FakeAdapter()

    current = state(
        6000.0
    )

    profile = MassProfile(
        mass_u=60.0
    )

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        adapter.command = (
            target.parameters[
                "sputter_voltage_v"
            ]
        )

        return fake_applied(
            target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda adapter, policy, noise_floor_a=None:
            measurement(
                10e-9
            ),
    )

    result = (
        module.scan_parameter_1d(
            adapter,
            current,
            profile,
            "sputter_voltage_v",
            ScanPolicy(
                steps=(
                    3000.0,
                    500.0,
                )
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(
                uncertainty_multiple=2.0,
                minimum_relative_improvement=0.001,
            ),
        )
    )

    assert result.best_command == 6000.0
    assert result.improved is False


def test_disabled_parameter_cannot_be_scanned():
    current = MachineState(
        mass_u=60.0,
        cup=3,
        parameters={
            "hv2_voltage_v": 1000.0,
        },
    )

    profile = MassProfile(
        mass_u=60.0
    )

    with pytest.raises(
        ValueError
    ):
        module.scan_parameter_1d(
            FakeAdapter(),
            current,
            profile,
            "hv2_voltage_v",
            ScanPolicy(
                steps=(100.0,)
            ),
            {
                "hv2_voltage_v": (
                    SettlingPolicy(
                        max_readback_span=5.0
                    )
                )
            },
            MeasurementPolicy(),
            ComparisonPolicy(),
        )


def test_state_and_profile_mass_must_match():
    current = state(
        6000.0
    )

    profile = MassProfile(
        mass_u=180.0
    )

    with pytest.raises(
        ValueError
    ):
        module.scan_parameter_1d(
            FakeAdapter(),
            current,
            profile,
            "sputter_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
        )


def test_maintenance_hook_must_restore_same_cup(
    monkeypatch,
):
    adapter = FakeAdapter()

    current = state(
        6000.0
    )

    profile = MassProfile(
        mass_u=60.0
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda adapter, policy, noise_floor_a=None:
            measurement(
                10e-9
            ),
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            fake_applied(target),
    )

    def bad_hook(
        current_state
    ):
        return MachineState(
            mass_u=60.0,
            cup=2,
            parameters=(
                current_state.parameters.copy()
            ),
        )

    with pytest.raises(
        ValueError
    ):
        module.scan_parameter_1d(
            adapter,
            current,
            profile,
            "sputter_voltage_v",
            ScanPolicy(
                steps=(3000.0,)
            ),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            maintenance_hook=bad_hook,
        )