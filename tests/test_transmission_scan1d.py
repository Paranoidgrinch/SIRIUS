import pytest

import sirius.transmission_scan1d as module
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


def measurement(
    value,
    sem=1e-12,
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
        below_noise_floor=False,
        samples=(),
    )


def reference(
    current,
    *,
    state_id="ref",
    time=0.0,
):
    return SourceReference(
        measurement=measurement(
            current,
            sem=1e-12,
        ),
        state_id=state_id,
        mass_u=60.0,
        monotonic_s=time,
        created_at_utc=(
            "2026-08-26T15:00:00+00:00"
        ),
    )


def state(
    lens2=5000.0,
    *,
    cup=2,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        stage=2,
        role="working",
        parameters={
            "lens2_voltage_v": lens2,
        },
    )


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


def settling_policies():
    return {
        "lens2_voltage_v": (
            SettlingPolicy(
                max_readback_span=5.0
            )
        )
    }


def test_transmission_scan_requires_reference():
    tracker = SourceReferenceTracker()

    with pytest.raises(
        ValueError
    ):
        module.scan_parameter_transmission_1d(
            FakeAdapter(),
            state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker,
            "lens2_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            settling_policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
        )


def test_transmission_scan_requires_downstream_cup():
    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9
        )
    )

    with pytest.raises(
        ValueError
    ):
        module.scan_parameter_transmission_1d(
            FakeAdapter(),
            state(
                cup=1
            ),
            MassProfile(
                mass_u=60.0
            ),
            tracker,
            "lens2_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            settling_policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
        )


def test_transmission_scan_finds_higher_normalized_transmission(
    monkeypatch,
):
    adapter = FakeAdapter()

    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9,
            state_id="ref-1",
        )
    )

    current = state(
        lens2=5000.0
    )

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "lens2_voltage_v",
        5000.0,
        7000.0,
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
                "lens2_voltage_v"
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

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        if adapter.command is None:
            return measurement(
                8.0e-9
            )

        values = {
            6000.0: 9.0e-9,
            7000.0: 8.5e-9,
        }

        return measurement(
            values[
                adapter.command
            ]
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    result = (
        module.scan_parameter_transmission_1d(
            adapter,
            current,
            profile,
            tracker,
            "lens2_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            settling_policies(),
            MeasurementPolicy(),
            ComparisonPolicy(
                uncertainty_multiple=2.0,
                minimum_relative_improvement=0.0,
            ),
        )
    )

    assert result.best_command == pytest.approx(
        6000.0
    )

    assert (
        result.best_transmission.transmission
        == pytest.approx(
            0.9
        )
    )


def test_reference_update_corrects_source_drift(
    monkeypatch,
):
    """
    Baseline:
        Cup1 = 10 nA
        Cup2 = 8 nA
        T = 80 %

    After source drift:
        Cup1 = 8 nA
        Cup2 candidate = 6.8 nA
        T = 85 %

    Raw Cup2 current fell from 8 to 6.8 nA, but normalized transmission
    improved from 80 to 85 %.
    """

    adapter = FakeAdapter()

    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9,
            state_id="ref-before",
            time=0.0,
        )
    )

    current = state(
        lens2=5000.0
    )

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "lens2_voltage_v",
        5000.0,
        6000.0,
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            applied(
                current,
                target,
            ),
    )

    measurements = iter(
        [
            measurement(
                8.0e-9
            ),
            measurement(
                6.8e-9
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

    maintenance_calls = 0

    def maintenance_hook(
        physical_state
    ):
        nonlocal maintenance_calls

        maintenance_calls += 1

        if tracker.latest.state_id == (
            "ref-before"
        ):
            tracker.add(
                reference(
                    8.0e-9,
                    state_id="ref-after",
                    time=600.0,
                )
            )

        return physical_state

    result = (
        module.scan_parameter_transmission_1d(
            adapter,
            current,
            profile,
            tracker,
            "lens2_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            settling_policies(),
            MeasurementPolicy(),
            ComparisonPolicy(
                uncertainty_multiple=0.0,
                minimum_relative_improvement=0.0,
            ),
            maintenance_hook=(
                maintenance_hook
            ),
        )
    )

    assert maintenance_calls >= 1

    assert (
        result.baseline_transmission.transmission
        == pytest.approx(
            0.8
        )
    )

    assert (
        result.best_transmission.transmission
        == pytest.approx(
            0.85
        )
    )

    assert result.best_command == pytest.approx(
        6000.0
    )

    # Critical check: raw downstream current became smaller.
    assert (
        result.best_measurement.mean_a
        < result.baseline_measurement.mean_a
    )


def test_maintenance_hook_must_restore_same_cup(
    monkeypatch,
):
    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9
        )
    )

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "lens2_voltage_v",
        5000.0,
        6000.0,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                8e-9
            ),
    )

    def bad_hook(
        physical_state
    ):
        return MachineState(
            mass_u=60.0,
            cup=3,
            stage=2,
            parameters=(
                physical_state.parameters.copy()
            ),
        )

    with pytest.raises(
        ValueError
    ):
        module.scan_parameter_transmission_1d(
            FakeAdapter(),
            state(),
            profile,
            tracker,
            "lens2_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            settling_policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            maintenance_hook=(
                bad_hook
            ),
        )


def test_reference_mass_must_match_state():
    tracker = SourceReferenceTracker()

    wrong_reference = SourceReference(
        measurement=measurement(
            10e-9
        ),
        state_id="wrong",
        mass_u=180.0,
        monotonic_s=0.0,
        created_at_utc=(
            "2026-08-26T15:00:00+00:00"
        ),
    )

    # Add directly so the tracker itself does not establish another mass.
    tracker.references.append(
        wrong_reference
    )

    with pytest.raises(
        ValueError
    ):
        module.scan_parameter_transmission_1d(
            FakeAdapter(),
            state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker,
            "lens2_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            settling_policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
        )


def test_flat_transmission_keeps_initial_command(
    monkeypatch,
):
    tracker = SourceReferenceTracker()

    tracker.add(
        reference(
            10e-9
        )
    )

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_learned_range(
        "lens2_voltage_v",
        5000.0,
        7000.0,
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=False:
            applied(
                current,
                target,
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

    result = (
        module.scan_parameter_transmission_1d(
            FakeAdapter(),
            state(
                5000.0
            ),
            profile,
            tracker,
            "lens2_voltage_v",
            ScanPolicy(
                steps=(1000.0,)
            ),
            settling_policies(),
            MeasurementPolicy(),
            ComparisonPolicy(
                minimum_relative_improvement=0.001
            ),
        )
    )

    assert result.best_command == 5000.0
    assert result.improved is False