from dataclasses import dataclass

import pytest

import sirius.reference_orchestrator as module
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.reference import (
    SourceReferenceTracker,
)
from sirius.state import MachineState
from sirius.transition import (
    AppliedStateResult,
    StateTransitionPlan,
)


@dataclass
class FakeAdapter:
    readbacks: dict[str, float]

    def read_parameter(self, name):
        return self.readbacks.get(name)


def measurement(
    mean_a=10e-9,
    *,
    below_noise_floor=False,
):
    return BeamMeasurement(
        mean_a=mean_a,
        sigma_a=0.1e-9,
        sem_a=0.02e-9,
        n=10,
        duration_s=1.0,
        relative_sem=0.002,
        precision_threshold_a=0.1e-9,
        drift_delta_a=0.0,
        stop_reason="precision_reached",
        below_noise_floor=below_noise_floor,
        samples=(),
    )


def state(
    *,
    cup,
    role,
    lens2=5000.0,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        role=role,
        parameters={
            "extraction_voltage_v": 19000.0,
            "einzel_lens_voltage_v": 18000.0,
            "magnet_current_a": 42.0,
            "lens2_voltage_v": lens2,
        },
    )


def applied_result(
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
        selected_cup=target.cup,
    )


def test_reference_state_must_use_cup1():
    working = state(
        cup=3,
        role="working",
    )

    invalid_reference = state(
        cup=2,
        role="cup1_reference",
    )

    with pytest.raises(ValueError):
        module.validate_reference_states(
            working,
            invalid_reference,
        )


def test_working_state_must_define_cup():
    working = MachineState(
        mass_u=60.0,
        cup=None,
        parameters={},
    )

    reference = MachineState(
        mass_u=60.0,
        cup=1,
        parameters={},
    )

    with pytest.raises(ValueError):
        module.validate_reference_states(
            working,
            reference,
        )


def test_reference_check_switches_to_reference_and_restores_working(
    monkeypatch,
):
    working = state(
        cup=3,
        role="working",
        lens2=5500.0,
    )

    reference = state(
        cup=1,
        role="cup1_reference",
        lens2=5000.0,
    )

    adapter = FakeAdapter(
        readbacks={}
    )

    tracker = SourceReferenceTracker()

    calls = []

    def fake_capture(adapter, current):
        return current

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=True,
    ):
        calls.append(
            (
                current.cup,
                target.cup,
                target.role,
            )
        )

        return applied_result(
            target
        )

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
        monotonic=None,
    ):
        return measurement()

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        fake_capture,
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

    clock_values = iter(
        [100.0]
    )

    result = module.perform_source_reference_check(
        adapter,
        working,
        reference,
        tracker,
        settling_policies={},
        measurement_policy=MeasurementPolicy(),
        monotonic=lambda: next(clock_values),
        utc_now=lambda: "2026-08-26T12:00:00+00:00",
    )

    assert calls == [
        (
            3,
            1,
            "cup1_reference",
        ),
        (
            1,
            3,
            "working",
        ),
    ]

    assert result.reference.measurement.mean_a == pytest.approx(
        10e-9
    )

    assert tracker.latest is result.reference

    assert result.working_state_after.cup == 3


def test_reference_is_linked_to_saved_cup1_state(
    monkeypatch,
):
    working = state(
        cup=4,
        role="working",
    )

    reference = state(
        cup=1,
        role="cup1_reference",
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    monkeypatch.setattr(
        module,
        "apply_state",
        lambda adapter, current, target, settling_policies, select_target_cup=True:
            applied_result(target),
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda adapter, policy, noise_floor_a=None, monotonic=None:
            measurement(),
    )

    tracker = SourceReferenceTracker()

    result = module.perform_source_reference_check(
        FakeAdapter({}),
        working,
        reference,
        tracker,
        settling_policies={},
        measurement_policy=MeasurementPolicy(),
        monotonic=lambda: 1000.0,
        utc_now=lambda: "2026-08-26T12:00:00+00:00",
    )

    assert (
        result.reference.state_id
        == reference.state_id
    )


def test_below_noise_reference_is_rejected_and_working_state_restored(
    monkeypatch,
):
    working = state(
        cup=5,
        role="working",
    )

    reference = state(
        cup=1,
        role="cup1_reference",
    )

    calls = []

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=True,
    ):
        calls.append(
            target.cup
        )

        return applied_result(
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
        lambda adapter, policy, noise_floor_a=None, monotonic=None:
            measurement(
                0.1e-12,
                below_noise_floor=True,
            ),
    )

    tracker = SourceReferenceTracker()

    with pytest.raises(
        module.InvalidReferenceMeasurementError
    ):
        module.perform_source_reference_check(
            FakeAdapter({}),
            working,
            reference,
            tracker,
            settling_policies={},
            measurement_policy=MeasurementPolicy(),
            monotonic=lambda: 100.0,
        )

    assert calls == [
        1,
        5,
    ]

    assert tracker.latest is None


def test_measurement_failure_triggers_controlled_restore(
    monkeypatch,
):
    working = state(
        cup=6,
        role="working",
    )

    reference = state(
        cup=1,
        role="cup1_reference",
    )

    calls = []

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=True,
    ):
        calls.append(
            target.cup
        )

        return applied_result(
            target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    def fail_measurement(*args, **kwargs):
        raise RuntimeError(
            "Keithley failed"
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fail_measurement,
    )

    tracker = SourceReferenceTracker()

    with pytest.raises(
        RuntimeError,
        match="Keithley failed",
    ):
        module.perform_source_reference_check(
            FakeAdapter({}),
            working,
            reference,
            tracker,
            settling_policies={},
            measurement_policy=MeasurementPolicy(),
        )

    assert calls == [
        1,
        6,
    ]


def test_failed_reference_transition_does_not_trigger_blind_restore(
    monkeypatch,
):
    working = state(
        cup=3,
        role="working",
    )

    reference = state(
        cup=1,
        role="cup1_reference",
    )

    calls = []

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fail_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=True,
    ):
        calls.append(
            target.cup
        )

        raise RuntimeError(
            "HV transition failed"
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fail_apply,
    )

    with pytest.raises(
        RuntimeError,
        match="HV transition failed",
    ):
        module.perform_source_reference_check(
            FakeAdapter({}),
            working,
            reference,
            SourceReferenceTracker(),
            settling_policies={},
            measurement_policy=MeasurementPolicy(),
        )

    assert calls == [
        1
    ]


def test_measurement_and_restore_failure_are_both_preserved(
    monkeypatch,
):
    working = state(
        cup=4,
        role="working",
    )

    reference = state(
        cup=1,
        role="cup1_reference",
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    call_count = 0

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=True,
    ):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return applied_result(
                target
            )

        raise RuntimeError(
            "restore failed"
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs: (
            (_ for _ in ()).throw(
                RuntimeError(
                    "measurement failed"
                )
            )
        ),
    )

    with pytest.raises(
        module.ReferenceMeasurementAndRestoreError
    ) as exc:
        module.perform_source_reference_check(
            FakeAdapter({}),
            working,
            reference,
            SourceReferenceTracker(),
            settling_policies={},
            measurement_policy=MeasurementPolicy(),
        )

    assert str(
        exc.value.measurement_error
    ) == "measurement failed"

    assert str(
        exc.value.restore_error
    ) == "restore failed"