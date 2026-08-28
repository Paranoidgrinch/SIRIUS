from types import SimpleNamespace

import pytest

import sirius.qpt_scan2d as module
from sirius.coupled_transition import (
    qpt_transition_policy,
)
from sirius.comparison import ComparisonPolicy
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.qpt_model import (
    QPT1_PARAMETER,
    QPT2_PARAMETER,
    QPT3_PARAMETER,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState


def measurement(
    value=5e-9,
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


def state():
    return MachineState(
        mass_u=60.0,
        cup=4,
        stage=4,
        parameters={
            QPT1_PARAMETER: 2000.0,
            QPT2_PARAMETER: 3000.0,
            QPT3_PARAMETER: 2000.0,
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
            monotonic_s=0.0,
            created_at_utc=(
                "2026-08-28T13:00:00+00:00"
            ),
        )
    )

    return result


def settling():
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        QPT1_PARAMETER: policy,
        QPT2_PARAMETER: policy,
        QPT3_PARAMETER: policy,
    }


def test_qpt_policy_rejects_wrong_coupled_group():
    from sirius.coupled_transition import (
        CoupledTransitionPolicy,
    )

    wrong = CoupledTransitionPolicy(
        parameter_order=(
            QPT1_PARAMETER,
            QPT2_PARAMETER,
        ),
        max_step_by_parameter={
            QPT1_PARAMETER: 100.0,
            QPT2_PARAMETER: 100.0,
        },
    )

    with pytest.raises(
        ValueError,
        match="exactly",
    ):
        module.QPT2DScanPolicy(
            transition_policy=wrong
        )


def test_qpt_scanner_uses_coupled_executor_when_configured(
    monkeypatch,
):
    current = state()

    coupled_calls = []

    def fake_coupled(
        adapter,
        current,
        target,
        settling_policies,
        policy,
        *,
        logger=None,
    ):
        coupled_calls.append(
            (
                current,
                target,
                policy,
            )
        )

        return SimpleNamespace(
            final_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_coupled_transition",
        fake_coupled,
    )

    def forbidden_direct(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Direct apply_state() must not be used for configured QPT scan"
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        forbidden_direct,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    transition_policy = (
        qpt_transition_policy(
            max_step_v=100.0
        )
    )

    scan_policy = module.QPT2DScanPolicy(
        initial_focus_half_width_v=250.0,
        initial_asymmetry_half_width_v=250.0,
        levels=(
            module.QPTScanLevel(
                focus_step_v=250.0,
                asymmetry_step_v=250.0,
            ),
        ),
        max_points_per_level=100,
        transition_policy=(
            transition_policy
        ),
    )

    result = module.scan_qpt_focus_asymmetry_2d(
        object(),
        current,
        tracker(),
        scan_policy,
        settling(),
        MeasurementPolicy(),
        ComparisonPolicy(),
    )

    assert len(
        coupled_calls
    ) > 0

    assert all(
        call[
            2
        ] is transition_policy
        for call
        in coupled_calls
    )

    assert result.final_state.cup == 4


def test_qpt_scanner_keeps_direct_mode_without_policy(
    monkeypatch,
):
    current = state()

    direct_calls = []

    def fake_direct(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        direct_calls.append(
            target
        )

        return SimpleNamespace(
            observed_state=target
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_direct,
    )

    def forbidden_coupled(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Coupled executor must not be used without a policy"
        )

    monkeypatch.setattr(
        module,
        "apply_coupled_transition",
        forbidden_coupled,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                5e-9
            ),
    )

    scan_policy = module.QPT2DScanPolicy(
        initial_focus_half_width_v=250.0,
        initial_asymmetry_half_width_v=250.0,
        levels=(
            module.QPTScanLevel(
                focus_step_v=250.0,
                asymmetry_step_v=250.0,
            ),
        ),
        max_points_per_level=100,
    )

    module.scan_qpt_focus_asymmetry_2d(
        object(),
        current,
        tracker(),
        scan_policy,
        settling(),
        MeasurementPolicy(),
        ComparisonPolicy(),
    )

    assert len(
        direct_calls
    ) > 0