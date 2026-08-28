from types import SimpleNamespace

import pytest

import sirius.cup3_coordinates as module
from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.scan1d import ScanPolicy
from sirius.settling import SettlingPolicy
from sirius.state import MachineState


def measurement(
    value=5e-9,
    *,
    below_noise=False,
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
        below_noise_floor=below_noise,
        samples=(),
    )


def transmission(
    value,
    sem=0.005,
):
    return SimpleNamespace(
        transmission=value,
        transmission_sem=sem,
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
                "2026-08-28T06:00:00+00:00"
            ),
        )
    )

    return result


def state():
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "deceleration_voltage_v": 1200.0,
            "acceleration_voltage_v": 800.0,
            "guidefield1_voltage_v": 10.0,
            "guidefield2_voltage_v": 20.0,
        },
    )


def policies():
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        "deceleration_voltage_v": policy,
        "acceleration_voltage_v": policy,
        "guidefield1_voltage_v": policy,
        "guidefield2_voltage_v": policy,
    }


def fake_direction_scan(
    *,
    positive_t,
    negative_t,
):
    return SimpleNamespace(
        initial_coordinate=0.0,
        baseline_transmission=(
            transmission(
                0.1
            )
        ),
        baseline_measurement=(
            measurement()
        ),
        points=(
            SimpleNamespace(
                coordinate_value=-10.0,
                transmission=(
                    transmission(
                        negative_t
                    )
                ),
                measurement=(
                    measurement()
                ),
            ),
            SimpleNamespace(
                coordinate_value=10.0,
                transmission=(
                    transmission(
                        positive_t
                    )
                ),
                measurement=(
                    measurement()
                ),
            ),
        ),
    )


def test_unknown_direction_window_includes_both_signs():
    minimum, maximum = (
        module._local_window(
            current=12.0,
            feasible_minimum=-30.0,
            feasible_maximum=30.0,
            half_width=16.0,
            include_both_signs=True,
        )
    )

    assert minimum < 0
    assert maximum > 0

    assert minimum <= 12.0 <= maximum


def test_local_window_never_drops_current_state():
    minimum, maximum = (
        module._local_window(
            current=25.0,
            feasible_minimum=-30.0,
            feasible_maximum=30.0,
            half_width=5.0,
            include_both_signs=False,
        )
    )

    assert minimum <= 25.0 <= maximum


def test_positive_guidefield_direction_is_learned():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = (
        module.learn_guidefield_direction_from_scan(
            profile,
            fake_direction_scan(
                positive_t=0.80,
                negative_t=0.40,
            ),
            ComparisonPolicy(
                uncertainty_multiple=2.0,
                minimum_relative_improvement=0.0,
            ),
        )
    )

    assert evidence.proposed_sign == 1

    assert evidence.profile_updated is True

    assert (
        profile.guidefield_forward_sign
        == 1
    )


def test_negative_guidefield_direction_is_learned():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = (
        module.learn_guidefield_direction_from_scan(
            profile,
            fake_direction_scan(
                positive_t=0.40,
                negative_t=0.80,
            ),
            ComparisonPolicy(
                uncertainty_multiple=2.0,
                minimum_relative_improvement=0.0,
            ),
        )
    )

    assert evidence.proposed_sign == -1

    assert (
        profile.guidefield_forward_sign
        == -1
    )


def test_indistinguishable_direction_is_not_learned():
    profile = MassProfile(
        mass_u=60.0
    )

    evidence = (
        module.learn_guidefield_direction_from_scan(
            profile,
            fake_direction_scan(
                positive_t=0.800,
                negative_t=0.799,
            ),
            ComparisonPolicy(
                uncertainty_multiple=2.0,
                minimum_relative_improvement=0.0,
            ),
        )
    )

    assert evidence.proposed_sign is None

    assert evidence.profile_updated is False

    assert (
        profile.guidefield_forward_sign
        is None
    )


def test_strong_new_evidence_can_correct_old_direction():
    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_guidefield_forward_sign(
        1
    )

    evidence = (
        module.learn_guidefield_direction_from_scan(
            profile,
            fake_direction_scan(
                positive_t=0.30,
                negative_t=0.80,
            ),
            ComparisonPolicy(
                uncertainty_multiple=2.0,
                minimum_relative_improvement=0.0,
            ),
        )
    )

    assert evidence.previous_sign == 1
    assert evidence.proposed_sign == -1

    assert (
        profile.guidefield_forward_sign
        == -1
    )


def test_missing_negative_side_does_not_force_learning():
    profile = MassProfile(
        mass_u=60.0
    )

    scan = SimpleNamespace(
        initial_coordinate=0.0,
        baseline_transmission=(
            transmission(
                0.1
            )
        ),
        baseline_measurement=(
            measurement()
        ),
        points=(
            SimpleNamespace(
                coordinate_value=10.0,
                transmission=(
                    transmission(
                        0.8
                    )
                ),
                measurement=(
                    measurement()
                ),
            ),
        ),
    )

    evidence = (
        module.learn_guidefield_direction_from_scan(
            profile,
            scan,
            ComparisonPolicy(),
        )
    )

    assert evidence.proposed_sign is None

    assert (
        profile.guidefield_forward_sign
        is None
    )


def test_end_electrode_optimizer_runs_difference_then_common(
    monkeypatch,
):
    current = state()

    calls = []

    def fake_scan(
        adapter,
        current_state,
        tracker,
        *,
        coordinate_name,
        coordinate_reader,
        command_builder,
        **kwargs,
    ):
        calls.append(
            coordinate_name
        )

        return SimpleNamespace(
            final_state=current_state
        )

    monkeypatch.setattr(
        module,
        "scan_derived_coordinate_transmission_1d",
        fake_scan,
    )

    result = (
        module.optimize_end_electrode_coordinates(
            object(),
            current,
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            policy=(
                module.EndElectrodeCoordinatePolicy(
                    passes=2
                )
            ),
        )
    )

    assert calls == [
        "end_electrode_difference_v",
        "end_electrode_common_v",
        "end_electrode_difference_v",
        "end_electrode_common_v",
    ]

    assert result.final_state is current


def test_guidefield_optimizer_runs_difference_then_common(
    monkeypatch,
):
    current = state()

    profile = MassProfile(
        mass_u=60.0
    )

    calls = []

    direction_scan = (
        fake_direction_scan(
            positive_t=0.8,
            negative_t=0.4,
        )
    )

    direction_scan.final_state = current

    def fake_scan(
        adapter,
        current_state,
        tracker,
        *,
        coordinate_name,
        **kwargs,
    ):
        calls.append(
            coordinate_name
        )

        if coordinate_name == (
            "guidefield_difference_v"
        ):
            return direction_scan

        return SimpleNamespace(
            final_state=current_state
        )

    monkeypatch.setattr(
        module,
        "scan_derived_coordinate_transmission_1d",
        fake_scan,
    )

    result = (
        module.optimize_guidefield_coordinates(
            object(),
            current,
            profile,
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(
                minimum_relative_improvement=0.0
            ),
            policy=(
                module.GuidefieldCoordinatePolicy(
                    passes=1
                )
            ),
        )
    )

    assert calls == [
        "guidefield_difference_v",
        "guidefield_common_v",
    ]

    assert (
        profile.guidefield_forward_sign
        == 1
    )

    assert len(
        result.direction_evidence
    ) == 1