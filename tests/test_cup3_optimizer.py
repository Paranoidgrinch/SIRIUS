from types import SimpleNamespace

import pytest

import sirius.cup3_optimizer as module
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
from sirius.rfq_matching import (
    LCSetting,
    RFQMatchingPolicy,
    RFQTargetQPolicy,
)
from sirius.settling import (
    SettlingPolicy,
)
from sirius.state import (
    MachineState,
)


class FakeRFQHardware:
    def __init__(self):
        self.generator_vpp = None

    def set_generator_amplitude_vpp(
        self,
        value,
    ):
        self.generator_vpp = float(
            value
        )


def measurement(
    current=8e-9,
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
        },
    )


def cup3_state():
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        role="working",
        parameters={
            "sputter_voltage_v": 8000.0,
            "extraction_voltage_v": 19600.0,
            "einzel_lens_voltage_v": 18200.0,
            "magnet_current_a": 34.0,

            "lens2_voltage_v": 6000.0,
            "steerer_x1_v": 20.0,
            "steerer_y1_v": -15.0,

            "ion_cooler_voltage_v": 26460.0,

            "deceleration_voltage_v": 1000.0,
            "acceleration_voltage_v": 1000.0,

            "guidefield1_voltage_v": 10.0,
            "guidefield2_voltage_v": 20.0,
        },
    )


def tracker():
    result = SourceReferenceTracker()

    result.add(
        SourceReference(
            measurement=measurement(
                10e-9
            ),
            state_id=(
                cup1_state().state_id
            ),
            mass_u=60.0,
            monotonic_s=100.0,
            created_at_utc=(
                "2026-08-27T10:00:00+00:00"
            ),
        )
    )

    return result


def policies():
    policy = SettlingPolicy(
        max_readback_span=5.0
    )

    return {
        name: policy
        for name in (
            *module.CUP3_PRIMARY_PARAMETERS,
            *module.CUP3_UPSTREAM_RETUNE_PARAMETERS,
            *module.CUP3_FROZEN_SOURCE_PARAMETERS,
        )
    }


def matching_result():
    return SimpleNamespace(
        mass_u=60.0,
        best_setting=LCSetting(
            inductance_uh=100.0,
            capacitance_pf=100.0,
        ),
        best_frequency_hz=1.8e6,
    )


def q_result(
    *,
    generator=10.0,
    q=0.45,
):
    return SimpleNamespace(
        mass_u=60.0,
        target_q=0.45,
        setting=LCSetting(
            inductance_uh=100.0,
            capacitance_pf=100.0,
        ),
        frequency_hz=1.8e6,
        generator_amplitude_vpp=(
            generator
        ),
        measured_rfq_vpp=500.0,
        measured_q=q,
    )


def residual_result(
    current_state,
    *,
    target=40.0,
):
    return SimpleNamespace(
        final_state=current_state,
        best_target_residual_energy_ev=(
            target
        ),
    )


def scan_result(
    current_state,
):
    return SimpleNamespace(
        final_state=current_state
    )


def rfq_matching_policy():
    return RFQMatchingPolicy(
        probe_generator_vpp=2.0,
        requested_frequency_hz=1.8e6,
        frequency_half_width_hz=100e3,
        coarse_frequency_step_hz=20e3,
        fine_frequency_step_hz=5e3,
    )


def rfq_q_policy():
    return RFQTargetQPolicy(
        generator_max_vpp=20.0,
        initial_generator_vpp=2.0,
    )


def patch_common(
    monkeypatch,
    *,
    phase_calls=None,
):
    if phase_calls is None:
        phase_calls = []

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state: state,
    )

    def fake_matching(
        *args,
        **kwargs,
    ):
        phase_calls.append(
            "rfq_matching"
        )

        return matching_result()

    monkeypatch.setattr(
        module,
        "search_rfq_resonance",
        fake_matching,
    )

    q_counter = {
        "n": 0
    }

    def fake_q(
        *args,
        **kwargs,
    ):
        q_counter["n"] += 1

        phase_calls.append(
            f"q_{q_counter['n']}"
        )

        return q_result()

    monkeypatch.setattr(
        module,
        "set_target_q",
        fake_q,
    )

    residual_counter = {
        "n": 0
    }

    def fake_residual(
        adapter,
        current_state,
        *args,
        **kwargs,
    ):
        residual_counter["n"] += 1

        phase_calls.append(
            f"residual_{residual_counter['n']}"
        )

        return residual_result(
            current_state
        )

    monkeypatch.setattr(
        module,
        "scan_residual_energy",
        fake_residual,
    )

    def fake_end_coordinates(
        adapter,
        current_state,
        *args,
        **kwargs,
    ):
        phase_calls.append(
            "end_coordinates"
        )

        return SimpleNamespace(
            final_state=current_state,
            scans=(),
        )

    monkeypatch.setattr(
        module,
        "optimize_end_electrode_coordinates",
        fake_end_coordinates,
    )

    def fake_guide_coordinates(
        adapter,
        current_state,
        *args,
        **kwargs,
    ):
        phase_calls.append(
            "guidefield_coordinates"
        )

        return SimpleNamespace(
            final_state=current_state,
            scans=(),
            direction_evidence=(),
        )

    monkeypatch.setattr(
        module,
        "optimize_guidefield_coordinates",
        fake_guide_coordinates,
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
        phase_calls.append(
            parameter_name
        )

        return scan_result(
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

    return phase_calls


def test_requires_cup3():
    invalid = cup3_state()
    invalid.cup = 2

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            invalid,
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_source_and_magnet_must_match_cup1():
    invalid = cup3_state()

    invalid.parameters[
        "magnet_current_a"
    ] = 35.0

    with pytest.raises(
        ValueError
    ):
        module._validate_inputs(
            invalid,
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
        )


def test_optimizer_phase_order(
    monkeypatch,
):
    calls = patch_common(
        monkeypatch
    )

    result = module.optimize_cup3(
        object(),
        FakeRFQHardware(),
        cup3_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        lc_candidates=(
            LCSetting(
                100.0,
                100.0,
            ),
        ),
        rfq_matching_policy=(
            rfq_matching_policy()
        ),
        rfq_q_policy=(
            rfq_q_policy()
        ),
        target_q=0.45,
        optimization_policy=(
            module.Cup3OptimizationPolicy(
                electrode_passes=1,
                guidefield_passes=1,
                upstream_passes=1,
            )
        ),
        monotonic=lambda: 100.0,
    )

    assert calls == [
        "rfq_matching",
        "q_1",
        "residual_1",

        "end_coordinates",
        "guidefield_coordinates",

        "einzel_lens_voltage_v",
        "lens2_voltage_v",
        "steerer_x1_v",
        "steerer_y1_v",

        "residual_2",
        "q_2",
    ]

    assert (
        result.final_q_result.measured_q
        == pytest.approx(
            0.45
        )
    )


def test_final_state_contains_measured_rfq_information(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    result = module.optimize_cup3(
        object(),
        FakeRFQHardware(),
        cup3_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        lc_candidates=(
            LCSetting(
                100.0,
                100.0,
            ),
        ),
        rfq_matching_policy=(
            rfq_matching_policy()
        ),
        rfq_q_policy=(
            rfq_q_policy()
        ),
        target_q=0.45,
        monotonic=lambda: 100.0,
    )

    assert (
        result.final_state.rfq.frequency_hz
        == pytest.approx(
            1.8e6
        )
    )

    assert (
        result.final_state.rfq.inductance_uh
        == pytest.approx(
            100.0
        )
    )

    assert (
        result.final_state.rfq.capacitance_pf
        == pytest.approx(
            100.0
        )
    )

    assert (
        result.final_state.rfq.rfq_vpp_measured
        == pytest.approx(
            500.0
        )
    )

    assert (
        result.final_state.rfq.q_measured
        == pytest.approx(
            0.45
        )
    )


def test_final_transmission_is_source_normalized(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    result = module.optimize_cup3(
        object(),
        FakeRFQHardware(),
        cup3_state(),
        cup1_state(),
        MassProfile(
            mass_u=60.0
        ),
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        lc_candidates=(
            LCSetting(
                100.0,
                100.0,
            ),
        ),
        rfq_matching_policy=(
            rfq_matching_policy()
        ),
        rfq_q_policy=(
            rfq_q_policy()
        ),
        target_q=0.45,
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


def test_accidental_source_change_aborts(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    def bad_scan(
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

        parameters[
            "magnet_current_a"
        ] = 35.0

        return scan_result(
            MachineState(
                mass_u=60.0,
                cup=3,
                stage=3,
                parameters=parameters,
                rfq=current_state.rfq,
            )
        )

    monkeypatch.setattr(
        module,
        "scan_parameter_transmission_1d",
        bad_scan,
    )

    hardware = FakeRFQHardware()

    with pytest.raises(
        module.Cup3OptimizationError
    ):
        module.optimize_cup3(
            object(),
            hardware,
            cup3_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            lc_candidates=(
                LCSetting(
                    100.0,
                    100.0,
                ),
            ),
            rfq_matching_policy=(
                rfq_matching_policy()
            ),
            rfq_q_policy=(
                rfq_q_policy()
            ),
            target_q=0.45,
            monotonic=lambda: 100.0,
        )

    assert (
        hardware.generator_vpp
        == 0.0
    )


def test_failed_downstream_optimization_turns_rf_off(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    def fail_residual(
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            "simulated optimizer failure"
        )

    monkeypatch.setattr(
        module,
        "scan_residual_energy",
        fail_residual,
    )

    hardware = FakeRFQHardware()

    with pytest.raises(
        RuntimeError
    ):
        module.optimize_cup3(
            object(),
            hardware,
            cup3_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            lc_candidates=(
                LCSetting(
                    100.0,
                    100.0,
                ),
            ),
            rfq_matching_policy=(
                rfq_matching_policy()
            ),
            rfq_q_policy=(
                rfq_q_policy()
            ),
            target_q=0.45,
            monotonic=lambda: 100.0,
        )

    assert (
        hardware.generator_vpp
        == 0.0
    )


def test_profile_stores_only_cup3_primary_commands(
    monkeypatch,
):
    patch_common(
        monkeypatch
    )

    profile = MassProfile(
        mass_u=60.0
    )

    profile.set_best_command(
        "einzel_lens_voltage_v",
        18000.0,
    )

    result = module.optimize_cup3(
        object(),
        FakeRFQHardware(),
        cup3_state(),
        cup1_state(),
        profile,
        tracker(),
        policies(),
        MeasurementPolicy(),
        ComparisonPolicy(),
        lc_candidates=(
            LCSetting(
                100.0,
                100.0,
            ),
        ),
        rfq_matching_policy=(
            rfq_matching_policy()
        ),
        rfq_q_policy=(
            rfq_q_policy()
        ),
        target_q=0.45,
        monotonic=lambda: 100.0,
    )

    for parameter_name in (
        module.CUP3_PRIMARY_PARAMETERS
    ):
        assert (
            profile.best_commands[
                parameter_name
            ]
            == result.final_state.parameters[
                parameter_name
            ]
        )

    # Cup-1 / upstream global starting value remains untouched.
    assert (
        profile.best_commands[
            "einzel_lens_voltage_v"
        ]
        == 18000.0
    )

    assert (
        profile.best_state_ids[
            "cup3_best"
        ]
        == result.final_state.state_id
    )

    assert (
        profile.metadata[
            "cup3_rfq"
        ][
            "q_measured"
        ]
        == pytest.approx(
            0.45
        )
    )


def test_below_noise_final_current_is_rejected_and_rf_disabled(
    monkeypatch,
):
    patch_common(
        monkeypatch
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

    hardware = FakeRFQHardware()

    with pytest.raises(
        module.Cup3OptimizationNoBeamError
    ):
        module.optimize_cup3(
            object(),
            hardware,
            cup3_state(),
            cup1_state(),
            MassProfile(
                mass_u=60.0
            ),
            tracker(),
            policies(),
            MeasurementPolicy(),
            ComparisonPolicy(),
            lc_candidates=(
                LCSetting(
                    100.0,
                    100.0,
                ),
            ),
            rfq_matching_policy=(
                rfq_matching_policy()
            ),
            rfq_q_policy=(
                rfq_q_policy()
            ),
            target_q=0.45,
            monotonic=lambda: 100.0,
        )

    assert (
        hardware.generator_vpp
        == 0.0
    )