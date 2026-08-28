from types import SimpleNamespace

import pytest

import sirius.final_characterization as module
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
)
from sirius.state import (
    MachineState,
    RFQState,
)


class FakeAdapter:
    def __init__(
        self,
        state,
    ):
        self.state = state

        self.selected_cups = []

        self.requested_states = []


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


def rfq():
    return RFQState(
        frequency_hz=1.8e6,
        generator_amplitude_vpp=10.0,
        inductance_uh=100.0,
        capacitance_pf=100.0,
        rfq_vpp_measured=500.0,
        q_target=0.45,
        q_measured=0.45,
    )


def final_state():
    return MachineState(
        mass_u=60.0,
        cup=6,
        stage=6,
        role="stage_best",
        parameters={
            "sputter_voltage_v": 8000.0,
            "extraction_voltage_v": 19600.0,
            "einzel_lens_voltage_v": 18000.0,
            "magnet_current_a": 34.0,

            "lens2_voltage_v": 6000.0,
            "steerer_x1_v": 20.0,
            "steerer_y1_v": -15.0,

            "ion_cooler_voltage_v": 26460.0,
            "deceleration_voltage_v": 1000.0,
            "acceleration_voltage_v": 1000.0,
            "guidefield1_voltage_v": 10.0,
            "guidefield2_voltage_v": 20.0,

            "quadrupole1_voltage_v": 2000.0,
            "quadrupole2_voltage_v": 3000.0,
            "quadrupole3_voltage_v": 2000.0,
            "steerer_x2_v": 10.0,
            "steerer_y2_v": -10.0,

            "esa_voltage_v": 2760.0,

            "lens4_voltage_v": 5000.0,
            "steerer_x3_v": 10.0,
            "steerer_y3_v": -10.0,
        },
        rfq=rfq(),
    )


def fake_transition(
    current,
    target,
):
    return SimpleNamespace(
        requested_state=target,
        observed_state=target,
        plan=SimpleNamespace(),
        settling_results=(),
        selected_cup=(
            target.cup
        ),
    )


def patch_normal_hardware(
    monkeypatch,
    adapter,
):
    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    def fake_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        assert (
            select_target_cup
            is True
        )

        adapter.selected_cups.append(
            target.cup
        )

        adapter.requested_states.append(
            target
        )

        adapter.state = target

        return fake_transition(
            current,
            target,
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        fake_apply,
    )


def test_requires_final_cup6_state():
    invalid = final_state()

    invalid.cup = 5

    with pytest.raises(
        ValueError,
        match="Cup-6",
    ):
        module._validate_initial_state(
            invalid
        )


def test_canonical_sequence_is_enforced():
    with pytest.raises(
        ValueError,
        match="canonical sequence",
    ):
        module.FinalCharacterizationPolicy(
            sequence=(
                1,
                2,
                3,
                1,
            )
        )


def test_final_characterization_selects_exact_cup_sequence(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    patch_normal_hardware(
        monkeypatch,
        adapter,
    )

    currents = {
        1: [
            10e-9,
            9.5e-9,
        ],
        2: [9e-9],
        3: [8e-9],
        4: [7e-9],
        5: [6e-9],
        6: [5e-9],
    }

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        cup = adapter.state.cup

        return measurement(
            currents[
                cup
            ].pop(0)
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    result = (
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
            monotonic=iter(
                range(
                    100,
                    200,
                )
            ).__next__,
        )
    )

    assert (
        tuple(
            adapter.selected_cups
        )
        == module.FINAL_CUP_SEQUENCE
    )

    assert (
        tuple(
            point.cup
            for point
            in result.points
        )
        == module.FINAL_CUP_SEQUENCE
    )


def test_transmissions_use_initial_frozen_cup1_measurement(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    patch_normal_hardware(
        monkeypatch,
        adapter,
    )

    currents = {
        1: [
            10e-9,
            9.5e-9,
        ],
        2: [9e-9],
        3: [8e-9],
        4: [7e-9],
        5: [6e-9],
        6: [5e-9],
    }

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        return measurement(
            currents[
                adapter.state.cup
            ].pop(0)
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    result = (
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
            monotonic=iter(
                range(
                    100,
                    200,
                )
            ).__next__,
        )
    )

    transmissions = (
        result.transmissions_by_cup
    )

    assert (
        transmissions[
            2
        ].transmission
        == pytest.approx(
            0.9
        )
    )

    assert (
        transmissions[
            3
        ].transmission
        == pytest.approx(
            0.8
        )
    )

    assert (
        transmissions[
            4
        ].transmission
        == pytest.approx(
            0.7
        )
    )

    assert (
        transmissions[
            5
        ].transmission
        == pytest.approx(
            0.6
        )
    )

    assert (
        transmissions[
            6
        ].transmission
        == pytest.approx(
            0.5
        )
    )


def test_final_cup1_reports_source_drift(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    patch_normal_hardware(
        monkeypatch,
        adapter,
    )

    currents = {
        1: [
            10e-9,
            9e-9,
        ],
        2: [9e-9],
        3: [8e-9],
        4: [7e-9],
        5: [6e-9],
        6: [5e-9],
    }

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        return measurement(
            currents[
                adapter.state.cup
            ].pop(0)
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    times = iter(
        (
            100.0,
            110.0,
            120.0,
            130.0,
            140.0,
            150.0,
            160.0,
        )
    )

    result = (
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
            monotonic=times.__next__,
        )
    )

    drift = (
        result.cup1_drift
    )

    assert drift.ratio == pytest.approx(
        0.9
    )

    assert (
        drift.drift_fraction
        == pytest.approx(
            -0.1
        )
    )

    assert (
        drift.drift_percent
        == pytest.approx(
            -10.0
        )
    )

    assert (
        drift.elapsed_s
        == pytest.approx(
            60.0
        )
    )


def test_no_machine_commands_change_between_cups(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    patch_normal_hardware(
        monkeypatch,
        adapter,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                10e-9
            ),
    )

    module.characterize_final_transmission(
        adapter,
        start,
        {},
        MeasurementPolicy(),
        policy=(
            module.FinalCharacterizationPolicy(
                cup_settle_s=0.0
            )
        ),
    )

    for target in (
        adapter.requested_states
    ):
        assert (
            target.parameters
            == start.parameters
        )

        assert (
            target.rfq
            == start.rfq
        )


def test_parameter_change_during_cup_selection_aborts(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    def bad_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        parameters = dict(
            target.parameters
        )

        if target.cup == 3:
            parameters[
                "esa_voltage_v"
            ] += 1.0

        observed = MachineState(
            mass_u=target.mass_u,
            parameters=parameters,
            cup=target.cup,
            stage=6,
            rfq=target.rfq,
        )

        adapter.state = observed

        return SimpleNamespace(
            observed_state=observed
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        bad_apply,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                10e-9
            ),
    )

    with pytest.raises(
        module.FrozenConfigurationChangedError,
        match="esa_voltage_v",
    ):
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
        )


def test_rfq_change_during_characterization_aborts(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    monkeypatch.setattr(
        module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    def bad_apply(
        adapter,
        current,
        target,
        settling_policies,
        select_target_cup=False,
    ):
        changed_rfq = (
            target.rfq
        )

        if target.cup == 4:
            changed_rfq = RFQState(
                frequency_hz=1.7e6
            )

        observed = MachineState(
            mass_u=target.mass_u,
            parameters=dict(
                target.parameters
            ),
            cup=target.cup,
            stage=6,
            rfq=changed_rfq,
        )

        adapter.state = observed

        return SimpleNamespace(
            observed_state=observed
        )

    monkeypatch.setattr(
        module,
        "apply_state",
        bad_apply,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                10e-9
            ),
    )

    with pytest.raises(
        module.FrozenConfigurationChangedError,
        match="RFQ",
    ):
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
        )


def test_initial_cup1_must_be_valid_signal(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    patch_normal_hardware(
        monkeypatch,
        adapter,
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
        module.InvalidInitialCup1ReferenceError
    ):
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
        )


def test_downstream_below_noise_is_recorded_not_discarded(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    patch_normal_hardware(
        monkeypatch,
        adapter,
    )

    cup1_count = {
        "n": 0
    }

    def fake_measure(
        adapter,
        policy,
        *,
        noise_floor_a=None,
    ):
        cup = adapter.state.cup

        if cup == 1:
            cup1_count["n"] += 1

            return measurement(
                10e-9
            )

        if cup == 6:
            return measurement(
                0.1e-12,
                below_noise=True,
            )

        return measurement(
            5e-9
        )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        fake_measure,
    )

    result = (
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
        )
    )

    assert (
        result.all_downstream_above_noise
        is False
    )

    cup6_point = next(
        point
        for point
        in result.points
        if point.cup == 6
    )

    assert (
        cup6_point.below_noise_floor
        is True
    )


def test_final_state_is_cup1(
    monkeypatch,
):
    start = final_state()

    adapter = FakeAdapter(
        start
    )

    patch_normal_hardware(
        monkeypatch,
        adapter,
    )

    monkeypatch.setattr(
        module,
        "measure_beam_current",
        lambda *args, **kwargs:
            measurement(
                10e-9
            ),
    )

    result = (
        module.characterize_final_transmission(
            adapter,
            start,
            {},
            MeasurementPolicy(),
            policy=(
                module.FinalCharacterizationPolicy(
                    cup_settle_s=0.0
                )
            ),
        )
    )

    assert result.final_state.cup == 1

    assert (
        result.final_state.parameters
        == start.parameters
    )