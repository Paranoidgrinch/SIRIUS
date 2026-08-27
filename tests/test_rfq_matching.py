import math

import pytest

from sirius.rfq_matching import (
    LCSetting,
    RFQMatchingPolicy,
    RFQTargetQPolicy,
    RFQUnsafeAmplitudeError,
    rank_lc_candidates,
    search_rfq_resonance,
    set_target_q,
)
from sirius.rfq_model import (
    mathieu_q_from_vpp,
)


class FakeRFQHardware:
    def __init__(
        self,
        *,
        gain=40.0,
    ):
        self.frequency_hz = 1.6e6
        self.generator_vpp = 0.0

        self.inductance_uh = 100.0
        self.capacitance_pf = 100.0

        self.gain = gain

        self.commands = []

    def set_frequency_hz(
        self,
        value,
    ):
        self.frequency_hz = float(
            value
        )

        self.commands.append(
            (
                "frequency",
                self.frequency_hz,
            )
        )

    def set_generator_amplitude_vpp(
        self,
        value,
    ):
        self.generator_vpp = float(
            value
        )

        self.commands.append(
            (
                "generator",
                self.generator_vpp,
            )
        )

    def set_matching(
        self,
        inductance_uh,
        capacitance_pf,
    ):
        self.inductance_uh = float(
            inductance_uh
        )

        self.capacitance_pf = float(
            capacitance_pf
        )

        self.commands.append(
            (
                "matching",
                self.inductance_uh,
                self.capacitance_pf,
            )
        )

    def read_rfq_vpp(
        self,
    ):
        if (
            self.inductance_uh == 100.0
            and self.capacitance_pf == 100.0
        ):
            resonance = 1.590e6
            matching_factor = 1.0

        else:
            resonance = 1.700e6
            matching_factor = 0.55

        width = 30_000.0

        detuning = (
            self.frequency_hz
            - resonance
        ) / width

        resonance_response = (
            1.0
            / (
                1.0
                + detuning ** 2
            )
        )

        return (
            self.generator_vpp
            * self.gain
            * matching_factor
            * resonance_response
        )


def matching_policy(
    *,
    probe=5.0,
):
    return RFQMatchingPolicy(
        probe_generator_vpp=probe,
        requested_frequency_hz=1.6e6,
        frequency_half_width_hz=100e3,
        coarse_frequency_step_hz=25e3,
        fine_frequency_step_hz=5e3,
        top_lc_candidates=2,
        measurements_per_point=3,
    )


def candidates():
    return (
        LCSetting(
            inductance_uh=100.0,
            capacitance_pf=100.0,
        ),
        LCSetting(
            inductance_uh=80.0,
            capacitance_pf=100.0,
        ),
    )


def test_lc_candidates_are_ranked_by_ideal_frequency():
    ranked = rank_lc_candidates(
        candidates(),
        1.6e6,
    )

    assert ranked[0] == LCSetting(
        inductance_uh=100.0,
        capacitance_pf=100.0,
    )


def test_invalid_lc_setting_is_rejected():
    with pytest.raises(
        ValueError
    ):
        LCSetting(
            inductance_uh=600.0,
            capacitance_pf=100.0,
        ).validate()


def test_resonance_search_finds_correct_matching_candidate():
    hardware = FakeRFQHardware()

    result = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    assert (
        result.best_setting
        == LCSetting(
            inductance_uh=100.0,
            capacitance_pf=100.0,
        )
    )


def test_resonance_search_refines_frequency():
    hardware = FakeRFQHardware()

    result = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    assert (
        result.best_frequency_hz
        == pytest.approx(
            1.590e6
        )
    )


def test_scope_vpp_is_used_for_q():
    hardware = FakeRFQHardware()

    result = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    expected_q = (
        mathieu_q_from_vpp(
            60.0,
            result.best_frequency_hz,
            result.best_measured_rfq_vpp,
        )
    )

    assert (
        result.best_measured_q
        == pytest.approx(
            expected_q
        )
    )


def test_successful_search_turns_probe_off_by_default():
    hardware = FakeRFQHardware()

    search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    assert hardware.generator_vpp == 0.0


def test_best_matching_and_frequency_remain_selected():
    hardware = FakeRFQHardware()

    result = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    assert (
        hardware.inductance_uh
        == result.best_setting.inductance_uh
    )

    assert (
        hardware.capacitance_pf
        == result.best_setting.capacitance_pf
    )

    assert (
        hardware.frequency_hz
        == result.best_frequency_hz
    )


def test_unsafe_probe_amplitude_aborts_and_turns_rf_off():
    hardware = FakeRFQHardware(
        gain=300.0
    )

    with pytest.raises(
        RFQUnsafeAmplitudeError
    ):
        search_rfq_resonance(
            hardware,
            mass_u=60.0,
            lc_candidates=candidates(),
            policy=matching_policy(
                probe=5.0
            ),
            sleeper=lambda _: None,
        )

    assert hardware.generator_vpp == 0.0


def test_target_q_controller_reaches_target_from_measured_vpp():
    hardware = FakeRFQHardware(
        gain=40.0
    )

    matching = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    target = 0.45

    result = set_target_q(
        hardware,
        matching,
        target,
        RFQTargetQPolicy(
            generator_max_vpp=20.0,
            initial_generator_vpp=2.0,
            relative_q_tolerance=0.01,
            max_iterations=10,
            measurements_per_iteration=3,
        ),
        sleeper=lambda _: None,
    )

    assert result.measured_q == pytest.approx(
        target,
        rel=0.01,
    )

    assert (
        hardware.generator_vpp
        == result.generator_amplitude_vpp
    )


def test_target_q_controller_uses_gradual_scale_up():
    hardware = FakeRFQHardware(
        gain=40.0
    )

    matching = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    result = set_target_q(
        hardware,
        matching,
        0.45,
        RFQTargetQPolicy(
            generator_max_vpp=20.0,
            initial_generator_vpp=2.0,
            maximum_scale_up=1.5,
            relative_q_tolerance=0.01,
            max_iterations=10,
        ),
        sleeper=lambda _: None,
    )

    commands = [
        item.generator_amplitude_vpp
        for item in result.iterations
    ]

    for previous, current in zip(
        commands,
        commands[1:],
    ):
        assert (
            current
            <= previous * 1.5 + 1e-12
        )


def test_target_above_q_0_9_is_rejected():
    hardware = FakeRFQHardware()

    matching = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    with pytest.raises(
        ValueError
    ):
        set_target_q(
            hardware,
            matching,
            0.91,
            RFQTargetQPolicy(
                generator_max_vpp=20.0,
                initial_generator_vpp=2.0,
            ),
            sleeper=lambda _: None,
        )


def test_target_q_failure_turns_rf_off():
    hardware = FakeRFQHardware(
        gain=1.0
    )

    matching = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    with pytest.raises(
        Exception
    ):
        set_target_q(
            hardware,
            matching,
            0.8,
            RFQTargetQPolicy(
                generator_max_vpp=2.1,
                initial_generator_vpp=2.0,
                max_iterations=3,
            ),
            sleeper=lambda _: None,
        )

    assert hardware.generator_vpp == 0.0


def test_all_resonance_points_are_preserved():
    hardware = FakeRFQHardware()

    result = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    assert len(
        result.points
    ) > 0

    assert {
        point.scan_level
        for point in result.points
    } == {
        "coarse",
        "fine",
    }


def test_median_scope_measurement_rejects_single_glitch():
    class GlitchHardware(
        FakeRFQHardware
    ):
        def __init__(self):
            super().__init__()
            self.index = 0

        def read_rfq_vpp(self):
            normal = super().read_rfq_vpp()

            self.index += 1

            if self.index % 3 == 2:
                return normal * 1.2

            return normal

    hardware = GlitchHardware()

    result = search_rfq_resonance(
        hardware,
        mass_u=60.0,
        lc_candidates=candidates(),
        policy=matching_policy(),
        sleeper=lambda _: None,
    )

    best = max(
        result.points,
        key=lambda point: (
            point.measured_rfq_vpp
        ),
    )

    assert len(
        best.samples_vpp
    ) == 3

    assert (
        best.measured_rfq_vpp
        == pytest.approx(
            statistics_median(
                best.samples_vpp
            )
        )
    )


def statistics_median(
    values,
):
    ordered = sorted(
        values
    )

    return ordered[
        len(ordered) // 2
    ]