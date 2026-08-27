import pytest

from sirius.rfq_model import (
    RFQ_OPERATIONAL_Q_MAX,
    evaluate_lc_resonance,
    evaluate_rfq,
    generator_vpp_for_q,
    ideal_lc_resonance_frequency_hz,
    lc_settings_within_range,
    mathieu_q_from_vpp,
    nominal_rfq_vpp,
    rfq_vpp_for_q,
    validate_q_target,
)
from sirius.state import RFQState


def test_q_from_500vpp_for_mass60_at_1_8mhz():
    q = mathieu_q_from_vpp(
        mass_u=60.0,
        frequency_hz=1.8e6,
        rfq_vpp=500.0,
    )

    assert q == pytest.approx(
        0.4592718512,
        rel=1e-9,
    )


def test_q_inverse_recovers_vpp():
    q = mathieu_q_from_vpp(
        60.0,
        1.8e6,
        500.0,
    )

    recovered = rfq_vpp_for_q(
        60.0,
        1.8e6,
        q,
    )

    assert recovered == pytest.approx(
        500.0,
        rel=1e-10,
    )


def test_q_scales_inversely_with_mass():
    q60 = mathieu_q_from_vpp(
        60.0,
        1.8e6,
        500.0,
    )

    q120 = mathieu_q_from_vpp(
        120.0,
        1.8e6,
        500.0,
    )

    assert q120 == pytest.approx(
        q60 / 2.0
    )


def test_q_scales_with_inverse_frequency_squared():
    q_low = mathieu_q_from_vpp(
        60.0,
        1.0e6,
        500.0,
    )

    q_high = mathieu_q_from_vpp(
        60.0,
        2.0e6,
        500.0,
    )

    assert q_high == pytest.approx(
        q_low / 4.0
    )


def test_operational_q_target_above_0_9_is_rejected():
    with pytest.raises(
        ValueError
    ):
        validate_q_target(
            0.91
        )

    assert validate_q_target(
        0.9
    ) == 0.9


def test_1000vpp_at_mass60_1_8mhz_exceeds_operational_limit():
    q = mathieu_q_from_vpp(
        60.0,
        1.8e6,
        1000.0,
    )

    assert q > RFQ_OPERATIONAL_Q_MAX


def test_measured_vpp_is_authoritative():
    rfq = RFQState(
        frequency_hz=1.8e6,
        generator_amplitude_vpp=10.0,
        rfq_vpp_measured=500.0,
        q_target=0.45,
    )

    result = evaluate_rfq(
        60.0,
        rfq,
        voltage_gain=30.0,
    )

    # Nominal:
    # generator 10 Vpp * gain 30 = 300 Vpp.
    assert result.nominal_rfq_vpp == pytest.approx(
        300.0
    )

    # Measured scope amplitude is 500 Vpp.
    assert result.measured_rfq_vpp == pytest.approx(
        500.0
    )

    assert result.q_nominal == pytest.approx(
        mathieu_q_from_vpp(
            60.0,
            1.8e6,
            300.0,
        )
    )

    assert result.q_measured == pytest.approx(
        mathieu_q_from_vpp(
            60.0,
            1.8e6,
            500.0,
        )
    )

    assert (
        result.authoritative_q
        == result.q_measured
    )

    assert (
        result.authoritative_source
        == "measured_rfq_vpp"
    )


def test_nominal_q_is_not_promoted_when_scope_measurement_missing():
    rfq = RFQState(
        frequency_hz=1.8e6,
        generator_amplitude_vpp=10.0,
    )

    result = evaluate_rfq(
        60.0,
        rfq,
        voltage_gain=30.0,
    )

    assert result.q_nominal is not None
    assert result.q_measured is None

    assert result.authoritative_q is None
    assert result.authoritative_source is None


def test_measured_overlimit_q_is_reported_as_unsafe():
    rfq = RFQState(
        frequency_hz=1.8e6,
        rfq_vpp_measured=1000.0,
    )

    result = evaluate_rfq(
        60.0,
        rfq,
    )

    assert result.q_measured > 0.9

    assert (
        result.measured_within_limit
        is False
    )


def test_nominal_gain_conversion():
    assert nominal_rfq_vpp(
        generator_amplitude_vpp=10.0,
        voltage_gain=50.0,
    ) == pytest.approx(
        500.0
    )


def test_generator_vpp_for_q_is_nominal_inverse():
    target_q = mathieu_q_from_vpp(
        60.0,
        1.8e6,
        500.0,
    )

    generator = generator_vpp_for_q(
        60.0,
        1.8e6,
        target_q,
        voltage_gain=50.0,
    )

    assert generator == pytest.approx(
        10.0
    )


def test_ideal_lc_resonance():
    frequency = (
        ideal_lc_resonance_frequency_hz(
            inductance_uh=100.0,
            capacitance_pf=100.0,
        )
    )

    assert frequency == pytest.approx(
        1.5915494309e6,
        rel=1e-9,
    )


def test_lc_hardware_ranges():
    assert lc_settings_within_range(
        511.75,
        32757.5,
    ) is True

    assert lc_settings_within_range(
        512.0,
        100.0,
    ) is False

    assert lc_settings_within_range(
        100.0,
        33000.0,
    ) is False


def test_zero_l_or_c_cannot_define_resonance():
    with pytest.raises(
        ValueError
    ):
        ideal_lc_resonance_frequency_hz(
            0.0,
            100.0,
        )

    with pytest.raises(
        ValueError
    ):
        ideal_lc_resonance_frequency_hz(
            100.0,
            0.0,
        )


def test_lc_evaluation_reports_frequency_error():
    result = evaluate_lc_resonance(
        inductance_uh=100.0,
        capacitance_pf=100.0,
        requested_frequency_hz=1.6e6,
    )

    assert (
        result.ideal_resonance_frequency_hz
        < 1.6e6
    )

    assert result.frequency_error_hz < 0

    assert result.relative_frequency_error < 0


def test_frequency_is_required_for_rfq_evaluation():
    with pytest.raises(
        ValueError
    ):
        evaluate_rfq(
            60.0,
            RFQState(),
        )