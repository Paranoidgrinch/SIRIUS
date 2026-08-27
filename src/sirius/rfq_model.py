from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.state import RFQState


ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27

RFQ_R0_M = 5.232e-3

RFQ_OPERATIONAL_Q_MAX = 0.9
RFQ_THEORETICAL_RF_ONLY_Q_MAX = 0.908

LC_MAX_INDUCTANCE_UH = 511.75
LC_MAX_CAPACITANCE_PF = 32757.5


@dataclass(frozen=True)
class RFQEvaluation:
    mass_u: float

    frequency_hz: float
    r0_m: float

    generator_amplitude_vpp: float | None
    voltage_gain: float | None

    nominal_rfq_vpp: float | None
    measured_rfq_vpp: float | None

    q_target: float | None
    q_nominal: float | None
    q_measured: float | None

    authoritative_q: float | None
    authoritative_source: str | None

    q_operational_limit: float

    target_within_limit: bool | None
    nominal_within_limit: bool | None
    measured_within_limit: bool | None


@dataclass(frozen=True)
class LCResonanceEstimate:
    inductance_uh: float
    capacitance_pf: float

    ideal_resonance_frequency_hz: float

    requested_frequency_hz: float | None
    frequency_error_hz: float | None
    relative_frequency_error: float | None


def _positive_finite(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero"
        )

    return value


def _nonnegative_finite(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if value < 0:
        raise ValueError(
            f"{name} must be non-negative"
        )

    return value


def validate_q_target(
    q_value: float,
) -> float:
    q_value = _positive_finite(
        "RFQ q target",
        q_value,
    )

    if q_value > RFQ_OPERATIONAL_Q_MAX:
        raise ValueError(
            "SIRIUS RFQ q target must not exceed "
            f"{RFQ_OPERATIONAL_Q_MAX}"
        )

    return q_value


def mathieu_q_from_v0(
    mass_u: float,
    frequency_hz: float,
    v0_v: float,
    *,
    r0_m: float = RFQ_R0_M,
    charge_state_magnitude: int = 1,
) -> float:
    """
    Mathieu q using

        q = 4 |Q| V0 / (m r0^2 omega^2)

    where V0 is the RF amplitude from zero to peak.

    charge_state_magnitude is the absolute charge state. For the negative
    singly charged ions used by SIRIUS this is 1.
    """

    mass = _positive_finite(
        "Ion mass",
        mass_u,
    )

    frequency = _positive_finite(
        "RFQ frequency",
        frequency_hz,
    )

    radius = _positive_finite(
        "RFQ r0",
        r0_m,
    )

    amplitude = _nonnegative_finite(
        "RFQ V0",
        v0_v,
    )

    if charge_state_magnitude < 1:
        raise ValueError(
            "charge_state_magnitude must be at least 1"
        )

    mass_kg = (
        mass
        * ATOMIC_MASS_UNIT_KG
    )

    charge_c = (
        charge_state_magnitude
        * ELEMENTARY_CHARGE_C
    )

    omega = (
        2.0
        * math.pi
        * frequency
    )

    return (
        4.0
        * charge_c
        * amplitude
        / (
            mass_kg
            * radius ** 2
            * omega ** 2
        )
    )


def mathieu_q_from_vpp(
    mass_u: float,
    frequency_hz: float,
    rfq_vpp: float,
    *,
    r0_m: float = RFQ_R0_M,
    charge_state_magnitude: int = 1,
) -> float:
    """
    Convenience conversion for scope measurements reported as Vpp.

    The formula uses V0 = Vpp / 2.
    """

    vpp = _nonnegative_finite(
        "RFQ Vpp",
        rfq_vpp,
    )

    return mathieu_q_from_v0(
        mass_u,
        frequency_hz,
        vpp / 2.0,
        r0_m=r0_m,
        charge_state_magnitude=(
            charge_state_magnitude
        ),
    )


def rfq_vpp_for_q(
    mass_u: float,
    frequency_hz: float,
    q_value: float,
    *,
    r0_m: float = RFQ_R0_M,
    charge_state_magnitude: int = 1,
    enforce_operational_limit: bool = True,
) -> float:
    """
    Inverse Mathieu relation.

    Returns the RFQ peak-to-peak voltage corresponding to a requested q.
    """

    mass = _positive_finite(
        "Ion mass",
        mass_u,
    )

    frequency = _positive_finite(
        "RFQ frequency",
        frequency_hz,
    )

    radius = _positive_finite(
        "RFQ r0",
        r0_m,
    )

    if enforce_operational_limit:
        q_value = validate_q_target(
            q_value
        )
    else:
        q_value = _positive_finite(
            "RFQ q",
            q_value,
        )

    if charge_state_magnitude < 1:
        raise ValueError(
            "charge_state_magnitude must be at least 1"
        )

    mass_kg = (
        mass
        * ATOMIC_MASS_UNIT_KG
    )

    charge_c = (
        charge_state_magnitude
        * ELEMENTARY_CHARGE_C
    )

    omega = (
        2.0
        * math.pi
        * frequency
    )

    v0 = (
        q_value
        * mass_kg
        * radius ** 2
        * omega ** 2
        / (
            4.0
            * charge_c
        )
    )

    return (
        2.0
        * v0
    )


def nominal_rfq_vpp(
    generator_amplitude_vpp: float,
    voltage_gain: float,
) -> float:
    """
    Nominal RFQ voltage from generator Vpp and an assumed voltage gain.

    This is diagnostic only. The measured resonant RFQ Vpp remains
    authoritative whenever available.
    """

    generator = _nonnegative_finite(
        "Generator amplitude",
        generator_amplitude_vpp,
    )

    gain = _positive_finite(
        "RF voltage gain",
        voltage_gain,
    )

    return (
        generator
        * gain
    )


def generator_vpp_for_q(
    mass_u: float,
    frequency_hz: float,
    q_value: float,
    voltage_gain: float,
    *,
    r0_m: float = RFQ_R0_M,
) -> float:
    """
    Nominal generator Vpp required for q assuming a specified fixed gain.

    This is only a starting estimate because the real resonant gain may
    depend on frequency, matching, load and operating conditions.
    """

    gain = _positive_finite(
        "RF voltage gain",
        voltage_gain,
    )

    required_rfq_vpp = rfq_vpp_for_q(
        mass_u,
        frequency_hz,
        q_value,
        r0_m=r0_m,
    )

    return (
        required_rfq_vpp
        / gain
    )


def _within_operational_limit(
    q_value: float | None,
) -> bool | None:
    if q_value is None:
        return None

    return (
        0.0
        < q_value
        <= RFQ_OPERATIONAL_Q_MAX
    )


def evaluate_rfq(
    mass_u: float,
    rfq_state: RFQState,
    *,
    voltage_gain: float | None = None,
    r0_m: float = RFQ_R0_M,
) -> RFQEvaluation:
    """
    Evaluate nominal and measured Mathieu q values.

    Authority rule:

        measured RFQ Vpp available
            -> q_measured is authoritative

        no measured RFQ Vpp
            -> no authoritative q is claimed

    q_nominal remains a useful reference but never replaces a missing
    scope measurement.
    """

    mass = _positive_finite(
        "Ion mass",
        mass_u,
    )

    if rfq_state.frequency_hz is None:
        raise ValueError(
            "RFQ frequency is required"
        )

    frequency = _positive_finite(
        "RFQ frequency",
        rfq_state.frequency_hz,
    )

    target_q = rfq_state.q_target

    if target_q is not None:
        target_q = validate_q_target(
            target_q
        )

    nominal_vpp = None
    nominal_q = None

    if (
        rfq_state.generator_amplitude_vpp
        is not None
        and voltage_gain is not None
    ):
        nominal_vpp = nominal_rfq_vpp(
            rfq_state.generator_amplitude_vpp,
            voltage_gain,
        )

        nominal_q = mathieu_q_from_vpp(
            mass,
            frequency,
            nominal_vpp,
            r0_m=r0_m,
        )

    measured_vpp = (
        rfq_state.rfq_vpp_measured
    )

    measured_q = None

    if measured_vpp is not None:
        measured_vpp = _nonnegative_finite(
            "Measured RFQ Vpp",
            measured_vpp,
        )

        measured_q = mathieu_q_from_vpp(
            mass,
            frequency,
            measured_vpp,
            r0_m=r0_m,
        )

    if measured_q is not None:
        authoritative_q = measured_q
        authoritative_source = (
            "measured_rfq_vpp"
        )

    else:
        authoritative_q = None
        authoritative_source = None

    return RFQEvaluation(
        mass_u=mass,
        frequency_hz=frequency,
        r0_m=float(r0_m),
        generator_amplitude_vpp=(
            rfq_state.generator_amplitude_vpp
        ),
        voltage_gain=voltage_gain,
        nominal_rfq_vpp=nominal_vpp,
        measured_rfq_vpp=measured_vpp,
        q_target=target_q,
        q_nominal=nominal_q,
        q_measured=measured_q,
        authoritative_q=(
            authoritative_q
        ),
        authoritative_source=(
            authoritative_source
        ),
        q_operational_limit=(
            RFQ_OPERATIONAL_Q_MAX
        ),
        target_within_limit=(
            _within_operational_limit(
                target_q
            )
        ),
        nominal_within_limit=(
            _within_operational_limit(
                nominal_q
            )
        ),
        measured_within_limit=(
            _within_operational_limit(
                measured_q
            )
        ),
    )


def lc_settings_within_range(
    inductance_uh: float,
    capacitance_pf: float,
) -> bool:
    inductance = _nonnegative_finite(
        "LC inductance",
        inductance_uh,
    )

    capacitance = _nonnegative_finite(
        "LC capacitance",
        capacitance_pf,
    )

    return (
        inductance
        <= LC_MAX_INDUCTANCE_UH
        and capacitance
        <= LC_MAX_CAPACITANCE_PF
    )


def ideal_lc_resonance_frequency_hz(
    inductance_uh: float,
    capacitance_pf: float,
) -> float:
    """
    Ideal lumped-LC resonance estimate.

        f0 = 1 / (2 pi sqrt(L C))

    This is only a starting estimate. Real RFQ resonance includes the
    transformer, RFQ capacitance, wiring, parasitics and load, so the
    experimentally measured resonance remains authoritative.
    """

    inductance = _positive_finite(
        "LC inductance",
        inductance_uh,
    )

    capacitance = _positive_finite(
        "LC capacitance",
        capacitance_pf,
    )

    if not lc_settings_within_range(
        inductance,
        capacitance,
    ):
        raise ValueError(
            "LC settings exceed available matching-network range"
        )

    inductance_h = (
        inductance
        * 1e-6
    )

    capacitance_f = (
        capacitance
        * 1e-12
    )

    return (
        1.0
        / (
            2.0
            * math.pi
            * math.sqrt(
                inductance_h
                * capacitance_f
            )
        )
    )


def evaluate_lc_resonance(
    inductance_uh: float,
    capacitance_pf: float,
    *,
    requested_frequency_hz: float | None = None,
) -> LCResonanceEstimate:
    resonance = (
        ideal_lc_resonance_frequency_hz(
            inductance_uh,
            capacitance_pf,
        )
    )

    frequency_error = None
    relative_error = None

    if requested_frequency_hz is not None:
        requested = _positive_finite(
            "Requested RFQ frequency",
            requested_frequency_hz,
        )

        frequency_error = (
            resonance
            - requested
        )

        relative_error = (
            frequency_error
            / requested
        )

    return LCResonanceEstimate(
        inductance_uh=float(
            inductance_uh
        ),
        capacitance_pf=float(
            capacitance_pf
        ),
        ideal_resonance_frequency_hz=(
            resonance
        ),
        requested_frequency_hz=(
            requested_frequency_hz
        ),
        frequency_error_hz=(
            frequency_error
        ),
        relative_frequency_error=(
            relative_error
        ),
    )