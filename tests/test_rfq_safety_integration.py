import ast
from pathlib import Path

import pytest

from sirius.rfq_matching import (
    RFQUnsafeAmplitudeError,
    rfq_vpp_for_q,
)

from sirius.rfq_matching_safe import (
    RFQSafetyCapabilityError,
    _SafetyInterlockedRFQHardware,
)

from sirius.rfq_safety import (
    RFQFaultLatchedError,
    RFQSafetyState,
)


class Hardware:
    def __init__(
        self,
    ):
        self.events = []

        self.rf_enabled = False

        self.frequency_hz = None

        self.rfq_vpp = 0.0

    def set_generator_amplitude_vpp(
        self,
        value,
    ):
        value = float(
            value
        )

        self.events.append(
            (
                "generator",
                value,
            )
        )

        self.rf_enabled = (
            value > 0
        )

    def read_rf_enabled(
        self,
    ):
        return bool(
            self.rf_enabled
        )

    def set_frequency_hz(
        self,
        value,
    ):
        value = float(
            value
        )

        self.events.append(
            (
                "frequency",
                value,
            )
        )

        self.frequency_hz = (
            value
        )

    def set_matching(
        self,
        inductance_uh,
        capacitance_pf,
    ):
        self.events.append(
            (
                "matching",
                float(
                    inductance_uh
                ),
                float(
                    capacitance_pf
                ),
            )
        )

    def read_rfq_vpp(
        self,
    ):
        return float(
            self.rfq_vpp
        )


class HardwareWithoutOffAck:
    def set_generator_amplitude_vpp(
        self,
        value,
    ):
        pass

    def set_frequency_hz(
        self,
        value,
    ):
        pass

    def set_matching(
        self,
        inductance_uh,
        capacitance_pf,
    ):
        pass

    def read_rfq_vpp(
        self,
    ):
        return 0.0


def proxy(
    hardware,
):
    return _SafetyInterlockedRFQHardware(
        hardware,
        mass_u=27.0,
        q_abort_limit=0.9,
        sleeper=lambda seconds:
            None,
    )


def test_safe_wrapper_requires_positive_rf_off_ack_capability():
    with pytest.raises(
        RFQSafetyCapabilityError,
        match="read_rf_enabled",
    ):
        proxy(
            HardwareWithoutOffAck()
        )


def test_frequency_change_turns_rf_off_first():
    hardware = Hardware()

    safe = proxy(
        hardware
    )

    # Establish a safely configured state first.
    safe.set_matching(
        200.0,
        10000.0,
    )

    safe.set_frequency_hz(
        1.8e6
    )

    safe.set_generator_amplitude_vpp(
        10.0
    )

    hardware.events.clear()

    safe.set_frequency_hz(
        1.9e6
    )

    assert hardware.events[
        0
    ] == (
        "generator",
        0.0,
    )

    assert hardware.events[
        -1
    ] == (
        "frequency",
        1.9e6,
    )

    assert (
        safe.safety_controller.state
        == RFQSafetyState.CONFIGURED_OFF
    )


def test_matching_change_turns_rf_off_first():
    hardware = Hardware()

    safe = proxy(
        hardware
    )

    safe.set_matching(
        200.0,
        10000.0,
    )

    safe.set_frequency_hz(
        1.8e6
    )

    safe.set_generator_amplitude_vpp(
        10.0
    )

    hardware.events.clear()

    safe.set_matching(
        180.0,
        12000.0,
    )

    assert hardware.events[
        0
    ] == (
        "generator",
        0.0,
    )

    assert hardware.events[
        -1
    ][
        0
    ] == "matching"


def test_safe_measured_q_transitions_to_rf_on_safe():
    hardware = Hardware()

    safe = proxy(
        hardware
    )

    safe.set_matching(
        200.0,
        10000.0,
    )

    safe.set_frequency_hz(
        1.8e6
    )

    safe.set_generator_amplitude_vpp(
        10.0
    )

    hardware.rfq_vpp = (
        rfq_vpp_for_q(
            27.0,
            1.8e6,
            0.5,
            enforce_operational_limit=False,
        )
    )

    safe.read_rfq_vpp()

    assert (
        safe.safety_controller.state
        == RFQSafetyState.RF_ON_SAFE
    )


def test_unsafe_measured_q_latches_fault_and_requests_rf_off():
    hardware = Hardware()

    safe = proxy(
        hardware
    )

    safe.set_matching(
        200.0,
        10000.0,
    )

    safe.set_frequency_hz(
        1.8e6
    )

    safe.set_generator_amplitude_vpp(
        10.0
    )

    hardware.rfq_vpp = (
        rfq_vpp_for_q(
            27.0,
            1.8e6,
            0.91,
            enforce_operational_limit=False,
        )
    )

    with pytest.raises(
        RFQUnsafeAmplitudeError
    ):
        safe.read_rfq_vpp()

    assert (
        safe.safety_controller.state
        == RFQSafetyState.FAULT_LATCHED
    )

    assert hardware.events[
        -1
    ] == (
        "generator",
        0.0,
    )


def test_off_confirmation_does_not_clear_latched_fault():
    hardware = Hardware()

    safe = proxy(
        hardware
    )

    safe.set_matching(
        200.0,
        10000.0,
    )

    safe.set_frequency_hz(
        1.8e6
    )

    safe.set_generator_amplitude_vpp(
        10.0
    )

    hardware.rfq_vpp = (
        rfq_vpp_for_q(
            27.0,
            1.8e6,
            0.91,
            enforce_operational_limit=False,
        )
    )

    with pytest.raises(
        RFQUnsafeAmplitudeError
    ):
        safe.read_rfq_vpp()

    # Confirming that RF is physically off must not reset the fault.
    safe.set_generator_amplitude_vpp(
        0.0
    )

    assert (
        safe.safety_controller.state
        == RFQSafetyState.FAULT_LATCHED
    )

    with pytest.raises(
        RFQFaultLatchedError
    ):
        safe.set_frequency_hz(
            2.0e6
        )


def test_cup3_does_not_import_raw_rfq_matching():
    path = Path(
        "src/sirius/cup3_optimizer.py"
    )

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    imports = {
        node.module
        for node
        in ast.walk(
            tree
        )
        if isinstance(
            node,
            ast.ImportFrom,
        )
    }

    assert (
        "sirius.rfq_matching_safe"
        in imports
    )

    assert (
        "sirius.rfq_matching"
        not in imports
    )