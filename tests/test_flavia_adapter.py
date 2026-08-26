from dataclasses import dataclass

import pytest

from sirius.flavia_adapter import FlaviaBackendAdapter


@dataclass
class FakeChannel:
    value: object
    timestamp: float = 123.0
    quality: str = "good"
    source: str = "test"


class FakeModel:
    def __init__(self):
        self.channels = {}

    def get(self, name):
        return self.channels.get(name)

    def set(self, name, value):
        self.channels[name] = FakeChannel(value=value)


class FakeCup:
    def __init__(self):
        self.selected = None

    def select_cup(self, cup):
        self.selected = cup


class FakeBackend:
    def __init__(self):
        self.model = FakeModel()
        self.cup = FakeCup()

        self.channel_commands = []
        self.magnet_commands = []
        self.trace_reset_count = 0

    def set_channel(self, channel_name, value):
        self.channel_commands.append(
            (channel_name, value)
        )

    def set_magnet_current(self, current_a):
        self.magnet_commands.append(current_a)

    def reset_keithley_trace(self):
        self.trace_reset_count += 1


def test_regular_parameter_uses_flavia_set_channel():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    adapter.set_parameter(
        "extraction_voltage_v",
        19600.0,
    )

    assert backend.channel_commands == [
        ("cs/extraction/set_u_v", 19600.0)
    ]


def test_magnet_uses_dedicated_backend_method():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    adapter.set_parameter(
        "magnet_current_a",
        42.5,
    )

    assert backend.magnet_commands == [42.5]
    assert backend.channel_commands == []


def test_signed_steerer_is_converted_to_hardware_voltage():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    adapter.set_parameter(
        "steerer_x1_v",
        20.0,
    )

    assert backend.channel_commands == [
        ("steerer/1x/set_u", 270.0)
    ]


def test_negative_signed_steerer_is_converted():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    adapter.set_parameter(
        "steerer_y2_v",
        -75.0,
    )

    assert backend.channel_commands == [
        ("steerer/2y/set_u", 175.0)
    ]


def test_sirius_limits_are_enforced_before_backend_call():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    with pytest.raises(ValueError):
        adapter.set_parameter(
            "sputter_voltage_v",
            9500.0,
        )

    assert backend.channel_commands == []


def test_disabled_future_input_cannot_be_set_yet():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    with pytest.raises(ValueError):
        adapter.set_parameter(
            "hv2_voltage_v",
            1000.0,
        )

    assert backend.channel_commands == []


def test_parameter_readback_uses_measurement_channel():
    backend = FakeBackend()
    backend.model.set(
        "cs/extraction/meas_u_v",
        19598.5,
    )

    adapter = FlaviaBackendAdapter(backend)

    assert adapter.read_parameter(
        "extraction_voltage_v"
    ) == 19598.5


def test_steerer_readback_is_converted_to_signed_coordinate():
    backend = FakeBackend()
    backend.model.set(
        "steerer/1x/meas_u",
        230.0,
    )

    adapter = FlaviaBackendAdapter(backend)

    assert adapter.read_parameter(
        "steerer_x1_v"
    ) == -20.0


def test_cup_selection_uses_cup_worker():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    adapter.select_cup(3)

    assert backend.cup.selected == 3


def test_cup_zero_is_allowed_for_all_cups_out():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    adapter.select_cup(0)

    assert backend.cup.selected == 0


def test_invalid_cup_is_rejected():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    with pytest.raises(ValueError):
        adapter.select_cup(7)


def test_selected_cup_is_read_from_flavia_model():
    backend = FakeBackend()
    backend.model.set(
        "cup/selected",
        4,
    )

    adapter = FlaviaBackendAdapter(backend)

    assert adapter.read_selected_cup() == 4


def test_beam_current_is_read_from_keithley_channel():
    backend = FakeBackend()
    backend.model.set(
        "keithley/current_A",
        8.2e-9,
    )

    adapter = FlaviaBackendAdapter(backend)

    assert adapter.read_beam_current_a() == 8.2e-9


def test_keithley_snapshot_reads_flavia_statistics():
    backend = FakeBackend()

    backend.model.set(
        "keithley/current_A",
        8.2e-9,
    )
    backend.model.set(
        "keithley/stats/mean_nA",
        8.1,
    )
    backend.model.set(
        "keithley/stats/sigma_nA",
        0.2,
    )
    backend.model.set(
        "keithley/stats/n",
        10,
    )
    backend.model.set(
        "keithley/connected",
        True,
    )
    backend.model.set(
        "keithley/mode",
        "TRACE",
    )

    adapter = FlaviaBackendAdapter(backend)

    snapshot = adapter.read_keithley_snapshot()

    assert snapshot.current_a == 8.2e-9
    assert snapshot.mean_na == 8.1
    assert snapshot.sigma_na == 0.2
    assert snapshot.n == 10
    assert snapshot.connected is True
    assert snapshot.mode == "TRACE"


def test_channel_snapshot_preserves_metadata():
    backend = FakeBackend()
    backend.model.set(
        "keithley/current_A",
        1.2e-12,
    )

    adapter = FlaviaBackendAdapter(backend)

    snapshot = adapter.read_channel(
        "keithley/current_A"
    )

    assert snapshot is not None
    assert snapshot.value == 1.2e-12
    assert snapshot.timestamp == 123.0
    assert snapshot.quality == "good"
    assert snapshot.source == "test"


def test_keithley_trace_reset_uses_backend_method():
    backend = FakeBackend()
    adapter = FlaviaBackendAdapter(backend)

    adapter.reset_keithley_trace()

    assert backend.trace_reset_count == 1