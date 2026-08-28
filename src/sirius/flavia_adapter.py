from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sirius.parameters import (
    PARAMETERS,
    hardware_steerer_to_sirius,
    sirius_steerer_to_hardware,
    validate_parameter,
)


STEERER_PARAMETERS = {
    "steerer_x1_v",
    "steerer_y1_v",
    "steerer_x2_v",
    "steerer_y2_v",
    "steerer_x3_v",
    "steerer_y3_v",
}


READBACK_CHANNELS: dict[str, str] = {
    "sputter_voltage_v": "cs/sputter/meas_u_v",
    "extraction_voltage_v": "cs/extraction/meas_u_v",
    "einzel_lens_voltage_v": "cs/einzellens/meas_u_v",
    "magnet_current_a": "magnet_current_meas",
    "lens2_voltage_v": "cs/lens2/meas_u_v",
    "steerer_x1_v": "steerer/1x/meas_u",
    "steerer_y1_v": "steerer/1y/meas_u",
    "ion_cooler_voltage_v": "cs/ion_cooler/meas_u_v",
    "deceleration_voltage_v": "hv/1/meas_v",
    "hv2_voltage_v": "hv/2/meas_v",
    "hv3_voltage_v": "hv/3/meas_v",
    "acceleration_voltage_v": "hv/4/meas_v",
    "guidefield1_voltage_v": "psu/1/meas_v",
    "guidefield2_voltage_v": "psu/2/meas_v",
    "quadrupole1_voltage_v": "cs/qp1/meas_u_v",
    "quadrupole2_voltage_v": "cs/qp2/meas_u_v",
    "quadrupole3_voltage_v": "cs/qp3/meas_u_v",
    "steerer_x2_v": "steerer/2x/meas_u",
    "steerer_y2_v": "steerer/2y/meas_u",
    "esa_voltage_v": "cs/esa/meas_u_v",
    "steerer_x3_v": "steerer/3x/meas_u",
    "steerer_y3_v": "steerer/3y/meas_u",
    "lens4_voltage_v": "cs/lens4/meas_u_v",
}


@dataclass(frozen=True)
class ChannelSnapshot:
    value: Any
    timestamp: float | None
    quality: str | None
    source: str | None


@dataclass(frozen=True)
class KeithleySnapshot:
    current_a: float | None
    mean_na: float | None
    sigma_na: float | None
    n: int | None
    connected: bool | None
    mode: str | None


class FlaviaBackendAdapter:
    """
    Thin SIRIUS interface to an already running FLAVIA Backend.

    SIRIUS deliberately does not create or communicate with hardware
    workers directly. FLAVIA remains responsible for MQTT, magnet TCP,
    cup HTTP and Keithley communication.
    """

    def __init__(self, backend: Any):
        self.backend = backend

    def read_channel(self, channel_name: str) -> ChannelSnapshot | None:
        channel = self.backend.model.get(channel_name)

        if channel is None:
            return None

        return ChannelSnapshot(
            value=channel.value,
            timestamp=getattr(channel, "timestamp", None),
            quality=getattr(channel, "quality", None),
            source=getattr(channel, "source", None),
        )

    def read_channel_value(
        self,
        channel_name: str,
        default: Any = None,
    ) -> Any:
        snapshot = self.read_channel(channel_name)

        if snapshot is None or snapshot.value is None:
            return default

        return snapshot.value

    def set_parameter(self, name: str, value: float) -> None:
        """
        Set one SIRIUS parameter through the FLAVIA Backend.

        Steerer coordinates are represented in SIRIUS relative to the
        fixed 250 V bias:
            -250 V SIRIUS ->   0 V FLAVIA
               0 V SIRIUS -> 250 V FLAVIA
            +250 V SIRIUS -> 500 V FLAVIA
        """
        value = float(validate_parameter(name, float(value)))

        if name == "magnet_current_a":
            self.backend.set_magnet_current(value)
            return

        definition = PARAMETERS[name]

        if definition.flavia_channel is None:
            raise ValueError(
                f"{name} has no FLAVIA set channel"
            )

        hardware_value = value

        if name in STEERER_PARAMETERS:
            hardware_value = sirius_steerer_to_hardware(value)

        self.backend.set_channel(
            definition.flavia_channel,
            hardware_value,
        )

    def read_parameter(self, name: str) -> float | None:
        """
        Read the physical FLAVIA readback for a SIRIUS parameter.

        Steerer readbacks are translated back into the signed SIRIUS
        coordinate system.
        """
        if name not in PARAMETERS:
            raise KeyError(f"Unknown SIRIUS parameter: {name}")

        channel_name = READBACK_CHANNELS.get(name)

        if channel_name is None:
            return None

        value = self.read_channel_value(channel_name)

        if value is None:
            return None

        result = float(value)

        if name in STEERER_PARAMETERS:
            result = hardware_steerer_to_sirius(result)

        return result

    def select_cup(self, cup: int) -> None:
        """
        Select a beamline Faraday cup.

        Cup 0 means that no measurement cup is inserted.
        SIRIUS currently uses cups 1 through 6 for beam measurements.
        """
        cup = int(cup)

        if not 0 <= cup <= 6:
            raise ValueError("Cup must be between 0 and 6")

        self.backend.cup.select_cup(cup)

    def read_selected_cup(self):
        """
        Return the raw FLAVIA selected-cup readback.

        Validation and normalization are deliberately handled by the
        SIRIUS cup-acknowledgement layer. This prevents malformed values
        such as NaN, infinity, or non-integer cup identifiers from being
        silently coerced here.
        """
        return self.read_channel_value("cup/selected")

    def read_beam_current_a(self) -> float | None:
        """
        Return the latest Keithley beam-current magnitude in ampere.
        """
        value = self.read_channel_value("keithley/current_A")

        if value is None:
            return None

        return float(value)

    def read_keithley_snapshot(self) -> KeithleySnapshot:
        """
        Read the current Keithley state from the shared FLAVIA DataModel.

        These values are diagnostic snapshots only. The later SIRIUS
        measurement engine will perform its own adaptive statistical
        decision-making from the current stream.
        """
        current_a = self.read_channel_value("keithley/current_A")
        mean_na = self.read_channel_value("keithley/stats/mean_nA")
        sigma_na = self.read_channel_value("keithley/stats/sigma_nA")
        n = self.read_channel_value("keithley/stats/n")
        connected = self.read_channel_value("keithley/connected")
        mode = self.read_channel_value("keithley/mode")

        return KeithleySnapshot(
            current_a=None if current_a is None else float(current_a),
            mean_na=None if mean_na is None else float(mean_na),
            sigma_na=None if sigma_na is None else float(sigma_na),
            n=None if n is None else int(n),
            connected=None if connected is None else bool(connected),
            mode=None if mode is None else str(mode),
        )

    def reset_keithley_trace(self) -> None:
        self.backend.reset_keithley_trace()