from __future__ import annotations

from dataclasses import dataclass


STEERER_BIAS_V = 250.0


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    flavia_channel: str | None
    minimum: float
    maximum: float
    unit: str
    optimizable: bool = True
    enabled: bool = True


PARAMETERS: dict[str, ParameterDefinition] = {
    "sputter_voltage_v": ParameterDefinition(
        "sputter_voltage_v", "cs/sputter/set_u_v", 0.0, 9000.0, "V"
    ),
    "extraction_voltage_v": ParameterDefinition(
        "extraction_voltage_v", "cs/extraction/set_u_v", 0.0, 25000.0, "V"
    ),
    "einzel_lens_voltage_v": ParameterDefinition(
        "einzel_lens_voltage_v", "cs/einzellens/set_u_v", 0.0, 25000.0, "V"
    ),
    "magnet_current_a": ParameterDefinition(
        "magnet_current_a", None, 0.0, 120.0, "A"
    ),
    "lens2_voltage_v": ParameterDefinition(
        "lens2_voltage_v", "cs/lens2/set_u_v", 0.0, 12500.0, "V"
    ),
    "steerer_x1_v": ParameterDefinition(
        "steerer_x1_v", "steerer/1x/set_u", -250.0, 250.0, "V"
    ),
    "steerer_y1_v": ParameterDefinition(
        "steerer_y1_v", "steerer/1y/set_u", -250.0, 250.0, "V"
    ),
    "ion_cooler_voltage_v": ParameterDefinition(
        "ion_cooler_voltage_v", "cs/ion_cooler/set_u_v", 0.0, 34000.0, "V"
    ),
    "deceleration_voltage_v": ParameterDefinition(
        "deceleration_voltage_v", "hv/1/set_v", 0.0, 6500.0, "V"
    ),
    "hv2_voltage_v": ParameterDefinition(
        "hv2_voltage_v", "hv/2/set_v", 0.0, 6500.0, "V",
        optimizable=False,
        enabled=False,
    ),
    "hv3_voltage_v": ParameterDefinition(
        "hv3_voltage_v", "hv/3/set_v", 0.0, 6500.0, "V",
        optimizable=False,
        enabled=False,
    ),
    "acceleration_voltage_v": ParameterDefinition(
        "acceleration_voltage_v", "hv/4/set_v", 0.0, 6500.0, "V"
    ),
    "guidefield1_voltage_v": ParameterDefinition(
        "guidefield1_voltage_v", "psu/1/set_v", 0.0, 30.0, "V"
    ),
    "guidefield2_voltage_v": ParameterDefinition(
        "guidefield2_voltage_v", "psu/2/set_v", 0.0, 75.0, "V"
    ),
    "quadrupole1_voltage_v": ParameterDefinition(
        "quadrupole1_voltage_v", "cs/qp1/set_u_v", 0.0, 6000.0, "V"
    ),
    "quadrupole2_voltage_v": ParameterDefinition(
        "quadrupole2_voltage_v", "cs/qp2/set_u_v", 0.0, 6000.0, "V"
    ),
    "quadrupole3_voltage_v": ParameterDefinition(
        "quadrupole3_voltage_v", "cs/qp3/set_u_v", 0.0, 6000.0, "V"
    ),
    "steerer_x2_v": ParameterDefinition(
        "steerer_x2_v", "steerer/2x/set_u", -250.0, 250.0, "V"
    ),
    "steerer_y2_v": ParameterDefinition(
        "steerer_y2_v", "steerer/2y/set_u", -250.0, 250.0, "V"
    ),
    "esa_voltage_v": ParameterDefinition(
        "esa_voltage_v", "cs/esa/set_u_v", 0.0, 3000.0, "V"
    ),
    "steerer_x3_v": ParameterDefinition(
        "steerer_x3_v", "steerer/3x/set_u", -250.0, 250.0, "V"
    ),
    "steerer_y3_v": ParameterDefinition(
        "steerer_y3_v", "steerer/3y/set_u", -250.0, 250.0, "V"
    ),
    "lens4_voltage_v": ParameterDefinition(
        "lens4_voltage_v", "cs/lens4/set_u_v", 0.0, 10000.0, "V"
    ),
}


def validate_parameter(name: str, value: float) -> float:
    definition = PARAMETERS[name]

    if not definition.enabled:
        raise ValueError(f"{name} is currently disabled")

    if not definition.minimum <= value <= definition.maximum:
        raise ValueError(
            f"{name}={value} outside allowed range "
            f"{definition.minimum}..{definition.maximum} {definition.unit}"
        )

    return value


def sirius_steerer_to_hardware(value_v: float) -> float:
    if not -250.0 <= value_v <= 250.0:
        raise ValueError("SIRIUS steerer coordinate must be between -250 and +250 V")

    return STEERER_BIAS_V + value_v


def hardware_steerer_to_sirius(value_v: float) -> float:
    return value_v - STEERER_BIAS_V
