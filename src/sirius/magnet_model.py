from __future__ import annotations

import math
from dataclasses import dataclass

from sirius.physics import beam_energy_ev


# Exact constants currently used by the FLAVIA magnet calculator.
ATOMIC_MASS_UNIT_KG = 1.66054e-27
ELEMENTARY_CHARGE_C = 1.60218e-19

MAGNET_RADIUS_M = 0.5

MAGNET_CALIBRATION_SLOPE_KG_PER_A = 0.10886
MAGNET_CALIBRATION_OFFSET_KG = 0.0348763

MAGNET_MAX_CURRENT_A = 120.0


@dataclass(frozen=True)
class MagnetPrediction:
    mass_u: float
    beam_energy_ev: float

    magnetic_field_t: float
    magnetic_field_kg: float

    calculated_current_a: float
    command_current_a: float

    current_clamped: bool


def _validate_mass(
    mass_u: float,
) -> float:
    mass = float(mass_u)

    if not math.isfinite(mass):
        raise ValueError(
            "Ion mass must be finite"
        )

    if mass <= 0:
        raise ValueError(
            "Ion mass must be greater than zero"
        )

    return mass


def magnetic_field_for_beam(
    mass_u: float,
    beam_energy_ev_value: float,
) -> float:
    """
    Calculate the required analyzing-magnet field in tesla for a
    singly charged ion.

    This reproduces the physical calculation used by the current
    FLAVIA MagnetCalculatorDialog.
    """

    mass = _validate_mass(
        mass_u
    )

    energy_ev = float(
        beam_energy_ev_value
    )

    if not math.isfinite(
        energy_ev
    ):
        raise ValueError(
            "Beam energy must be finite"
        )

    if energy_ev < 0:
        raise ValueError(
            "Beam energy must be non-negative"
        )

    if energy_ev == 0:
        return 0.0

    mass_kg = (
        mass
        * ATOMIC_MASS_UNIT_KG
    )

    numerator = math.sqrt(
        2.0
        * energy_ev
        * ELEMENTARY_CHARGE_C
        * mass_kg
    )

    denominator = (
        ELEMENTARY_CHARGE_C
        * MAGNET_RADIUS_M
    )

    return (
        numerator
        / denominator
    )


def magnet_current_for_field_kg(
    magnetic_field_kg: float,
) -> float:
    """
    Convert magnetic field in kG to magnet current using the empirical
    calibration currently used by FLAVIA.
    """

    field = float(
        magnetic_field_kg
    )

    if not math.isfinite(field):
        raise ValueError(
            "Magnetic field must be finite"
        )

    if field < 0:
        raise ValueError(
            "Magnetic field must be non-negative"
        )

    return (
        field
        + MAGNET_CALIBRATION_OFFSET_KG
    ) / MAGNET_CALIBRATION_SLOPE_KG_PER_A


def magnetic_field_kg_for_current(
    current_a: float,
) -> float:
    """
    Inverse of the FLAVIA empirical magnet calibration.

    At very small currents the linear fit mathematically predicts a
    negative field. SIRIUS clips that unphysical extrapolation to zero.
    """

    current = float(
        current_a
    )

    if not math.isfinite(current):
        raise ValueError(
            "Magnet current must be finite"
        )

    if current < 0:
        raise ValueError(
            "Magnet current must be non-negative"
        )

    field_kg = (
        MAGNET_CALIBRATION_SLOPE_KG_PER_A
        * current
        - MAGNET_CALIBRATION_OFFSET_KG
    )

    return max(
        0.0,
        field_kg,
    )


def predict_magnet(
    mass_u: float,
    sputter_voltage_v: float,
    extraction_voltage_v: float,
) -> MagnetPrediction:
    """
    Predict the analyzing-magnet current from ion mass and source energy.

    The returned calculated_current_a is the raw FLAVIA prediction.
    command_current_a is clipped to the SIRIUS/FLAVIA 0..120 A range.
    """

    energy = beam_energy_ev(
        sputter_voltage_v,
        extraction_voltage_v,
    )

    field_t = magnetic_field_for_beam(
        mass_u,
        energy,
    )

    field_kg = (
        field_t
        * 10.0
    )

    calculated_current = (
        magnet_current_for_field_kg(
            field_kg
        )
    )

    command_current = min(
        max(
            calculated_current,
            0.0,
        ),
        MAGNET_MAX_CURRENT_A,
    )

    return MagnetPrediction(
        mass_u=float(mass_u),
        beam_energy_ev=float(energy),
        magnetic_field_t=field_t,
        magnetic_field_kg=field_kg,
        calculated_current_a=(
            calculated_current
        ),
        command_current_a=(
            command_current
        ),
        current_clamped=(
            not math.isclose(
                calculated_current,
                command_current,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
    )


def beam_energy_for_magnet_current_ev(
    mass_u: float,
    current_a: float,
) -> float:
    """
    Calculate the beam energy selected by a fixed analyzing-magnet
    current for a given ion mass.

    This is the inverse relation needed when SIRIUS keeps the magnet
    fixed and changes extraction voltage instead.
    """

    mass = _validate_mass(
        mass_u
    )

    field_kg = (
        magnetic_field_kg_for_current(
            current_a
        )
    )

    if field_kg == 0:
        return 0.0

    field_t = (
        field_kg
        / 10.0
    )

    mass_kg = (
        mass
        * ATOMIC_MASS_UNIT_KG
    )

    energy_ev = (
        field_t ** 2
        * ELEMENTARY_CHARGE_C
        * MAGNET_RADIUS_M ** 2
        / (
            2.0
            * mass_kg
        )
    )

    return energy_ev


def extraction_voltage_for_fixed_magnet(
    mass_u: float,
    magnet_current_a: float,
    sputter_voltage_v: float,
) -> float:
    """
    Calculate the extraction-voltage magnitude corresponding to the
    central rigidity of a fixed magnet current.

        E_total = U_sputter + U_extraction

    therefore

        U_extraction = E_selected - U_sputter

    This does not enforce the SIRIUS extraction-voltage hard limits;
    the parameter/safety layer remains responsible for that.
    """

    sputter = float(
        sputter_voltage_v
    )

    if not math.isfinite(sputter):
        raise ValueError(
            "Sputter voltage must be finite"
        )

    if sputter < 0:
        raise ValueError(
            "Sputter voltage must be non-negative"
        )

    selected_energy = (
        beam_energy_for_magnet_current_ev(
            mass_u,
            magnet_current_a,
        )
    )

    extraction = (
        selected_energy
        - sputter
    )

    if extraction < 0:
        raise ValueError(
            "Fixed magnet corresponds to a total beam energy below "
            "the supplied sputter voltage"
        )

    return extraction