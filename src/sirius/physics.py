from __future__ import annotations


def beam_energy_ev(
    sputter_voltage_v: float,
    extraction_voltage_v: float,
) -> float:
    """
    Kinetic energy of a singly charged negative ion leaving the source.

    FLAVIA exposes voltage magnitudes as positive values although the
    physical source potentials are negative. For a singly charged anion,
    the beam energy in eV is therefore the sum of the voltage magnitudes.
    """
    if sputter_voltage_v < 0:
        raise ValueError("Sputter voltage magnitude must be non-negative")

    if extraction_voltage_v < 0:
        raise ValueError("Extraction voltage magnitude must be non-negative")

    return sputter_voltage_v + extraction_voltage_v


def cooler_residual_energy_ev(
    sputter_voltage_v: float,
    extraction_voltage_v: float,
    ion_cooler_voltage_v: float,
) -> float:
    """
    Residual kinetic energy of the ion after electrostatic deceleration.

    Example:
        sputter = 8000 V
        extraction = 19600 V
        cooler = 27558 V

        beam energy = 27600 eV
        residual energy = 42 eV
    """
    if ion_cooler_voltage_v < 0:
        raise ValueError("Ion cooler voltage magnitude must be non-negative")

    return (
        beam_energy_ev(
            sputter_voltage_v,
            extraction_voltage_v,
        )
        - ion_cooler_voltage_v
    )


def ion_cooler_voltage_for_energy(
    sputter_voltage_v: float,
    extraction_voltage_v: float,
    residual_energy_ev: float,
) -> float:
    """
    Calculate the ion-cooler voltage magnitude required for a requested
    residual ion energy.
    """
    if residual_energy_ev < 0:
        raise ValueError("Residual energy must be non-negative")

    beam_energy = beam_energy_ev(
        sputter_voltage_v,
        extraction_voltage_v,
    )

    if residual_energy_ev > beam_energy:
        raise ValueError(
            "Residual energy cannot exceed the incoming beam energy"
        )

    return beam_energy - residual_energy_ev


def guidefield_command_difference_v(
    guidefield1_voltage_v: float,
    guidefield2_voltage_v: float,
) -> float:
    """
    Difference between the two FLAVIA guide-field command values.

    No physical polarity assumption is made here. The effective sign of
    the longitudinal guide field will be determined experimentally and
    can later be stored in the learned mass profile.
    """
    return guidefield1_voltage_v - guidefield2_voltage_v