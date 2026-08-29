from __future__ import annotations

from typing import Mapping

from sirius.cup_ack import (
    CupSelectionPolicy,
)
from sirius.hardware_guard import (
    HardwareGuardPolicy,
    build_strict_hardware_guard,
    require_complete_hardware_guard,
)
from sirius.hardware_safety import (
    HardwareSafetyConfig,
)


COLOGNEAMS_COMMISSIONING_PROFILE_NAME = (
    "cologneams_commissioning_v1"
)

COLOGNEAMS_COMMISSIONING_PROFILE_STATUS = (
    "provisional"
)


# ---------------------------------------------------------------------------
# Initial conservative command-space limits for machine commissioning.
#
# IMPORTANT:
#
# These are SIRIUS software transition limits, NOT manufacturer ratings.
#
# They intentionally start conservatively and are expected to be reviewed
# channel-by-channel after controlled real-machine validation.
# ---------------------------------------------------------------------------

COLOGNEAMS_COMMISSIONING_MAX_STEPS: dict[
    str,
    float,
] = {
    # --------------------------------------------------------
    # Ion source / low-energy injection
    # --------------------------------------------------------

    "sputter_voltage_v":
        100.0,

    "extraction_voltage_v":
        100.0,

    "einzel_lens_voltage_v":
        100.0,

    "magnet_current_a":
        0.25,

    "lens2_voltage_v":
        50.0,

    "steerer_x1_v":
        5.0,

    "steerer_y1_v":
        5.0,

    # --------------------------------------------------------
    # Ion cooler
    # --------------------------------------------------------

    "ion_cooler_voltage_v":
        100.0,

    # HV1 / HV4:
    # deliberately smaller than the main cooler-voltage step because
    # they are optimized as coupled entrance/exit coordinates.
    "deceleration_voltage_v":
        50.0,

    "acceleration_voltage_v":
        50.0,

    "guidefield1_voltage_v":
        0.5,

    "guidefield2_voltage_v":
        0.5,

    # --------------------------------------------------------
    # QPT transport
    # --------------------------------------------------------

    "quadrupole1_voltage_v":
        50.0,

    "quadrupole2_voltage_v":
        50.0,

    "quadrupole3_voltage_v":
        50.0,

    "steerer_x2_v":
        5.0,

    "steerer_y2_v":
        5.0,

    # --------------------------------------------------------
    # ESA / downstream transport
    # --------------------------------------------------------

    "esa_voltage_v":
        25.0,

    "steerer_x3_v":
        5.0,

    "steerer_y3_v":
        5.0,

    "lens4_voltage_v":
        50.0,
}


COOLER_END_COMMISSIONING_MAX_STEP_V = (
    COLOGNEAMS_COMMISSIONING_MAX_STEPS[
        "deceleration_voltage_v"
    ]
)

QPT_COMMISSIONING_MAX_STEP_V = (
    COLOGNEAMS_COMMISSIONING_MAX_STEPS[
        "quadrupole1_voltage_v"
    ]
)


def cologneams_commissioning_max_steps(
) -> dict[
    str,
    float,
]:
    """
    Return a mutable copy of the initial CologneAMS commissioning limits.

    Callers can deliberately modify the copy for a specific commissioning
    run without modifying the canonical profile constants.
    """

    return dict(
        COLOGNEAMS_COMMISSIONING_MAX_STEPS
    )


def build_cologneams_commissioning_guard(
    *,
    overrides: Mapping[
        str,
        float,
    ] | None = None,
    max_total_steps: int = 10000,
) -> HardwareGuardPolicy:
    """
    Build the complete initial CologneAMS guard.

    overrides:
        Explicit per-run changes to one or more commissioning limits.

        This is deliberately explicit so experimental changes remain
        visible in the run configuration / manifest rather than being
        silently learned by SIRIUS.
    """

    steps = (
        cologneams_commissioning_max_steps()
    )

    if overrides is not None:
        for (
            parameter_name,
            max_step,
        ) in overrides.items():
            if (
                parameter_name
                not in steps
            ):
                raise ValueError(
                    "Cannot override unknown or non-required "
                    "CologneAMS commissioning parameter "
                    f"{parameter_name}"
                )

            steps[
                parameter_name
            ] = float(
                max_step
            )

    guard = (
        build_strict_hardware_guard(
            steps,
            max_total_steps=(
                max_total_steps
            ),
        )
    )

    require_complete_hardware_guard(
        guard
    )

    return guard


def build_cologneams_hardware_safety(
    *,
    cup_selection_policy: (
        CupSelectionPolicy | None
    ) = None,
    max_step_overrides: Mapping[
        str,
        float,
    ] | None = None,
    max_total_steps: int = 10000,
) -> HardwareSafetyConfig:
    """
    Construct the complete initial CologneAMS SIRIUS safety configuration.

    One factory therefore supplies consistently:

        - general hardware guard
        - HV1/HV4 coupled-transition step limit
        - QPT coupled-transition step limit
        - positive Faraday-cup acknowledgement policy

    HV/QPT values are derived from the same central map used by the
    general guard, preventing two different limits from silently existing
    for the same physical channel.
    """

    steps = (
        cologneams_commissioning_max_steps()
    )

    if max_step_overrides is not None:
        for (
            parameter_name,
            max_step,
        ) in max_step_overrides.items():
            if (
                parameter_name
                not in steps
            ):
                raise ValueError(
                    "Cannot override unknown or non-required "
                    "CologneAMS commissioning parameter "
                    f"{parameter_name}"
                )

            steps[
                parameter_name
            ] = float(
                max_step
            )

    guard = (
        build_strict_hardware_guard(
            steps,
            max_total_steps=(
                max_total_steps
            ),
        )
    )

    cup_policy = (
        cup_selection_policy
        if cup_selection_policy
        is not None
        else CupSelectionPolicy()
    )

    cooler_end_step = min(
        float(
            steps[
                "deceleration_voltage_v"
            ]
        ),
        float(
            steps[
                "acceleration_voltage_v"
            ]
        ),
    )

    qpt_step = min(
        float(
            steps[
                "quadrupole1_voltage_v"
            ]
        ),
        float(
            steps[
                "quadrupole2_voltage_v"
            ]
        ),
        float(
            steps[
                "quadrupole3_voltage_v"
            ]
        ),
    )

    return HardwareSafetyConfig(
        cooler_end_max_step_v=(
            cooler_end_step
        ),
        qpt_max_step_v=(
            qpt_step
        ),
        cup_selection_policy=(
            cup_policy
        ),
        hardware_guard_policy=(
            guard
        ),
    )


def cologneams_safety_manifest(
    safety: HardwareSafetyConfig,
) -> dict:
    """
    Add machine/profile provenance around the normal HardwareSafetyConfig
    manifest representation.
    """

    require_complete_hardware_guard(
        safety.hardware_guard_policy
    )

    return {
        "machine":
            "CologneAMS",

        "profile_name":
            COLOGNEAMS_COMMISSIONING_PROFILE_NAME,

        "profile_status":
            COLOGNEAMS_COMMISSIONING_PROFILE_STATUS,

        "hardware_safety":
            safety.to_manifest_dict(),
    }