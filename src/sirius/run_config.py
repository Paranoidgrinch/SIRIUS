from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sirius.hardware_safety import (
    HardwareSafetyConfig,
)


@dataclass(frozen=True)
class SiriusRunConfig:
    """
    Run-wide SIRIUS configuration.

    HardwareSafetyConfig is mandatory. A real automated run therefore
    cannot silently omit the facility-specific coupled-transition limits.

    This object deliberately contains only settings that belong to the
    complete run rather than to one optimizer stage.
    """

    mass_u: float

    hardware_safety: HardwareSafetyConfig

    perform_final_characterization: bool = True

    metadata: dict[
        str,
        Any,
    ] | None = None

    def __post_init__(self) -> None:
        mass = float(
            self.mass_u
        )

        if (
            not math.isfinite(
                mass
            )
            or mass <= 0
        ):
            raise ValueError(
                "mass_u must be finite and greater than zero"
            )

        if not isinstance(
            self.hardware_safety,
            HardwareSafetyConfig,
        ):
            raise TypeError(
                "hardware_safety must be a HardwareSafetyConfig"
            )

        if (
            self.metadata is not None
            and not isinstance(
                self.metadata,
                dict,
            )
        ):
            raise TypeError(
                "metadata must be a dict or None"
            )

    def to_manifest_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "mass_u": float(
                self.mass_u
            ),
            "perform_final_characterization": bool(
                self.perform_final_characterization
            ),
            "hardware_safety": (
                self.hardware_safety.to_manifest_dict()
            ),
            "metadata": dict(
                self.metadata or {}
            ),
        }