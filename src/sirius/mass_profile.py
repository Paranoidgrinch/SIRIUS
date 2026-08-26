from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sirius.parameters import PARAMETERS


MASS_PROFILE_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_finite(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    return value


@dataclass
class LearnedRange:
    """
    Experimentally useful search interval for one SIRIUS parameter.

    This interval is always subordinate to the hard bounds defined in
    sirius.parameters.
    """

    minimum: float
    maximum: float

    evidence_points: int = 0
    source: str = "learned"

    updated_at_utc: str = field(
        default_factory=utc_now_iso
    )

    def validate(
        self,
        parameter_name: str,
    ) -> None:
        if parameter_name not in PARAMETERS:
            raise ValueError(
                f"Unknown SIRIUS parameter: {parameter_name}"
            )

        minimum = _validate_finite(
            "minimum",
            self.minimum,
        )

        maximum = _validate_finite(
            "maximum",
            self.maximum,
        )

        if minimum > maximum:
            raise ValueError(
                "Learned range minimum must not exceed maximum"
            )

        if self.evidence_points < 0:
            raise ValueError(
                "evidence_points must be non-negative"
            )

        hard = PARAMETERS[
            parameter_name
        ]

        if minimum < hard.minimum:
            raise ValueError(
                f"Learned minimum for {parameter_name} "
                f"is below hard minimum {hard.minimum}"
            )

        if maximum > hard.maximum:
            raise ValueError(
                f"Learned maximum for {parameter_name} "
                f"is above hard maximum {hard.maximum}"
            )


@dataclass
class MassProfile:
    """
    Persistent SIRIUS knowledge for one ion mass.

    Profiles intentionally key only on ion mass at this stage, matching
    the current SIRIUS design requirement.
    """

    mass_u: float

    learned_ranges: dict[
        str,
        LearnedRange,
    ] = field(
        default_factory=dict
    )

    best_commands: dict[
        str,
        float,
    ] = field(
        default_factory=dict
    )

    best_state_ids: dict[
        str,
        str,
    ] = field(
        default_factory=dict
    )

    guidefield_forward_sign: int | None = None

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    created_at_utc: str = field(
        default_factory=utc_now_iso
    )

    updated_at_utc: str = field(
        default_factory=utc_now_iso
    )

    schema_version: int = MASS_PROFILE_SCHEMA_VERSION

    def validate(self) -> None:
        if not math.isfinite(
            float(self.mass_u)
        ):
            raise ValueError(
                "Ion mass must be finite"
            )

        if self.mass_u <= 0:
            raise ValueError(
                "Ion mass must be greater than zero"
            )

        for (
            parameter_name,
            learned_range,
        ) in self.learned_ranges.items():
            learned_range.validate(
                parameter_name
            )

        for (
            parameter_name,
            value,
        ) in self.best_commands.items():
            if parameter_name not in PARAMETERS:
                raise ValueError(
                    f"Unknown SIRIUS parameter: {parameter_name}"
                )

            value = _validate_finite(
                parameter_name,
                value,
            )

            definition = PARAMETERS[
                parameter_name
            ]

            if not (
                definition.minimum
                <= value
                <= definition.maximum
            ):
                raise ValueError(
                    f"Best command {parameter_name}={value} "
                    f"is outside hard bounds "
                    f"{definition.minimum}..{definition.maximum}"
                )

        for (
            label,
            state_id,
        ) in self.best_state_ids.items():
            if not label:
                raise ValueError(
                    "Best-state label must not be empty"
                )

            if not state_id:
                raise ValueError(
                    "Best-state ID must not be empty"
                )

        if (
            self.guidefield_forward_sign
            not in (
                None,
                -1,
                1,
            )
        ):
            raise ValueError(
                "guidefield_forward_sign must be -1, +1 or None"
            )

    def touch(self) -> None:
        self.updated_at_utc = (
            utc_now_iso()
        )

    def effective_bounds(
        self,
        parameter_name: str,
    ) -> tuple[float, float]:
        """
        Return the search bounds SIRIUS should currently use.

        If no learned range exists, the hard SIRIUS bounds are returned.
        """

        if parameter_name not in PARAMETERS:
            raise KeyError(
                f"Unknown SIRIUS parameter: {parameter_name}"
            )

        learned = self.learned_ranges.get(
            parameter_name
        )

        if learned is not None:
            learned.validate(
                parameter_name
            )

            return (
                float(learned.minimum),
                float(learned.maximum),
            )

        hard = PARAMETERS[
            parameter_name
        ]

        return (
            float(hard.minimum),
            float(hard.maximum),
        )

    def set_learned_range(
        self,
        parameter_name: str,
        minimum: float,
        maximum: float,
        *,
        evidence_points: int = 0,
        source: str = "learned",
    ) -> None:
        learned = LearnedRange(
            minimum=float(minimum),
            maximum=float(maximum),
            evidence_points=int(
                evidence_points
            ),
            source=str(source),
        )

        learned.validate(
            parameter_name
        )

        self.learned_ranges[
            parameter_name
        ] = learned

        self.touch()

    def clear_learned_range(
        self,
        parameter_name: str,
    ) -> None:
        self.learned_ranges.pop(
            parameter_name,
            None,
        )

        self.touch()

    def set_best_command(
        self,
        parameter_name: str,
        value: float,
    ) -> None:
        if parameter_name not in PARAMETERS:
            raise KeyError(
                f"Unknown SIRIUS parameter: {parameter_name}"
            )

        value = _validate_finite(
            parameter_name,
            value,
        )

        definition = PARAMETERS[
            parameter_name
        ]

        if not (
            definition.minimum
            <= value
            <= definition.maximum
        ):
            raise ValueError(
                f"{parameter_name}={value} outside hard bounds "
                f"{definition.minimum}..{definition.maximum}"
            )

        self.best_commands[
            parameter_name
        ] = value

        self.touch()

    def set_best_state(
        self,
        label: str,
        state_id: str,
    ) -> None:
        if not label:
            raise ValueError(
                "Best-state label must not be empty"
            )

        if not state_id:
            raise ValueError(
                "Best-state ID must not be empty"
            )

        self.best_state_ids[
            str(label)
        ] = str(state_id)

        self.touch()

    def set_guidefield_forward_sign(
        self,
        sign: int | None,
    ) -> None:
        if sign not in (
            None,
            -1,
            1,
        ):
            raise ValueError(
                "Guidefield sign must be -1, +1 or None"
            )

        self.guidefield_forward_sign = sign

        self.touch()

    def to_dict(self) -> dict[str, Any]:
        self.validate()

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "MassProfile":
        data = dict(data)

        ranges_data = data.pop(
            "learned_ranges",
            {},
        )

        learned_ranges = {
            name: LearnedRange(
                **range_data
            )
            for (
                name,
                range_data,
            ) in ranges_data.items()
        }

        profile = cls(
            learned_ranges=learned_ranges,
            **data,
        )

        profile.validate()

        return profile


def mass_filename(
    mass_u: float,
) -> str:
    """
    Produce a deterministic filesystem-safe mass-profile filename.

    Examples:
        60.0  -> mass_60.json
        60.5  -> mass_60p5.json
    """

    if not math.isfinite(
        float(mass_u)
    ):
        raise ValueError(
            "Ion mass must be finite"
        )

    if mass_u <= 0:
        raise ValueError(
            "Ion mass must be greater than zero"
        )

    mass_text = (
        f"{float(mass_u):g}"
        .replace(".", "p")
    )

    return (
        f"mass_{mass_text}.json"
    )


class MassProfileStore:
    """
    Persistent storage for learned SIRIUS mass profiles.

    Writes are performed through a temporary file followed by replacement
    so an interrupted write is less likely to corrupt the existing profile.
    """

    def __init__(
        self,
        directory: str | Path,
    ):
        self.directory = Path(
            directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def path_for_mass(
        self,
        mass_u: float,
    ) -> Path:
        return (
            self.directory
            / mass_filename(
                mass_u
            )
        )

    def exists(
        self,
        mass_u: float,
    ) -> bool:
        return self.path_for_mass(
            mass_u
        ).exists()

    def save(
        self,
        profile: MassProfile,
    ) -> Path:
        profile.validate()

        path = self.path_for_mass(
            profile.mass_u
        )

        temporary = path.with_suffix(
            ".json.tmp"
        )

        payload = profile.to_dict()

        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )

            handle.write("\n")

        temporary.replace(
            path
        )

        return path

    def load(
        self,
        mass_u: float,
    ) -> MassProfile:
        path = self.path_for_mass(
            mass_u
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(
                handle
            )

        profile = MassProfile.from_dict(
            data
        )

        if not math.isclose(
            profile.mass_u,
            float(mass_u),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Mass profile file contains a different ion mass"
            )

        return profile

    def load_or_create(
        self,
        mass_u: float,
    ) -> MassProfile:
        if self.exists(
            mass_u
        ):
            return self.load(
                mass_u
            )

        return MassProfile(
            mass_u=float(mass_u)
        )