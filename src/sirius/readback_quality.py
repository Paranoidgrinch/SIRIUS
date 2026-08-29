from __future__ import annotations

from dataclasses import dataclass


class ReadbackQualityError(
    RuntimeError
):
    pass


class MissingReadbackQualityError(
    ReadbackQualityError
):
    pass


class RejectedReadbackQualityError(
    ReadbackQualityError
):
    pass


@dataclass(frozen=True)
class ReadbackQualityPolicy:
    """
    Explicit allow-list for FLAVIA readback quality values.

    SIRIUS deliberately does not invent facility-specific meanings for
    values such as GOOD, VALID, OK, CONNECTED, etc.

    A real-machine configuration must explicitly provide the quality
    values that FLAVIA documents / exposes as acceptable.

    Any non-listed value fails closed.
    """

    accepted_values: tuple[
        str,
        ...
    ]

    allow_missing: bool = False

    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.accepted_values:
            raise ValueError(
                "accepted_values must contain at least one explicit "
                "readback quality value"
            )

        normalized = tuple(
            self.normalize(
                value
            )
            for value
            in self.accepted_values
        )

        if any(
            not value
            for value
            in normalized
        ):
            raise ValueError(
                "accepted readback quality values must not be empty"
            )

        if len(
            normalized
        ) != len(
            set(
                normalized
            )
        ):
            raise ValueError(
                "accepted readback quality values must be unique"
            )

    @property
    def strict(
        self,
    ) -> bool:
        return not self.allow_missing

    def normalize(
        self,
        value,
    ) -> str:
        text = str(
            value
        ).strip()

        if not self.case_sensitive:
            text = text.casefold()

        return text

    def accepts(
        self,
        quality,
    ) -> bool:
        if quality is None:
            return bool(
                self.allow_missing
            )

        normalized = self.normalize(
            quality
        )

        if not normalized:
            return bool(
                self.allow_missing
            )

        accepted = {
            self.normalize(
                value
            )
            for value
            in self.accepted_values
        }

        return (
            normalized
            in accepted
        )

    def require_accepted(
        self,
        quality,
        *,
        parameter_name: str,
    ) -> str | None:
        if quality is None:
            if self.allow_missing:
                return None

            raise MissingReadbackQualityError(
                f"{parameter_name}: FLAVIA readback quality is missing"
            )

        normalized = self.normalize(
            quality
        )

        if not normalized:
            if self.allow_missing:
                return None

            raise MissingReadbackQualityError(
                f"{parameter_name}: FLAVIA readback quality is empty"
            )

        if not self.accepts(
            quality
        ):
            raise RejectedReadbackQualityError(
                f"{parameter_name}: FLAVIA readback quality "
                f"{quality!r} is not in the explicit accepted set"
            )

        return normalized