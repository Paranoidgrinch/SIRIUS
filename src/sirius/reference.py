from __future__ import annotations

import math
from dataclasses import dataclass, field

from sirius.measurement import BeamMeasurement


DEFAULT_REFERENCE_INTERVAL_S = 600.0


@dataclass(frozen=True)
class SourceReference:
    """
    One Cup-1 source-current reference measurement.

    monotonic_s is used for robust internal timing during a SIRIUS run.
    created_at_utc is retained for logging and reproducibility.
    """

    measurement: BeamMeasurement
    state_id: str
    mass_u: float

    monotonic_s: float
    created_at_utc: str


@dataclass(frozen=True)
class TransmissionResult:
    """
    Beam transmission relative to the Cup-1 reference.
    """

    cup: int

    current_a: float
    current_sem_a: float

    reference_current_a: float
    reference_sem_a: float

    transmission: float
    transmission_percent: float
    transmission_sem: float
    transmission_sem_percent: float

    reference_state_id: str


@dataclass
class SourceReferenceTracker:
    """
    Tracks Cup-1 source references during one SIRIUS run.

    The default interval is 10 minutes.
    """

    interval_s: float = DEFAULT_REFERENCE_INTERVAL_S
    references: list[SourceReference] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ValueError(
                "Reference interval must be greater than zero"
            )

    @property
    def latest(self) -> SourceReference | None:
        if not self.references:
            return None

        return self.references[-1]

    def add(self, reference: SourceReference) -> None:
        if reference.mass_u <= 0:
            raise ValueError(
                "Reference mass must be greater than zero"
            )

        if reference.measurement.mean_a <= 0:
            raise ValueError(
                "Cup-1 reference current must be greater than zero"
            )

        if self.references:
            previous = self.references[-1]

            if reference.monotonic_s < previous.monotonic_s:
                raise ValueError(
                    "Reference timestamps must be monotonic"
                )

            if reference.mass_u != previous.mass_u:
                raise ValueError(
                    "Cannot mix ion masses in one reference tracker"
                )

        self.references.append(reference)

    def is_due(self, now_monotonic_s: float) -> bool:
        """
        Return True if a new Cup-1 source check is required.

        Before the first reference, a check is always due.
        """

        latest = self.latest

        if latest is None:
            return True

        elapsed = now_monotonic_s - latest.monotonic_s

        return elapsed >= self.interval_s

    def seconds_until_due(
        self,
        now_monotonic_s: float,
    ) -> float:
        latest = self.latest

        if latest is None:
            return 0.0

        remaining = (
            self.interval_s
            - (now_monotonic_s - latest.monotonic_s)
        )

        return max(0.0, remaining)

    def relative_source_drift(
        self,
    ) -> float | None:
        """
        Relative change of the latest Cup-1 current compared with the
        first Cup-1 reference of the run.

        Example:
            -0.10 means the source is 10 % lower than at run start.
        """

        if len(self.references) < 2:
            return None

        first = self.references[0].measurement.mean_a
        latest = self.references[-1].measurement.mean_a

        if first == 0:
            return None

        return (latest - first) / abs(first)


def transmission_from_reference(
    cup: int,
    measurement: BeamMeasurement,
    reference: SourceReference,
) -> TransmissionResult:
    """
    Calculate transmission relative to the current Cup-1 source reference.

        T = I_cup / I_cup1

    Uncertainty is propagated from the SEM of both measurements.
    """

    if not 1 <= cup <= 6:
        raise ValueError(
            "Cup must be between 1 and 6"
        )

    current = float(measurement.mean_a)
    current_sem = float(measurement.sem_a)

    reference_current = float(
        reference.measurement.mean_a
    )

    reference_sem = float(
        reference.measurement.sem_a
    )

    if reference_current <= 0:
        raise ValueError(
            "Reference current must be greater than zero"
        )

    transmission = current / reference_current

    if current > 0:
        relative_variance = (
            (current_sem / current) ** 2
            + (
                reference_sem
                / reference_current
            ) ** 2
        )

        transmission_sem = (
            abs(transmission)
            * math.sqrt(relative_variance)
        )

    else:
        transmission_sem = (
            current_sem / reference_current
        )

    return TransmissionResult(
        cup=cup,
        current_a=current,
        current_sem_a=current_sem,
        reference_current_a=reference_current,
        reference_sem_a=reference_sem,
        transmission=transmission,
        transmission_percent=100.0 * transmission,
        transmission_sem=transmission_sem,
        transmission_sem_percent=(
            100.0 * transmission_sem
        ),
        reference_state_id=reference.state_id,
    )