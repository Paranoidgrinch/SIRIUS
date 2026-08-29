from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.settling import SettlingPolicy
from sirius.state import (
    MachineState,
    utc_now_iso,
)
from sirius.transition import (
    AppliedStateResult,
    capture_readbacks,
)
from sirius.safe_transition import apply_state


@dataclass(frozen=True)
class SourceReferenceCheckResult:
    """
    Complete result of one periodic Cup-1 reference check.
    """

    working_state_before: MachineState

    reference_application: AppliedStateResult

    measurement: BeamMeasurement
    reference: SourceReference

    working_state_after: MachineState
    restoration: AppliedStateResult


class InvalidReferenceMeasurementError(RuntimeError):
    pass


class ReferenceMeasurementAndRestoreError(RuntimeError):
    """
    Raised when the Cup-1 measurement fails and restoring the previous
    working state fails as well.
    """

    def __init__(
        self,
        measurement_error: Exception,
        restore_error: Exception,
    ):
        self.measurement_error = measurement_error
        self.restore_error = restore_error

        super().__init__(
            "Cup-1 reference measurement failed and restoring the "
            "previous working state failed as well"
        )


def validate_reference_states(
    working_state: MachineState,
    reference_state: MachineState,
) -> None:
    working_state.validate()
    reference_state.validate()

    if working_state.mass_u != reference_state.mass_u:
        raise ValueError(
            "Working state and Cup-1 reference state must use the same ion mass"
        )

    if reference_state.cup != 1:
        raise ValueError(
            "Cup-1 reference state must explicitly select cup 1"
        )

    if working_state.cup is None:
        raise ValueError(
            "Working state must define its active cup so it can be restored"
        )


def perform_source_reference_check(
    adapter,
    working_state: MachineState,
    reference_state: MachineState,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    *,
    noise_floor_a: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], str] = utc_now_iso,
) -> SourceReferenceCheckResult:
    """
    Perform one complete Cup-1 source reference cycle.

    Sequence:

        1. Capture current working-state readbacks.
        2. Apply the saved optimized Cup-1 reference state.
        3. Insert/select Cup 1.
        4. Adaptively measure the Keithley beam current.
        5. Store the new source reference.
        6. Restore the original working-state commands and cup.
        7. Capture the restored physical readbacks.

    Parameter transitions compare command values only. A stable systematic
    command/readback offset therefore does not cause unnecessary commands.

    The analyzing magnet is only moved if its stored command value actually
    differs between the working and reference states.
    """

    validate_reference_states(
        working_state,
        reference_state,
    )

    # Capture the actual physical state immediately before leaving the
    # optimization point. Command values remain unchanged.
    working_snapshot = capture_readbacks(
        adapter,
        working_state,
    )

    # This transition may fail because of hardware. In that case apply_state
    # already surfaces the partial transition and we deliberately do not issue
    # an automatic blind rollback.
    reference_application = apply_state(
        adapter,
        current=working_snapshot,
        target=reference_state,
        settling_policies=settling_policies,
        select_target_cup=True,
    )

    reference_machine_state = (
        reference_application.observed_state
    )

    try:
        measurement = measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
            monotonic=monotonic,
        )

        if measurement.below_noise_floor:
            raise InvalidReferenceMeasurementError(
                "Cup-1 reference signal is below the configured noise floor"
            )

        if measurement.mean_a <= 0:
            raise InvalidReferenceMeasurementError(
                "Cup-1 reference current must be greater than zero"
            )

        measured_at = monotonic()

        new_reference = SourceReference(
            measurement=measurement,
            state_id=reference_state.state_id,
            mass_u=reference_state.mass_u,
            monotonic_s=measured_at,
            created_at_utc=utc_now(),
        )

        tracker.add(
            new_reference
        )

    except Exception as measurement_error:
        # The reference machine state was reached successfully, so a
        # controlled restoration of the known previous working state is
        # appropriate here.
        try:
            apply_state(
                adapter,
                current=reference_machine_state,
                target=working_snapshot,
                settling_policies=settling_policies,
                select_target_cup=True,
            )

        except Exception as restore_error:
            raise ReferenceMeasurementAndRestoreError(
                measurement_error,
                restore_error,
            ) from restore_error

        raise

    restoration = apply_state(
        adapter,
        current=reference_machine_state,
        target=working_snapshot,
        settling_policies=settling_policies,
        select_target_cup=True,
    )

    working_state_after = (
        restoration.observed_state
    )

    return SourceReferenceCheckResult(
        working_state_before=working_snapshot,
        reference_application=reference_application,
        measurement=measurement,
        reference=new_reference,
        working_state_after=working_state_after,
        restoration=restoration,
    )