from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

from sirius.parameters import PARAMETERS
from sirius.settling import (
    SettlingPolicy,
    SettlingResult,
    set_and_wait,
)
from sirius.state import MachineState


@dataclass(frozen=True)
class ParameterChange:
    name: str
    old_command: float | None
    new_command: float


@dataclass(frozen=True)
class StateTransitionPlan:
    source_state_id: str
    target_state_id: str
    changes: tuple[ParameterChange, ...]

    @property
    def is_noop(self) -> bool:
        return len(self.changes) == 0

    @property
    def changed_parameters(self) -> tuple[str, ...]:
        return tuple(
            change.name
            for change in self.changes
        )


@dataclass(frozen=True)
class AppliedStateResult:
    requested_state: MachineState
    observed_state: MachineState

    plan: StateTransitionPlan

    settling_results: tuple[SettlingResult, ...]

    selected_cup: int | None


class MissingSettlingPolicyError(KeyError):
    pass


class StateTransitionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        applied_parameters: tuple[str, ...],
        failed_parameter: str | None,
    ):
        self.applied_parameters = applied_parameters
        self.failed_parameter = failed_parameter

        super().__init__(message)


def _commands_equal(
    first: float,
    second: float,
) -> bool:
    """
    Compare stored command values.

    These are reproducibility values rather than noisy hardware readbacks,
    so only a tiny floating-point tolerance is required.
    """

    return math.isclose(
        float(first),
        float(second),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def plan_state_transition(
    current: MachineState,
    target: MachineState,
) -> StateTransitionPlan:
    """
    Determine which FLAVIA commands actually need to change.

    Physical readbacks are deliberately ignored here.

    Example:
        current command  = 19000 V
        current readback = 18600 V
        target command   = 19000 V

    Result:
        no command is sent.
    """

    current.validate()
    target.validate()

    if current.mass_u != target.mass_u:
        raise ValueError(
            "Automatic state transition between different ion masses "
            "is not allowed"
        )

    changes: list[ParameterChange] = []

    # PARAMETERS preserves our defined beamline-oriented ordering.
    for name in PARAMETERS:
        if name not in target.parameters:
            continue

        new_value = float(
            target.parameters[name]
        )

        old_value = current.parameters.get(
            name
        )

        if (
            old_value is None
            or not _commands_equal(
                old_value,
                new_value,
            )
        ):
            changes.append(
                ParameterChange(
                    name=name,
                    old_command=(
                        None
                        if old_value is None
                        else float(old_value)
                    ),
                    new_command=new_value,
                )
            )

    return StateTransitionPlan(
        source_state_id=current.state_id,
        target_state_id=target.state_id,
        changes=tuple(changes),
    )


def capture_readbacks(
    adapter,
    state: MachineState,
) -> MachineState:
    """
    Capture currently available physical readbacks for every parameter
    represented in a MachineState.

    Missing readbacks are skipped rather than guessed.
    """

    readbacks = dict(
        state.readbacks
    )

    for name in state.parameters:
        value = adapter.read_parameter(
            name
        )

        if value is not None:
            readbacks[name] = float(
                value
            )

    observed = replace(
        state,
        readbacks=readbacks,
    )

    observed.validate()

    return observed


def apply_state(
    adapter,
    current: MachineState,
    target: MachineState,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    *,
    select_target_cup: bool = True,
) -> AppliedStateResult:
    """
    Apply only command differences between two known SIRIUS states.

    Each changed parameter requires an explicit settling policy.

    No automatic rollback is attempted after a failed hardware transition.
    A partially applied state is surfaced to the caller so that SIRIUS
    does not issue additional blind commands after a hardware problem.
    """

    plan = plan_state_transition(
        current,
        target,
    )

    settling_results: list[
        SettlingResult
    ] = []

    applied_parameters: list[str] = []

    for change in plan.changes:
        policy = settling_policies.get(
            change.name
        )

        if policy is None:
            raise MissingSettlingPolicyError(
                f"No settling policy configured for {change.name}"
            )

        try:
            result = set_and_wait(
                adapter,
                change.name,
                change.new_command,
                policy,
            )

        except Exception as exc:
            raise StateTransitionError(
                (
                    "State transition failed while applying "
                    f"{change.name}: {exc}"
                ),
                applied_parameters=tuple(
                    applied_parameters
                ),
                failed_parameter=change.name,
            ) from exc

        settling_results.append(
            result
        )

        applied_parameters.append(
            change.name
        )

    selected_cup = None

    if (
        select_target_cup
        and target.cup is not None
    ):
        adapter.select_cup(
            target.cup
        )

        selected_cup = target.cup

    observed = capture_readbacks(
        adapter,
        target,
    )

    return AppliedStateResult(
        requested_state=target,
        observed_state=observed,
        plan=plan,
        settling_results=tuple(
            settling_results
        ),
        selected_cup=selected_cup,
    )