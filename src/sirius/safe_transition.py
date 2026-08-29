from __future__ import annotations

import math
from copy import deepcopy
from typing import Mapping

from sirius.cup_ack import CupSelectionPolicy
from sirius.command_cadence import (
    CommandCadenceController,
)
from sirius.hardware_guard import (
    HardwareGuardPolicy,
    HardwareSafetyViolation,
    ParameterSafetyRule,
    execute_guarded_transition,
    plan_guarded_transition,
)
from sirius.readback_freshness import (
    ReadbackFreshnessPolicy,
    wait_for_fresh_parameter_readback,
)
from sirius.settling import SettlingPolicy
from sirius.state import MachineState
from sirius.transition import (
    apply_state as _raw_apply_state,
)


def _command_cadence_controller(
    adapter,
) -> CommandCadenceController:
    controller = getattr(
        adapter,
        "command_cadence_controller",
        None,
    )

    if controller is None:
        controller = (
            CommandCadenceController()
        )

        setattr(
            adapter,
            "command_cadence_controller",
            controller,
        )

    if not isinstance(
        controller,
        CommandCadenceController,
    ):
        raise HardwareSafetyViolation(
            "adapter.command_cadence_controller must be "
            "CommandCadenceController"
        )

    return controller


def _commands_changed(
    current: MachineState,
    target: MachineState,
) -> bool:
    if (
        set(current.parameters)
        != set(target.parameters)
    ):
        return True

    return any(
        not math.isclose(
            float(current.parameters[name]),
            float(target.parameters[name]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for name
        in current.parameters
    )


def _parameter_only_target(
    current: MachineState,
    target: MachineState,
) -> MachineState:
    """
    Build the logical command target while retaining the currently
    selected Faraday cup.

    Parameter transitions are completed before a requested cup change.
    This keeps the two independently acknowledged hardware actions
    serialized.
    """

    state = MachineState(
        mass_u=target.mass_u,
        parameters=dict(
            target.parameters
        ),
        readbacks=dict(
            current.readbacks
        ),
        cup=current.cup,
        stage=current.stage,
        role=current.role,
        rfq=deepcopy(
            current.rfq
        ),
        fixed_conditions=deepcopy(
            current.fixed_conditions
        ),
        metadata=deepcopy(
            current.metadata
        ),
    )

    state.validate()

    return state


def _require_guard_step_handshake(
    parameter_name: str,
    rule: ParameterSafetyRule,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> SettlingPolicy:
    """
    Real guarded commands currently require the strongest handshake:

        command
        -> fresh readback
        -> stable readback
        -> next command

    We deliberately do not implement a weaker fire-and-forget path.
    """

    if not rule.require_readback:
        raise HardwareSafetyViolation(
            f"{parameter_name}: guarded hardware execution requires "
            "readback confirmation"
        )

    if not rule.require_settling:
        raise HardwareSafetyViolation(
            f"{parameter_name}: guarded hardware execution requires "
            "readback settling before the next command"
        )

    try:
        return settling_policies[
            parameter_name
        ]
    except KeyError as exc:
        raise HardwareSafetyViolation(
            "No settling policy configured for guarded parameter "
            f"{parameter_name}"
        ) from exc


def _state_with_verified_readback(
    state: MachineState,
    parameter_name: str,
    readback_value: float,
) -> MachineState:
    readbacks = dict(
        state.readbacks
    )

    readbacks[
        parameter_name
    ] = float(
        readback_value
    )

    verified = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            state.parameters
        ),
        readbacks=readbacks,
        cup=state.cup,
        stage=state.stage,
        role=state.role,
        rfq=deepcopy(
            state.rfq
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata=deepcopy(
            state.metadata
        ),
    )

    verified.validate()

    return verified


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
    cup_selection_policy: (
        CupSelectionPolicy | None
    ) = None,
    hardware_guard_policy: (
        HardwareGuardPolicy | None
    ) = None,
):
    """
    Safety boundary for SIRIUS MachineState transitions.

    Application code must use this function rather than calling
    sirius.transition.apply_state() directly.

    With a HardwareGuardPolicy:
        - logical command jumps are split into bounded microsteps
        - exactly one physical parameter is changed per guard step
        - every step requires fresh/stable readback confirmation
        - the next command is not issued until the previous step returns
        - parameter changes finish before a requested Faraday-cup change

    Without a guard:
        preserve the historical direct transition behaviour. This exists
        for isolated tests/offline code only; SiriusRuntime requires a
        guard before real execution.
    """

    current.validate()
    target.validate()

    effective_guard = (
        hardware_guard_policy
    )

    if effective_guard is None:
        effective_guard = getattr(
            adapter,
            "hardware_guard_policy",
            None,
        )

    if effective_guard is None:
        return _raw_apply_state(
            adapter,
            current=current,
            target=target,
            settling_policies=(
                settling_policies
            ),
            select_target_cup=(
                select_target_cup
            ),
            cup_selection_policy=(
                cup_selection_policy
            ),
        )

    if not isinstance(
        effective_guard,
        HardwareGuardPolicy,
    ):
        raise TypeError(
            "hardware_guard_policy must be HardwareGuardPolicy"
        )

    physical_state = (
        current
    )

    # ============================================================
    # Phase 1:
    # Parameter commands. No cup selection may happen inside this phase.
    # ============================================================

    if _commands_changed(
        physical_state,
        target,
    ):
        parameter_target = (
            _parameter_only_target(
                physical_state,
                target,
            )
        )

        plan = plan_guarded_transition(
            physical_state,
            parameter_target,
            effective_guard,
        )

        executor_state = (
            physical_state
        )

        def execute_step(
            step,
            rule,
        ):
            nonlocal executor_state

            settling_policy = (
                _require_guard_step_handshake(
                    step.parameter_name,
                    rule,
                    settling_policies,
                )
            )

            freshness_policy = getattr(
                adapter,
                "readback_freshness_policy",
                None,
            )

            if not isinstance(
                freshness_policy,
                ReadbackFreshnessPolicy,
            ):
                raise HardwareSafetyViolation(
                    "Guarded hardware execution requires an explicit "
                    "ReadbackFreshnessPolicy"
                )

            capture_barrier = getattr(
                adapter,
                "capture_parameter_readback_freshness_barrier",
                None,
            )

            if not callable(
                capture_barrier
            ):
                raise HardwareSafetyViolation(
                    "Adapter cannot capture parameter readback "
                    "freshness barriers"
                )

            cadence_controller = (
                _command_cadence_controller(
                    adapter
                )
            )

            # The reservation waits until this physical channel may
            # receive another command. It is deliberately performed
            # before capturing the freshness barrier.
            cadence_controller.reserve(
                step.parameter_name,
                rule.minimum_command_interval_s,
            )

            # IMPORTANT:
            # capture immediately BEFORE the physical command.
            barrier = capture_barrier(
                step.parameter_name
            )

            transition = _raw_apply_state(
                adapter,
                current=executor_state,
                target=step.target_state,
                settling_policies={
                    step.parameter_name:
                        settling_policy,
                },
                select_target_cup=False,
                cup_selection_policy=(
                    cup_selection_policy
                ),
            )

            # Even if the lower settling layer returned, the guarded step
            # is NOT complete until FLAVIA has produced at least one
            # timestamped readback strictly newer than the pre-command
            # barrier.
            fresh = (
                wait_for_fresh_parameter_readback(
                    adapter,
                    step.parameter_name,
                    not_before_source_timestamp=(
                        barrier
                    ),
                    policy=(
                        freshness_policy
                    ),
                    quality_policy=getattr(
                        adapter,
                        "readback_quality_policy",
                        None,
                    ),
                )
            )

            executor_state = (
                _state_with_verified_readback(
                    transition.observed_state,
                    step.parameter_name,
                    fresh.value,
                )
            )

            return executor_state

        guarded_result = (
            execute_guarded_transition(
                plan,
                effective_guard,
                execute_step,
            )
        )

        physical_state = (
            guarded_result.final_state
        )

    # ============================================================
    # Phase 2:
    # Cup acknowledgement and exact target-state semantics.
    #
    # At this point no parameter command differs from target anymore.
    # ============================================================

    return _raw_apply_state(
        adapter,
        current=physical_state,
        target=target,
        settling_policies=(
            settling_policies
        ),
        select_target_cup=(
            select_target_cup
        ),
        cup_selection_policy=(
            cup_selection_policy
        ),
    )