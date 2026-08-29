from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    Callable,
    Mapping,
)

from sirius.parameters import PARAMETERS
from sirius.state import MachineState


class HardwareSafetyViolation(
    RuntimeError
):
    """
    A requested transition violates a deterministic hardware-safety rule.

    This is a configuration / command-planning failure, not an optimizer
    decision and not a measurement failure.
    """


class HardwareTransitionFailure(
    RuntimeError
):
    """
    A previously valid transition plan could not be completed.

    completed_steps:
        number of positively completed hardware steps

    last_state:
        last positively confirmed machine state

    No automatic rollback is implied.
    """

    def __init__(
        self,
        message: str,
        *,
        last_state: MachineState,
        completed_steps: int,
    ):
        super().__init__(
            message
        )

        self.last_state = (
            last_state
        )

        self.completed_steps = int(
            completed_steps
        )


@dataclass(frozen=True)
class ParameterSafetyRule:
    """
    Deterministic transition constraints for one SIRIUS parameter.

    max_step:
        Maximum allowed command-space change between two positively
        completed hardware states.

    require_readback:
        A fresh device readback must be obtained after the command.

    require_settling:
        The readback must satisfy the configured settling criterion before
        another hardware command may be issued.

    No command/readback equality is implied.
    """

    max_step: float

    require_readback: bool = True
    require_settling: bool = True

    def __post_init__(self) -> None:
        value = float(
            self.max_step
        )

        if (
            not math.isfinite(
                value
            )
            or value <= 0
        ):
            raise ValueError(
                "max_step must be finite and greater than zero"
            )

        if (
            self.require_settling
            and not self.require_readback
        ):
            raise ValueError(
                "Settling cannot be required without readback"
            )


@dataclass(frozen=True)
class HardwareGuardPolicy:
    """
    Run-wide deterministic hardware guard.

    Every changed parameter must have an explicit rule by default.
    This prevents newly introduced hardware channels from silently
    bypassing transition safety.
    """

    parameter_rules: Mapping[
        str,
        ParameterSafetyRule,
    ]

    reject_unconfigured_changes: bool = True

    max_total_steps: int = 10000

    def __post_init__(self) -> None:
        if self.max_total_steps < 1:
            raise ValueError(
                "max_total_steps must be at least 1"
            )

        for (
            parameter_name,
            rule,
        ) in self.parameter_rules.items():
            if parameter_name not in PARAMETERS:
                raise ValueError(
                    "Unknown SIRIUS parameter in hardware guard: "
                    f"{parameter_name}"
                )

            if not isinstance(
                rule,
                ParameterSafetyRule,
            ):
                raise TypeError(
                    f"Rule for {parameter_name} must be "
                    "ParameterSafetyRule"
                )

    def rule_for(
        self,
        parameter_name: str,
    ) -> ParameterSafetyRule | None:
        return self.parameter_rules.get(
            parameter_name
        )


@dataclass(frozen=True)
class HardwareGuardCoverage:
    """
    Static audit of the deterministic hardware guard.

    required_parameters:
        every currently enabled and optimizable SIRIUS parameter

    missing_parameters:
        parameters that could be optimized but have no explicit rule

    weak_readback_parameters:
        rules that do not require readback

    weak_settling_parameters:
        rules that do not require settling

    A real SIRIUS run is permitted only when ready=True.
    """

    required_parameters: tuple[
        str,
        ...
    ]

    configured_parameters: tuple[
        str,
        ...
    ]

    missing_parameters: tuple[
        str,
        ...
    ]

    weak_readback_parameters: tuple[
        str,
        ...
    ]

    weak_settling_parameters: tuple[
        str,
        ...
    ]

    @property
    def ready(
        self,
    ) -> bool:
        return not (
            self.missing_parameters
            or self.weak_readback_parameters
            or self.weak_settling_parameters
        )


def required_guard_parameters(
) -> tuple[
    str,
    ...
]:
    """
    Return every parameter that can currently be changed by an optimizer.

    This derives the safety requirement from the canonical PARAMETERS
    registry rather than maintaining a second hand-written parameter list.
    """

    return tuple(
        name
        for name, definition
        in PARAMETERS.items()
        if (
            definition.enabled
            and definition.optimizable
        )
    )


def audit_hardware_guard_policy(
    policy: HardwareGuardPolicy,
) -> HardwareGuardCoverage:
    """
    Verify that every enabled/optimizable hardware parameter has an
    explicit strong handshake rule.

    Strong handshake means:

        bounded command step
        + readback required
        + settling required
    """

    if not isinstance(
        policy,
        HardwareGuardPolicy,
    ):
        raise TypeError(
            "policy must be HardwareGuardPolicy"
        )

    required = (
        required_guard_parameters()
    )

    configured = tuple(
        policy.parameter_rules
    )

    missing = tuple(
        name
        for name
        in required
        if name not in policy.parameter_rules
    )

    weak_readback = tuple(
        name
        for name
        in required
        if (
            name in policy.parameter_rules
            and not policy.parameter_rules[
                name
            ].require_readback
        )
    )

    weak_settling = tuple(
        name
        for name
        in required
        if (
            name in policy.parameter_rules
            and not policy.parameter_rules[
                name
            ].require_settling
        )
    )

    return HardwareGuardCoverage(
        required_parameters=required,
        configured_parameters=configured,
        missing_parameters=missing,
        weak_readback_parameters=(
            weak_readback
        ),
        weak_settling_parameters=(
            weak_settling
        ),
    )


def require_complete_hardware_guard(
    policy: HardwareGuardPolicy,
) -> HardwareGuardCoverage:
    """
    Fail closed unless the complete currently optimizable machine surface
    has deterministic transition protection.
    """

    coverage = (
        audit_hardware_guard_policy(
            policy
        )
    )

    if coverage.ready:
        return coverage

    problems = []

    if coverage.missing_parameters:
        problems.append(
            "missing rules: "
            + ", ".join(
                coverage.missing_parameters
            )
        )

    if coverage.weak_readback_parameters:
        problems.append(
            "readback not required: "
            + ", ".join(
                coverage.weak_readback_parameters
            )
        )

    if coverage.weak_settling_parameters:
        problems.append(
            "settling not required: "
            + ", ".join(
                coverage.weak_settling_parameters
            )
        )

    raise HardwareSafetyViolation(
        "Incomplete real-machine HardwareGuardPolicy; "
        + "; ".join(
            problems
        )
    )


def build_strict_hardware_guard(
    max_step_by_parameter: Mapping[
        str,
        float,
    ],
    *,
    max_total_steps: int = 10000,
) -> HardwareGuardPolicy:
    """
    Build the standard real-machine guard.

    Every enabled/optimizable parameter must receive an EXPLICIT max_step.
    No facility-specific step size is invented by SIRIUS.

    All generated rules require:
        readback
        settling
    """

    required = (
        required_guard_parameters()
    )

    missing = tuple(
        name
        for name
        in required
        if name not in max_step_by_parameter
    )

    if missing:
        raise HardwareSafetyViolation(
            "Explicit max_step missing for: "
            + ", ".join(
                missing
            )
        )

    unknown = tuple(
        name
        for name
        in max_step_by_parameter
        if name not in PARAMETERS
    )

    if unknown:
        raise HardwareSafetyViolation(
            "Unknown parameters in max_step configuration: "
            + ", ".join(
                unknown
            )
        )

    rules = {
        name: ParameterSafetyRule(
            max_step=float(
                max_step_by_parameter[
                    name
                ]
            ),
            require_readback=True,
            require_settling=True,
        )
        for name
        in required
    }

    policy = HardwareGuardPolicy(
        parameter_rules=rules,
        reject_unconfigured_changes=True,
        max_total_steps=max_total_steps,
    )

    require_complete_hardware_guard(
        policy
    )

    return policy


@dataclass(frozen=True)
class GuardedTransitionStep:
    """
    One permitted physical command step.

    Only one SIRIUS parameter changes per step.
    """

    parameter_name: str

    command_before: float
    command_after: float

    step_index: int
    step_count: int

    state_before: MachineState
    target_state: MachineState

    @property
    def delta(
        self,
    ) -> float:
        return (
            float(
                self.command_after
            )
            - float(
                self.command_before
            )
        )


@dataclass(frozen=True)
class GuardedTransitionPlan:
    initial_state: MachineState
    target_state: MachineState

    steps: tuple[
        GuardedTransitionStep,
        ...
    ]

    @property
    def command_count(
        self,
    ) -> int:
        return len(
            self.steps
        )


@dataclass(frozen=True)
class ConfirmedHardwareStep:
    planned_step: GuardedTransitionStep

    observed_state: MachineState


@dataclass(frozen=True)
class GuardedTransitionResult:
    plan: GuardedTransitionPlan

    completed_steps: tuple[
        ConfirmedHardwareStep,
        ...
    ]

    final_state: MachineState


StepExecutor = Callable[
    [
        GuardedTransitionStep,
        ParameterSafetyRule,
    ],
    MachineState,
]


def _finite(
    name: str,
    value: float,
) -> float:
    value = float(
        value
    )

    if not math.isfinite(
        value
    ):
        raise HardwareSafetyViolation(
            f"{name} must be finite"
        )

    return value


def _changed_parameters(
    current: MachineState,
    target: MachineState,
) -> tuple[
    str,
    ...
]:
    names = []

    for parameter_name in (
        current.parameters
    ):
        current_value = float(
            current.parameters[
                parameter_name
            ]
        )

        target_value = float(
            target.parameters[
                parameter_name
            ]
        )

        if not math.isclose(
            current_value,
            target_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            names.append(
                parameter_name
            )

    return tuple(
        names
    )


def _validate_transition_endpoints(
    current: MachineState,
    target: MachineState,
) -> None:
    current.validate()
    target.validate()

    if not math.isclose(
        float(
            current.mass_u
        ),
        float(
            target.mass_u
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise HardwareSafetyViolation(
            "Hardware guard cannot change ion mass"
        )

    if (
        set(
            current.parameters
        )
        != set(
            target.parameters
        )
    ):
        raise HardwareSafetyViolation(
            "Current and target states must contain the same "
            "parameter set"
        )

    # Cup selection is an independently acknowledged hardware action.
    # Parameter-transition planning must not hide a simultaneous cup move.
    if current.cup != target.cup:
        raise HardwareSafetyViolation(
            "Parameter hardware guard cannot combine parameter changes "
            "with a Faraday-cup change"
        )

    for parameter_name, value in (
        target.parameters.items()
    ):
        if parameter_name not in PARAMETERS:
            raise HardwareSafetyViolation(
                f"Unknown SIRIUS parameter: {parameter_name}"
            )

        definition = PARAMETERS[
            parameter_name
        ]

        numeric = _finite(
            parameter_name,
            value,
        )

        if not (
            definition.minimum
            <= numeric
            <= definition.maximum
        ):
            raise HardwareSafetyViolation(
                f"{parameter_name}={numeric} outside hard bounds "
                f"{definition.minimum}..{definition.maximum}"
            )


def _step_values(
    start: float,
    stop: float,
    max_step: float,
) -> tuple[
    float,
    ...
]:
    start = _finite(
        "transition start",
        start,
    )

    stop = _finite(
        "transition stop",
        stop,
    )

    max_step = _finite(
        "maximum hardware step",
        max_step,
    )

    if max_step <= 0:
        raise HardwareSafetyViolation(
            "Maximum hardware step must be greater than zero"
        )

    delta = (
        stop
        - start
    )

    if math.isclose(
        delta,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return ()

    number_of_steps = int(
        math.ceil(
            abs(
                delta
            )
            / max_step
        )
    )

    return tuple(
        (
            stop
            if index == number_of_steps
            else start
            + delta
            * index
            / number_of_steps
        )
        for index
        in range(
            1,
            number_of_steps + 1,
        )
    )


def _state_with_command(
    base: MachineState,
    parameter_name: str,
    command_value: float,
) -> MachineState:
    parameters = dict(
        base.parameters
    )

    parameters[
        parameter_name
    ] = float(
        command_value
    )

    readbacks = dict(
        base.readbacks
    )

    # The old readback is no longer evidence for the new command.
    readbacks.pop(
        parameter_name,
        None,
    )

    state = MachineState(
        mass_u=base.mass_u,
        parameters=parameters,
        readbacks=readbacks,
        cup=base.cup,
        stage=base.stage,
        role=base.role,
        rfq=base.rfq,
        fixed_conditions=base.fixed_conditions,
        metadata=base.metadata,
    )

    state.validate()

    return state


def plan_guarded_transition(
    current: MachineState,
    target: MachineState,
    policy: HardwareGuardPolicy,
) -> GuardedTransitionPlan:
    """
    Convert a logical MachineState transition into deterministic,
    bounded, one-parameter-at-a-time hardware steps.

    Planning itself performs no hardware action.
    """

    if not isinstance(
        policy,
        HardwareGuardPolicy,
    ):
        raise TypeError(
            "policy must be HardwareGuardPolicy"
        )

    _validate_transition_endpoints(
        current,
        target,
    )

    changed = (
        _changed_parameters(
            current,
            target,
        )
    )

    for parameter_name in changed:
        if (
            policy.rule_for(
                parameter_name
            )
            is None
            and policy.reject_unconfigured_changes
        ):
            raise HardwareSafetyViolation(
                "No hardware-safety rule configured for changed "
                f"parameter {parameter_name}"
            )

    working_state = (
        current
    )

    raw_steps: list[
        tuple[
            str,
            float,
            float,
            MachineState,
            MachineState,
        ]
    ] = []

    # Preserve the canonical MachineState parameter ordering rather than
    # inventing a physics-dependent global hardware ordering here.
    for parameter_name in changed:
        rule = policy.rule_for(
            parameter_name
        )

        if rule is None:
            # Only possible when reject_unconfigured_changes=False.
            values = (
                float(
                    target.parameters[
                        parameter_name
                    ]
                ),
            )
        else:
            values = _step_values(
                float(
                    working_state.parameters[
                        parameter_name
                    ]
                ),
                float(
                    target.parameters[
                        parameter_name
                    ]
                ),
                float(
                    rule.max_step
                ),
            )

        for command_value in values:
            before = (
                working_state
            )

            after = _state_with_command(
                before,
                parameter_name,
                command_value,
            )

            raw_steps.append(
                (
                    parameter_name,
                    float(
                        before.parameters[
                            parameter_name
                        ]
                    ),
                    float(
                        command_value
                    ),
                    before,
                    after,
                )
            )

            working_state = (
                after
            )

            if (
                len(
                    raw_steps
                )
                > policy.max_total_steps
            ):
                raise HardwareSafetyViolation(
                    "Transition requires more than "
                    f"{policy.max_total_steps} hardware commands"
                )

    steps = tuple(
        GuardedTransitionStep(
            parameter_name=parameter_name,
            command_before=command_before,
            command_after=command_after,
            step_index=index,
            step_count=len(
                raw_steps
            ),
            state_before=state_before,
            target_state=target_state,
        )
        for index, (
            parameter_name,
            command_before,
            command_after,
            state_before,
            target_state,
        )
        in enumerate(
            raw_steps,
            start=1,
        )
    )

    return GuardedTransitionPlan(
        initial_state=current,
        target_state=target,
        steps=steps,
    )


def execute_guarded_transition(
    plan: GuardedTransitionPlan,
    policy: HardwareGuardPolicy,
    executor: StepExecutor,
) -> GuardedTransitionResult:
    """
    Execute a previously validated plan serially.

    The executor MUST return only after the requested step has completed
    its required readback/settling handshake.

    Therefore the next command cannot be issued before the previous step
    has been positively completed.

    On failure, execution stops immediately and exposes the last positively
    confirmed state. No automatic rollback occurs.
    """

    if not isinstance(
        plan,
        GuardedTransitionPlan,
    ):
        raise TypeError(
            "plan must be GuardedTransitionPlan"
        )

    if not isinstance(
        policy,
        HardwareGuardPolicy,
    ):
        raise TypeError(
            "policy must be HardwareGuardPolicy"
        )

    last_state = (
        plan.initial_state
    )

    completed: list[
        ConfirmedHardwareStep
    ] = []

    try:
        for step in (
            plan.steps
        ):
            rule = policy.rule_for(
                step.parameter_name
            )

            if rule is None:
                if policy.reject_unconfigured_changes:
                    raise HardwareSafetyViolation(
                        "Missing safety rule during execution for "
                        f"{step.parameter_name}"
                    )

                raise HardwareSafetyViolation(
                    "Unsafe execution of an unconfigured parameter "
                    "is not supported"
                )

            # Strong sequencing invariant:
            # this function is synchronous. The next iteration cannot start
            # until executor() has returned a positively observed state.
            observed_state = executor(
                step,
                rule,
            )

            if not isinstance(
                observed_state,
                MachineState,
            ):
                raise TypeError(
                    "Hardware step executor must return MachineState"
                )

            observed_state.validate()

            if not math.isclose(
                float(
                    observed_state.parameters[
                        step.parameter_name
                    ]
                ),
                float(
                    step.command_after
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise HardwareSafetyViolation(
                    "Executor returned a state whose command does not "
                    "match the completed guarded step"
                )

            completed.append(
                ConfirmedHardwareStep(
                    planned_step=step,
                    observed_state=observed_state,
                )
            )

            last_state = (
                observed_state
            )

    except Exception as exc:
        if isinstance(
            exc,
            HardwareTransitionFailure,
        ):
            raise

        raise HardwareTransitionFailure(
            "Guarded hardware transition stopped after "
            f"{len(completed)} completed step(s): {exc}",
            last_state=last_state,
            completed_steps=len(
                completed
            ),
        ) from exc

    # Preserve the exact requested target semantics while retaining the
    # last observed readbacks.
    final_state = MachineState(
        mass_u=plan.target_state.mass_u,
        parameters=dict(
            plan.target_state.parameters
        ),
        readbacks=dict(
            last_state.readbacks
        ),
        cup=plan.target_state.cup,
        stage=plan.target_state.stage,
        role=plan.target_state.role,
        rfq=plan.target_state.rfq,
        fixed_conditions=plan.target_state.fixed_conditions,
        metadata=plan.target_state.metadata,
    )

    final_state.validate()

    return GuardedTransitionResult(
        plan=plan,
        completed_steps=tuple(
            completed
        ),
        final_state=final_state,
    )