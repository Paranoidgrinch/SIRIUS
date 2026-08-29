from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

from sirius.parameters import PARAMETERS
from sirius.settling import SettlingPolicy
from sirius.state import MachineState
from sirius.transition import (
    AppliedStateResult,
)
from sirius.safe_transition import apply_state


class CoupledTransitionError(RuntimeError):
    """
    Raised when a bounded coupled transition cannot be completed.

    No automatic rollback is attempted. The last positively observed
    state is preserved in the exception.
    """

    def __init__(
        self,
        message: str,
        *,
        last_state: MachineState,
        completed_macro_steps: int,
        completed_channel_steps: int,
    ):
        super().__init__(
            message
        )

        self.last_state = last_state
        self.completed_macro_steps = (
            completed_macro_steps
        )
        self.completed_channel_steps = (
            completed_channel_steps
        )


@dataclass(frozen=True)
class CoupledTransitionPolicy:
    """
    Safety policy for a logically coupled multi-channel transition.

    parameter_order:
        Explicit physical command order inside each interpolated macro
        step.

    max_step_by_parameter:
        Maximum allowed command change for one physical channel command.

    The values are software safety limits, not claims about the hardware's
    intrinsic safe slew rate.
    """

    parameter_order: tuple[
        str,
        ...
    ]

    max_step_by_parameter: Mapping[
        str,
        float,
    ]

    def __post_init__(self) -> None:
        if not self.parameter_order:
            raise ValueError(
                "parameter_order must not be empty"
            )

        if len(
            set(
                self.parameter_order
            )
        ) != len(
            self.parameter_order
        ):
            raise ValueError(
                "parameter_order contains duplicates"
            )

        for parameter_name in (
            self.parameter_order
        ):
            if parameter_name not in PARAMETERS:
                raise KeyError(
                    f"Unknown SIRIUS parameter: {parameter_name}"
                )

            if (
                parameter_name
                not in self.max_step_by_parameter
            ):
                raise KeyError(
                    "Missing maximum transition step for "
                    f"{parameter_name}"
                )

            step = float(
                self.max_step_by_parameter[
                    parameter_name
                ]
            )

            if (
                not math.isfinite(
                    step
                )
                or step <= 0
            ):
                raise ValueError(
                    f"Invalid maximum step for {parameter_name}: {step}"
                )


@dataclass(frozen=True)
class CoupledTransitionMacroStep:
    index: int

    fraction: float

    state: MachineState

    deltas_from_previous: dict[
        str,
        float,
    ]


@dataclass(frozen=True)
class CoupledTransitionPlan:
    source_state_id: str
    target_state_id: str

    parameter_names: tuple[
        str,
        ...
    ]

    macro_steps: tuple[
        CoupledTransitionMacroStep,
        ...
    ]

    @property
    def step_count(
        self,
    ) -> int:
        return len(
            self.macro_steps
        )


@dataclass(frozen=True)
class CoupledTransitionChannelStep:
    macro_step_index: int

    parameter_name: str

    requested_value: float

    transition: AppliedStateResult


@dataclass(frozen=True)
class CoupledTransitionResult:
    requested_state: MachineState

    plan: CoupledTransitionPlan

    channel_steps: tuple[
        CoupledTransitionChannelStep,
        ...
    ]

    final_state: MachineState

    @property
    def channel_step_count(
        self,
    ) -> int:
        return len(
            self.channel_steps
        )


def _commands_equal(
    first: float,
    second: float,
) -> bool:
    return math.isclose(
        float(first),
        float(second),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _validate_endpoint_pair(
    current: MachineState,
    target: MachineState,
    policy: CoupledTransitionPolicy,
) -> None:
    current.validate()
    target.validate()

    if not math.isclose(
        current.mass_u,
        target.mass_u,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Coupled transition may not change ion mass"
        )

    if current.cup != target.cup:
        raise ValueError(
            "Coupled transition may not change Faraday cup"
        )

    if current.stage != target.stage:
        raise ValueError(
            "Coupled transition may not change optimization stage"
        )

    if current.rfq != target.rfq:
        raise ValueError(
            "Coupled transition may not change RFQ configuration"
        )

    if (
        set(
            current.parameters
        )
        != set(
            target.parameters
        )
    ):
        raise ValueError(
            "Source and target must contain the same parameter set"
        )

    coupled = set(
        policy.parameter_order
    )

    for parameter_name in (
        current.parameters
    ):
        if parameter_name in coupled:
            continue

        if not _commands_equal(
            current.parameters[
                parameter_name
            ],
            target.parameters[
                parameter_name
            ],
        ):
            raise ValueError(
                "Coupled transition contains an unapproved command "
                f"change: {parameter_name}"
            )


def _required_macro_steps(
    current: MachineState,
    target: MachineState,
    policy: CoupledTransitionPolicy,
) -> int:
    required = 1

    for parameter_name in (
        policy.parameter_order
    ):
        delta = abs(
            float(
                target.parameters[
                    parameter_name
                ]
            )
            - float(
                current.parameters[
                    parameter_name
                ]
            )
        )

        maximum_step = float(
            policy.max_step_by_parameter[
                parameter_name
            ]
        )

        parameter_steps = max(
            1,
            int(
                math.ceil(
                    delta
                    / maximum_step
                )
            ),
        )

        required = max(
            required,
            parameter_steps,
        )

    return required


def _interpolated_state(
    current: MachineState,
    target: MachineState,
    policy: CoupledTransitionPolicy,
    *,
    fraction: float,
    macro_step_index: int,
) -> MachineState:
    if not (
        0.0
        < fraction
        <= 1.0
    ):
        raise ValueError(
            "Interpolation fraction must be in (0, 1]"
        )

    parameters = dict(
        current.parameters
    )

    readbacks = dict(
        current.readbacks
    )

    for parameter_name in (
        policy.parameter_order
    ):
        source_value = float(
            current.parameters[
                parameter_name
            ]
        )

        target_value = float(
            target.parameters[
                parameter_name
            ]
        )

        value = (
            source_value
            + fraction
            * (
                target_value
                - source_value
            )
        )

        # Remove old physical readback. It no longer belongs to this
        # interpolated command state.
        readbacks.pop(
            parameter_name,
            None,
        )

        parameters[
            parameter_name
        ] = value

    # The last interpolation point must reproduce the target commands
    # exactly rather than merely within floating-point arithmetic.
    if math.isclose(
        fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        for parameter_name in (
            policy.parameter_order
        ):
            parameters[
                parameter_name
            ] = float(
                target.parameters[
                    parameter_name
                ]
            )

    result = MachineState(
        mass_u=current.mass_u,
        parameters=parameters,
        readbacks=readbacks,
        cup=current.cup,
        stage=current.stage,
        role="coupled_transition_step",
        rfq=deepcopy(
            current.rfq
        ),
        fixed_conditions=deepcopy(
            current.fixed_conditions
        ),
        metadata={
            **deepcopy(
                current.metadata
            ),
            "coupled_transition": True,
            "coupled_transition_target_state_id": (
                target.state_id
            ),
            "coupled_transition_macro_step": (
                macro_step_index
            ),
            "coupled_transition_fraction": (
                float(
                    fraction
                )
            ),
        },
    )

    # This is the important hard-limit guard. Every interpolated state
    # must itself be a valid SIRIUS machine state before hardware access.
    result.validate()

    return result


def plan_coupled_transition(
    current: MachineState,
    target: MachineState,
    policy: CoupledTransitionPolicy,
) -> CoupledTransitionPlan:
    """
    Build a bounded linear command-space path between two valid states.

    Because SIRIUS hard parameter limits are intervals, interpolation
    between two valid endpoints remains within each individual channel's
    configured hard bounds.

    Every generated MachineState is nevertheless validated explicitly.
    """

    _validate_endpoint_pair(
        current,
        target,
        policy,
    )

    step_count = (
        _required_macro_steps(
            current,
            target,
            policy,
        )
    )

    macro_steps: list[
        CoupledTransitionMacroStep
    ] = []

    previous_values = {
        parameter_name: float(
            current.parameters[
                parameter_name
            ]
        )
        for parameter_name
        in policy.parameter_order
    }

    for index in range(
        1,
        step_count + 1,
    ):
        fraction = (
            index
            / step_count
        )

        state = _interpolated_state(
            current,
            target,
            policy,
            fraction=fraction,
            macro_step_index=index,
        )

        deltas: dict[
            str,
            float,
        ] = {}

        for parameter_name in (
            policy.parameter_order
        ):
            value = float(
                state.parameters[
                    parameter_name
                ]
            )

            delta = (
                value
                - previous_values[
                    parameter_name
                ]
            )

            maximum_step = float(
                policy.max_step_by_parameter[
                    parameter_name
                ]
            )

            if (
                abs(
                    delta
                )
                > maximum_step
                + 1e-9
            ):
                raise RuntimeError(
                    "Internal coupled-transition planner exceeded "
                    f"maximum step for {parameter_name}: "
                    f"{delta} V"
                )

            deltas[
                parameter_name
            ] = delta

            previous_values[
                parameter_name
            ] = value

        macro_steps.append(
            CoupledTransitionMacroStep(
                index=index,
                fraction=float(
                    fraction
                ),
                state=state,
                deltas_from_previous=(
                    deltas
                ),
            )
        )

    return CoupledTransitionPlan(
        source_state_id=(
            current.state_id
        ),
        target_state_id=(
            target.state_id
        ),
        parameter_names=(
            policy.parameter_order
        ),
        macro_steps=tuple(
            macro_steps
        ),
    )


def _single_parameter_target(
    current: MachineState,
    macro_target: MachineState,
    parameter_name: str,
    *,
    macro_step_index: int,
) -> MachineState:
    parameters = dict(
        current.parameters
    )

    parameters[
        parameter_name
    ] = float(
        macro_target.parameters[
            parameter_name
        ]
    )

    readbacks = dict(
        current.readbacks
    )

    readbacks.pop(
        parameter_name,
        None,
    )

    result = MachineState(
        mass_u=current.mass_u,
        parameters=parameters,
        readbacks=readbacks,
        cup=current.cup,
        stage=current.stage,
        role="coupled_transition_channel_step",
        rfq=deepcopy(
            current.rfq
        ),
        fixed_conditions=deepcopy(
            current.fixed_conditions
        ),
        metadata={
            **deepcopy(
                current.metadata
            ),
            "coupled_transition": True,
            "coupled_transition_macro_step": (
                macro_step_index
            ),
            "coupled_transition_parameter": (
                parameter_name
            ),
        },
    )

    result.validate()

    return result


def _assert_macro_target_reached(
    physical_state: MachineState,
    macro_target: MachineState,
    parameter_names: tuple[
        str,
        ...
    ],
) -> None:
    for parameter_name in (
        parameter_names
    ):
        if not _commands_equal(
            physical_state.parameters[
                parameter_name
            ],
            macro_target.parameters[
                parameter_name
            ],
        ):
            raise CoupledTransitionError(
                "Coupled macro-step command state was not reached",
                last_state=physical_state,
                completed_macro_steps=0,
                completed_channel_steps=0,
            )


def _finalize_observed_target_state(
    observed: MachineState,
    target: MachineState,
) -> MachineState:
    """
    Preserve the physical readbacks reached by the coupled executor while
    restoring the semantic identity of the originally requested target.

    Internal micro-step metadata must not leak out as the final scan state.
    """

    result = MachineState(
        mass_u=target.mass_u,
        parameters=dict(
            target.parameters
        ),
        readbacks=dict(
            observed.readbacks
        ),
        cup=target.cup,
        stage=target.stage,
        role=target.role,
        rfq=deepcopy(
            target.rfq
        ),
        fixed_conditions=deepcopy(
            target.fixed_conditions
        ),
        metadata=deepcopy(
            target.metadata
        ),
    )

    result.validate()

    return result


def apply_coupled_transition(
    adapter,
    current: MachineState,
    target: MachineState,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    policy: CoupledTransitionPolicy,
    *,
    logger=None,
) -> CoupledTransitionResult:
    """
    Execute a coupled command transition as bounded sequential microsteps.

    Important:
        This does NOT claim atomic hardware updates.

    Instead, each macro interpolation point is reached one physical
    channel at a time in the explicit policy.parameter_order. Every
    individual channel command is settled before proceeding.

    The maximum physical command jump is therefore bounded even though
    FLAVIA/hardware updates remain sequential.
    """

    plan = plan_coupled_transition(
        current,
        target,
        policy,
    )

    for parameter_name in (
        policy.parameter_order
    ):
        if parameter_name not in settling_policies:
            raise KeyError(
                "No settling policy configured for "
                f"{parameter_name}"
            )

    physical_state = current

    channel_steps: list[
        CoupledTransitionChannelStep
    ] = []

    completed_macro_steps = 0

    for macro_step in (
        plan.macro_steps
    ):
        for parameter_name in (
            policy.parameter_order
        ):
            requested_value = float(
                macro_step.state.parameters[
                    parameter_name
                ]
            )

            current_value = float(
                physical_state.parameters[
                    parameter_name
                ]
            )

            if _commands_equal(
                current_value,
                requested_value,
            ):
                continue

            micro_target = (
                _single_parameter_target(
                    physical_state,
                    macro_step.state,
                    parameter_name,
                    macro_step_index=(
                        macro_step.index
                    ),
                )
            )

            try:
                transition = apply_state(
                    adapter,
                    current=physical_state,
                    target=micro_target,
                    settling_policies=(
                        settling_policies
                    ),
                    select_target_cup=False,
                )

            except Exception as exc:
                raise CoupledTransitionError(
                    "Coupled transition failed while applying "
                    f"{parameter_name} in macro step "
                    f"{macro_step.index}/{plan.step_count}",
                    last_state=physical_state,
                    completed_macro_steps=(
                        completed_macro_steps
                    ),
                    completed_channel_steps=(
                        len(
                            channel_steps
                        )
                    ),
                ) from exc

            physical_state = (
                transition.observed_state
            )

            channel_step = (
                CoupledTransitionChannelStep(
                    macro_step_index=(
                        macro_step.index
                    ),
                    parameter_name=(
                        parameter_name
                    ),
                    requested_value=(
                        requested_value
                    ),
                    transition=(
                        transition
                    ),
                )
            )

            channel_steps.append(
                channel_step
            )

            if logger is not None:
                logger.log_state_transition(
                    transition
                )

        try:
            _assert_macro_target_reached(
                physical_state,
                macro_step.state,
                policy.parameter_order,
            )

        except CoupledTransitionError as exc:
            raise CoupledTransitionError(
                "Coupled transition failed to reach macro step "
                f"{macro_step.index}/{plan.step_count}",
                last_state=physical_state,
                completed_macro_steps=(
                    completed_macro_steps
                ),
                completed_channel_steps=(
                    len(
                        channel_steps
                    )
                ),
            ) from exc

        completed_macro_steps += 1

    # Commands must exactly match the requested final coupled endpoint.
    for parameter_name in (
        policy.parameter_order
    ):
        if not _commands_equal(
            physical_state.parameters[
                parameter_name
            ],
            target.parameters[
                parameter_name
            ],
        ):
            raise CoupledTransitionError(
                "Final coupled command state does not match target",
                last_state=physical_state,
                completed_macro_steps=(
                    completed_macro_steps
                ),
                completed_channel_steps=(
                    len(
                        channel_steps
                    )
                ),
            )

    final_state = _finalize_observed_target_state(
        physical_state,
        target,
    )

    if logger is not None:
        logger.log_event(
            "coupled_transition_completed",
            {
                "source_state_id": (
                    current.state_id
                ),
                "target_state_id": (
                    target.state_id
                ),
                "final_state_id": (
                    final_state.state_id
                ),
                "parameters": list(
                    policy.parameter_order
                ),
                "macro_steps": (
                    plan.step_count
                ),
                "channel_steps": (
                    len(
                        channel_steps
                    )
                ),
            },
        )

    return CoupledTransitionResult(
        requested_state=target,
        plan=plan,
        channel_steps=tuple(
            channel_steps
        ),
        final_state=(
            final_state
        ),
    )


def qpt_transition_policy(
    *,
    max_step_v: float,
    parameter_order: tuple[
        str,
        ...
    ] = (
        "quadrupole1_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole3_voltage_v",
    ),
) -> CoupledTransitionPolicy:
    """
    Create a QPT transition policy.

    The default order is deterministic only; it is not a claim about
    physically preferred QPT switching order.
    """

    expected = {
        "quadrupole1_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole3_voltage_v",
    }

    if set(
        parameter_order
    ) != expected:
        raise ValueError(
            "QPT transition order must contain QPT1, QPT2, and QPT3 "
            "exactly once"
        )

    return CoupledTransitionPolicy(
        parameter_order=(
            parameter_order
        ),
        max_step_by_parameter={
            parameter_name: float(
                max_step_v
            )
            for parameter_name
            in parameter_order
        },
    )


def cooler_end_transition_policy(
    *,
    max_step_v: float,
    parameter_order: tuple[
        str,
        ...
    ] = (
        "deceleration_voltage_v",
        "acceleration_voltage_v",
    ),
) -> CoupledTransitionPolicy:
    """
    Create an ion-cooler entrance/exit transition policy.

    The default order is deterministic only. No laboratory polarity or
    physically preferred switching direction is inferred.
    """

    expected = {
        "deceleration_voltage_v",
        "acceleration_voltage_v",
    }

    if set(
        parameter_order
    ) != expected:
        raise ValueError(
            "Cooler-end transition order must contain deceleration and "
            "acceleration exactly once"
        )

    return CoupledTransitionPolicy(
        parameter_order=(
            parameter_order
        ),
        max_step_by_parameter={
            parameter_name: float(
                max_step_v
            )
            for parameter_name
            in parameter_order
        },
    )