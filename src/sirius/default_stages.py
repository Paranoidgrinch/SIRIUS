from __future__ import annotations

import importlib
import inspect
from dataclasses import (
    dataclass,
    field,
    fields,
    is_dataclass,
    replace,
)
from typing import (
    Any,
    Callable,
    Mapping,
)

from sirius.cup3_coordinates import (
    EndElectrodeCoordinatePolicy,
)
from sirius.orchestrator import (
    SiriusRunContext,
    SiriusStage,
    SiriusStageRunners,
)
from sirius.qpt_scan2d import (
    QPT2DScanPolicy,
)
from sirius.state import (
    MachineState,
)


class DefaultStageWiringError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class StageFunctionSpec:
    stage: SiriusStage

    module_name: str

    function_names: tuple[
        str,
        ...
    ]


STAGE_FUNCTION_SPECS = {
    SiriusStage.CUP1: (
        StageFunctionSpec(
            stage=SiriusStage.CUP1,
            module_name=(
                "sirius.cup1_optimizer"
            ),
            function_names=(
                "optimize_cup1",
            ),
        )
    ),

    SiriusStage.CUP2: (
        StageFunctionSpec(
            stage=SiriusStage.CUP2,
            module_name=(
                "sirius.cup2_optimizer"
            ),
            function_names=(
                "optimize_cup2",
            ),
        )
    ),

    SiriusStage.CUP3: (
        StageFunctionSpec(
            stage=SiriusStage.CUP3,
            module_name=(
                "sirius.cup3_optimizer"
            ),
            function_names=(
                "optimize_cup3",
            ),
        )
    ),

    SiriusStage.CUP4: (
        StageFunctionSpec(
            stage=SiriusStage.CUP4,
            module_name=(
                "sirius.cup4_optimizer"
            ),
            function_names=(
                "optimize_cup4",
            ),
        )
    ),

    SiriusStage.CUP5: (
        StageFunctionSpec(
            stage=SiriusStage.CUP5,
            module_name=(
                "sirius.cup5_optimizer"
            ),
            function_names=(
                "optimize_cup5",
            ),
        )
    ),

    SiriusStage.CUP6: (
        StageFunctionSpec(
            stage=SiriusStage.CUP6,
            module_name=(
                "sirius.cup6_optimizer"
            ),
            function_names=(
                "optimize_cup6",
            ),
        )
    ),

    SiriusStage.FINAL_CHARACTERIZATION: (
        StageFunctionSpec(
            stage=(
                SiriusStage
                .FINAL_CHARACTERIZATION
            ),
            module_name=(
                "sirius.final_characterization"
            ),
            function_names=(
                "characterize_final_transmission",
            ),
        )
    ),
}


@dataclass(frozen=True)
class DefaultStageResources:
    """
    Resources exposed to the real stage functions.

    shared:
        values used by several stages, e.g.
        tracker, profile, settling policies, measurement policy.

    cupN:
        stage-specific values, especially optimizer policies and hardware
        interfaces that only exist for one section of the beamline.

    Stage-specific values override shared values of the same name.
    """

    shared: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    cup1: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    cup2: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    cup3: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    cup4: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    cup5: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    cup6: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    final_characterization: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def for_stage(
        self,
        stage: SiriusStage,
    ) -> dict[
        str,
        Any,
    ]:
        stage_mapping = {
            SiriusStage.CUP1:
                self.cup1,

            SiriusStage.CUP2:
                self.cup2,

            SiriusStage.CUP3:
                self.cup3,

            SiriusStage.CUP4:
                self.cup4,

            SiriusStage.CUP5:
                self.cup5,

            SiriusStage.CUP6:
                self.cup6,

            SiriusStage.FINAL_CHARACTERIZATION:
                self.final_characterization,
        }[
            stage
        ]

        return {
            **dict(
                self.shared
            ),
            **dict(
                stage_mapping
            ),
        }


def _load_stage_function(
    stage: SiriusStage,
) -> Callable:
    spec = STAGE_FUNCTION_SPECS[
        stage
    ]

    module = importlib.import_module(
        spec.module_name
    )

    for function_name in (
        spec.function_names
    ):
        candidate = getattr(
            module,
            function_name,
            None,
        )

        if callable(
            candidate
        ):
            return candidate

    raise DefaultStageWiringError(
        "Could not resolve SIRIUS stage function for "
        f"{stage.value}; module={spec.module_name}, "
        f"expected one of {spec.function_names}"
    )


def _state_parameter_names(
    stage: SiriusStage,
) -> set[
    str
]:
    common = {
        "current",
        "current_state",
        "state",
    }

    specific = {
        SiriusStage.CUP1: {
            "initial_state",
        },

        SiriusStage.CUP2: {
            "cup1_reference_state",
        },

        SiriusStage.CUP3: {
            "cup2_reference_state",
        },

        SiriusStage.CUP4: {
            "cup3_reference_state",
        },

        SiriusStage.CUP5: {
            "cup4_reference_state",
        },

        SiriusStage.CUP6: {
            "cup5_reference_state",
        },

        SiriusStage.FINAL_CHARACTERIZATION: {
            "final_cup6_state",
            "cup6_state",
        },
    }[
        stage
    ]

    return (
        common
        | specific
    )


def _bind_hardware_safety(
    value: Any,
    context: SiriusRunContext,
):
    """
    Recursively bind run-wide hardware safety into optimizer policies.

    This lets Cup-3 policy trees contain EndElectrodeCoordinatePolicy and
    Cup-4/Cup-5 policy trees contain QPT2DScanPolicy without the default
    stage assembler needing to know the field names of those parent
    policies.
    """

    if isinstance(
        value,
        EndElectrodeCoordinatePolicy,
    ):
        return (
            context
            .bind_end_electrode_policy(
                value
            )
        )

    if isinstance(
        value,
        QPT2DScanPolicy,
    ):
        return (
            context
            .bind_qpt_scan_policy(
                value
            )
        )

    if isinstance(
        value,
        tuple,
    ):
        changed = False
        output = []

        for item in value:
            bound = (
                _bind_hardware_safety(
                    item,
                    context,
                )
            )

            output.append(
                bound
            )

            changed = (
                changed
                or bound is not item
            )

        if not changed:
            return value

        return tuple(
            output
        )

    if isinstance(
        value,
        list,
    ):
        changed = False
        output = []

        for item in value:
            bound = (
                _bind_hardware_safety(
                    item,
                    context,
                )
            )

            output.append(
                bound
            )

            changed = (
                changed
                or bound is not item
            )

        if not changed:
            return value

        return output

    if isinstance(
        value,
        dict,
    ):
        output = {}

        changed = False

        for key, item in (
            value.items()
        ):
            bound = (
                _bind_hardware_safety(
                    item,
                    context,
                )
            )

            output[
                key
            ] = bound

            changed = (
                changed
                or bound is not item
            )

        if not changed:
            return value

        return output

    if (
        is_dataclass(
            value
        )
        and not isinstance(
            value,
            MachineState,
        )
    ):
        replacements = {}

        for dataclass_field in (
            fields(
                value
            )
        ):
            if not dataclass_field.init:
                continue

            original = getattr(
                value,
                dataclass_field.name,
            )

            bound = (
                _bind_hardware_safety(
                    original,
                    context,
                )
            )

            if bound is not original:
                replacements[
                    dataclass_field.name
                ] = bound

        if replacements:
            return replace(
                value,
                **replacements,
            )

    return value


def _automatic_value(
    name: str,
    *,
    stage: SiriusStage,
    state: MachineState,
    context: SiriusRunContext,
) -> tuple[
    bool,
    Any,
]:
    if name in (
        "adapter",
        "backend_adapter",
    ):
        return (
            True,
            context.adapter,
        )

    if name in (
        _state_parameter_names(
            stage
        )
    ):
        return (
            True,
            state,
        )

    if name in (
        "logger",
        "run_logger",
    ):
        return (
            True,
            context.logger,
        )

    if name in (
        "context",
        "run_context",
    ):
        return (
            True,
            context,
        )

    if name == "hardware_safety":
        return (
            True,
            context.hardware_safety,
        )

    if name == "cup_selection_policy":
        return (
            True,
            context.cup_selection_policy,
        )

    return (
        False,
        None,
    )


def _prepare_call(
    function: Callable,
    *,
    stage: SiriusStage,
    state: MachineState,
    context: SiriusRunContext,
    resources: Mapping[
        str,
        Any,
    ],
) -> tuple[
    list[
        Any
    ],
    dict[
        str,
        Any
    ],
]:
    signature = inspect.signature(
        function
    )

    args: list[
        Any
    ] = []

    kwargs: dict[
        str,
        Any
    ] = {}

    missing: list[
        str
    ] = []

    prepared_resources = {
        name: (
            _bind_hardware_safety(
                value,
                context,
            )
        )
        for name, value
        in resources.items()
    }

    for parameter in (
        signature.parameters.values()
    ):
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        automatic, value = (
            _automatic_value(
                parameter.name,
                stage=stage,
                state=state,
                context=context,
            )
        )

        supplied = automatic

        if (
            not supplied
            and parameter.name
            in prepared_resources
        ):
            value = prepared_resources[
                parameter.name
            ]

            supplied = True

        if not supplied:
            if (
                parameter.default
                is not inspect.Parameter.empty
            ):
                continue

            missing.append(
                parameter.name
            )

            continue

        if (
            parameter.kind
            == inspect.Parameter.POSITIONAL_ONLY
        ):
            args.append(
                value
            )

        else:
            kwargs[
                parameter.name
            ] = value

    if missing:
        raise DefaultStageWiringError(
            f"Missing resources for {stage.value} / "
            f"{function.__module__}.{function.__name__}: "
            f"{', '.join(sorted(missing))}"
        )

    return (
        args,
        kwargs,
    )


def _invoke_stage(
    function: Callable,
    *,
    stage: SiriusStage,
    state: MachineState,
    context: SiriusRunContext,
    resources: Mapping[
        str,
        Any,
    ],
):
    args, kwargs = (
        _prepare_call(
            function,
            stage=stage,
            state=state,
            context=context,
            resources=resources,
        )
    )

    return function(
        *args,
        **kwargs,
    )


def _make_runner(
    stage: SiriusStage,
    resources: DefaultStageResources,
):
    function = (
        _load_stage_function(
            stage
        )
    )

    stage_resources = (
        resources.for_stage(
            stage
        )
    )

    def runner(
        state: MachineState,
        context: SiriusRunContext,
    ):
        return _invoke_stage(
            function,
            stage=stage,
            state=state,
            context=context,
            resources=stage_resources,
        )

    return runner


def build_default_stages(
    resources: DefaultStageResources,
) -> SiriusStageRunners:
    """
    Assemble the actual SIRIUS Cup-1..6 optimizers and final
    characterization into the generic run orchestrator.
    """

    if not isinstance(
        resources,
        DefaultStageResources,
    ):
        raise TypeError(
            "resources must be DefaultStageResources"
        )

    return SiriusStageRunners(
        cup1=_make_runner(
            SiriusStage.CUP1,
            resources,
        ),

        cup2=_make_runner(
            SiriusStage.CUP2,
            resources,
        ),

        cup3=_make_runner(
            SiriusStage.CUP3,
            resources,
        ),

        cup4=_make_runner(
            SiriusStage.CUP4,
            resources,
        ),

        cup5=_make_runner(
            SiriusStage.CUP5,
            resources,
        ),

        cup6=_make_runner(
            SiriusStage.CUP6,
            resources,
        ),

        final_characterization=(
            _make_runner(
                SiriusStage
                .FINAL_CHARACTERIZATION,
                resources,
            )
        ),
    )


def inspect_default_stage_requirements(
) -> dict[
    str,
    tuple[
        str,
        ...
    ],
]:
    """
    Inspect the real stage functions without executing hardware code.

    Returned names exclude the values automatically supplied by the run
    context/state assembler and parameters that already have defaults.

    This is useful for building a runtime resource bundle and for detecting
    optimizer-interface changes during tests.
    """

    result = {}

    dummy_state_names = {
        stage: (
            _state_parameter_names(
                stage
            )
        )
        for stage in STAGE_FUNCTION_SPECS
    }

    automatic_common = {
        "adapter",
        "backend_adapter",
        "logger",
        "run_logger",
        "context",
        "run_context",
        "hardware_safety",
        "cup_selection_policy",
    }

    for stage in (
        STAGE_FUNCTION_SPECS
    ):
        function = (
            _load_stage_function(
                stage
            )
        )

        signature = (
            inspect.signature(
                function
            )
        )

        required = []

        automatic = (
            automatic_common
            | dummy_state_names[
                stage
            ]
        )

        for parameter in (
            signature.parameters.values()
        ):
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if parameter.name in automatic:
                continue

            if (
                parameter.default
                is not inspect.Parameter.empty
            ):
                continue

            required.append(
                parameter.name
            )

        result[
            stage.value
        ] = tuple(
            required
        )

    return result