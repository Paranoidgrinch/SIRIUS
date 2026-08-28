from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
)

from sirius.run_config import (
    SiriusRunConfig,
)
from sirius.state import (
    MachineState,
)


class SiriusStage(str, Enum):
    CUP1 = "cup1"
    CUP2 = "cup2"
    CUP3 = "cup3"
    CUP4 = "cup4"
    CUP5 = "cup5"
    CUP6 = "cup6"

    FINAL_CHARACTERIZATION = (
        "final_characterization"
    )


OPTIMIZATION_STAGE_SEQUENCE = (
    SiriusStage.CUP1,
    SiriusStage.CUP2,
    SiriusStage.CUP3,
    SiriusStage.CUP4,
    SiriusStage.CUP5,
    SiriusStage.CUP6,
)


class SiriusRunError(RuntimeError):
    """
    Fail-fast SIRIUS run error.

    last_state is the last positively completed machine state.

    No automatic rollback is implied.
    """

    def __init__(
        self,
        message: str,
        *,
        failed_stage: SiriusStage,
        last_state: MachineState,
    ):
        super().__init__(
            message
        )

        self.failed_stage = (
            failed_stage
        )

        self.last_state = (
            last_state
        )


@dataclass(frozen=True)
class SiriusRunContext:
    """
    Shared context passed to every stage runner.

    Stage implementations can obtain all run-wide hardware-safety
    policies here without each optimizer inventing its own values.
    """

    config: SiriusRunConfig

    adapter: Any

    logger: Any = None

    @property
    def hardware_safety(
        self,
    ):
        return (
            self.config.hardware_safety
        )

    @property
    def cup_selection_policy(
        self,
    ):
        return (
            self.hardware_safety
            .cup_selection_policy
        )

    def bind_end_electrode_policy(
        self,
        policy,
    ):
        return (
            self.hardware_safety
            .bind_end_electrode_policy(
                policy
            )
        )

    def bind_qpt_scan_policy(
        self,
        policy,
    ):
        return (
            self.hardware_safety
            .bind_qpt_scan_policy(
                policy
            )
        )


StageRunner = Callable[
    [
        MachineState,
        SiriusRunContext,
    ],
    Any,
]


@dataclass(frozen=True)
class SiriusStageRunners:
    """
    Concrete stage implementations used by one SIRIUS run.

    Each runner receives:
        current physical MachineState
        shared SiriusRunContext

    Each runner must return either:
        MachineState
        or an object exposing .final_state
    """

    cup1: StageRunner
    cup2: StageRunner
    cup3: StageRunner
    cup4: StageRunner
    cup5: StageRunner
    cup6: StageRunner

    final_characterization: (
        StageRunner | None
    ) = None

    def runner_for(
        self,
        stage: SiriusStage,
    ) -> StageRunner:
        mapping = {
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
        }

        if (
            stage
            == SiriusStage.FINAL_CHARACTERIZATION
        ):
            if (
                self.final_characterization
                is None
            ):
                raise ValueError(
                    "No final-characterization runner configured"
                )

            return (
                self.final_characterization
            )

        return mapping[
            stage
        ]


@dataclass(frozen=True)
class SiriusCompletedStage:
    stage: SiriusStage

    input_state: MachineState

    result: Any

    final_state: MachineState


@dataclass(frozen=True)
class SiriusRunResult:
    config: SiriusRunConfig

    initial_state: MachineState

    completed_stages: tuple[
        SiriusCompletedStage,
        ...
    ]

    final_state: MachineState

    final_characterization_result: (
        Any | None
    )

    @property
    def cup6_state(
        self,
    ) -> MachineState:
        for completed in reversed(
            self.completed_stages
        ):
            if (
                completed.stage
                == SiriusStage.CUP6
            ):
                return (
                    completed.final_state
                )

        raise RuntimeError(
            "Run contains no completed Cup-6 stage"
        )


def _extract_final_state(
    result: Any,
) -> MachineState:
    if isinstance(
        result,
        MachineState,
    ):
        return result

    final_state = getattr(
        result,
        "final_state",
        None,
    )

    if not isinstance(
        final_state,
        MachineState,
    ):
        raise TypeError(
            "Stage runner must return MachineState or an object "
            "with a MachineState final_state attribute"
        )

    return final_state


def _expected_cup(
    stage: SiriusStage,
) -> int:
    if stage == SiriusStage.CUP1:
        return 1

    if stage == SiriusStage.CUP2:
        return 2

    if stage == SiriusStage.CUP3:
        return 3

    if stage == SiriusStage.CUP4:
        return 4

    if stage == SiriusStage.CUP5:
        return 5

    if stage == SiriusStage.CUP6:
        return 6

    if (
        stage
        == SiriusStage.FINAL_CHARACTERIZATION
    ):
        # Canonical final characterization:
        # 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1
        return 1

    raise ValueError(
        f"Unknown SIRIUS stage: {stage}"
    )


def _validate_run_state(
    state: MachineState,
    *,
    config: SiriusRunConfig,
) -> None:
    state.validate()

    if not math.isclose(
        float(
            state.mass_u
        ),
        float(
            config.mass_u
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Machine state ion mass differs from SIRIUS run mass: "
            f"{state.mass_u} != {config.mass_u}"
        )


def _validate_stage_output(
    state: MachineState,
    *,
    stage: SiriusStage,
    config: SiriusRunConfig,
) -> None:
    _validate_run_state(
        state,
        config=config,
    )

    expected = _expected_cup(
        stage
    )

    if state.cup != expected:
        raise ValueError(
            f"{stage.value} must finish at Cup {expected}, "
            f"got Cup {state.cup}"
        )

    if not state.parameters:
        raise ValueError(
            f"{stage.value} returned an empty machine command state"
        )


def _save_checkpoint(
    logger,
    state: MachineState,
    stage: SiriusStage,
) -> None:
    if logger is None:
        return

    logger.save_state(
        state,
        f"{stage.value}_completed",
    )


def _log_event(
    logger,
    event: str,
    payload: dict[
        str,
        Any,
    ],
) -> None:
    if logger is None:
        return

    logger.log_event(
        event,
        payload,
    )


def _run_one_stage(
    stage: SiriusStage,
    runner: StageRunner,
    current_state: MachineState,
    context: SiriusRunContext,
) -> SiriusCompletedStage:
    _log_event(
        context.logger,
        "sirius_stage_started",
        {
            "stage": (
                stage.value
            ),
            "input_state_id": (
                current_state.state_id
            ),
            "cup": (
                current_state.cup
            ),
        },
    )

    try:
        result = runner(
            current_state,
            context,
        )

        final_state = (
            _extract_final_state(
                result
            )
        )

        _validate_stage_output(
            final_state,
            stage=stage,
            config=context.config,
        )

    except Exception as exc:
        _log_event(
            context.logger,
            "sirius_stage_failed",
            {
                "stage": (
                    stage.value
                ),
                "last_state_id": (
                    current_state.state_id
                ),
                "exception_type": (
                    type(exc).__name__
                ),
                "message": (
                    str(exc)
                ),
            },
        )

        raise SiriusRunError(
            "SIRIUS failed during "
            f"{stage.value}: {exc}",
            failed_stage=stage,
            last_state=current_state,
        ) from exc

    _save_checkpoint(
        context.logger,
        final_state,
        stage,
    )

    _log_event(
        context.logger,
        "sirius_stage_completed",
        {
            "stage": (
                stage.value
            ),
            "input_state_id": (
                current_state.state_id
            ),
            "final_state_id": (
                final_state.state_id
            ),
            "cup": (
                final_state.cup
            ),
        },
    )

    return SiriusCompletedStage(
        stage=stage,
        input_state=current_state,
        result=result,
        final_state=final_state,
    )


def run_sirius(
    *,
    adapter,
    initial_state: MachineState,
    config: SiriusRunConfig,
    stages: SiriusStageRunners,
    logger=None,
) -> SiriusRunResult:
    """
    Execute one complete automatic SIRIUS tuning run.

    Canonical order:

        Cup1
        -> Cup2
        -> Cup3
        -> Cup4
        -> Cup5
        -> Cup6
        -> frozen-state final characterization

    Every successfully completed stage becomes the only permitted input
    to the next stage.

    Failure semantics:
        fail immediately
        preserve the last completed state
        do not perform blind rollback
        do not skip ahead
    """

    _validate_run_state(
        initial_state,
        config=config,
    )

    context = SiriusRunContext(
        config=config,
        adapter=adapter,
        logger=logger,
    )

    _log_event(
        logger,
        "sirius_run_started",
        {
            "initial_state_id": (
                initial_state.state_id
            ),
            "mass_u": float(
                config.mass_u
            ),
            "configuration": (
                config.to_manifest_dict()
            ),
        },
    )

    if logger is not None:
        logger.save_state(
            initial_state,
            "sirius_run_initial",
        )

    current_state = (
        initial_state
    )

    completed: list[
        SiriusCompletedStage
    ] = []

    for stage in (
        OPTIMIZATION_STAGE_SEQUENCE
    ):
        runner = stages.runner_for(
            stage
        )

        completed_stage = (
            _run_one_stage(
                stage,
                runner,
                current_state,
                context,
            )
        )

        completed.append(
            completed_stage
        )

        current_state = (
            completed_stage.final_state
        )

    final_characterization_result = (
        None
    )

    if (
        config.perform_final_characterization
    ):
        stage = (
            SiriusStage
            .FINAL_CHARACTERIZATION
        )

        try:
            runner = stages.runner_for(
                stage
            )
        except Exception as exc:
            raise SiriusRunError(
                "Final characterization requested but no runner "
                "is configured",
                failed_stage=stage,
                last_state=current_state,
            ) from exc

        completed_stage = (
            _run_one_stage(
                stage,
                runner,
                current_state,
                context,
            )
        )

        completed.append(
            completed_stage
        )

        current_state = (
            completed_stage.final_state
        )

        final_characterization_result = (
            completed_stage.result
        )

    _log_event(
        logger,
        "sirius_run_completed",
        {
            "initial_state_id": (
                initial_state.state_id
            ),
            "final_state_id": (
                current_state.state_id
            ),
            "mass_u": float(
                config.mass_u
            ),
            "completed_stages": [
                item.stage.value
                for item
                in completed
            ],
        },
    )

    return SiriusRunResult(
        config=config,
        initial_state=initial_state,
        completed_stages=tuple(
            completed
        ),
        final_state=current_state,
        final_characterization_result=(
            final_characterization_result
        ),
    )