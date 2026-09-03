from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Mapping,
)

from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.cup2_optimizer import (
    cup2_primary_rcds_production_policy,
)
from sirius.cup4_optimizer import (
    cup4_primary_rcds_production_policy,
)
from sirius.cup6_optimizer import (
    cup6_primary_rcds_production_policy,
)
from sirius.default_stages import (
    DefaultStageResources,
    build_default_stages,
    inspect_default_stage_requirements,
)
from sirius.command_cadence import (
    CommandCadenceController,
)

from sirius.hardware_guard import (
    require_complete_hardware_guard,
)

from sirius.measurement import (
    MeasurementPolicy,
)
from sirius.orchestrator import (
    SiriusRunResult,
    SiriusStage,
    run_sirius,
)
from sirius.run_config import (
    SiriusRunConfig,
)
from sirius.settling import (
    SettlingPolicy,
)
from sirius.state import (
    MachineState,
)


class SiriusRuntimeConfigurationError(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class SiriusRuntimeResources:
    """
    Concrete resources required by the currently implemented SIRIUS
    Cup-1..6 optimization chain.

    Run-generated reference states are deliberately absent here.
    They are resolved from SiriusRunContext.completed_states.
    """

    profile: Any

    tracker: Any

    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ]

    measurement_policy: MeasurementPolicy

    comparison_policy: ComparisonPolicy

    rfq_hardware: Any

    lc_candidates: Any

    rfq_matching_policy: Any

    rfq_q_policy: Any

    target_q: float

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

    def __post_init__(self) -> None:
        q = float(
            self.target_q
        )

        if (
            not math.isfinite(
                q
            )
            or q <= 0
            or q > 0.9
        ):
            raise ValueError(
                "target_q must be finite and in (0, 0.9]"
            )

        if not isinstance(
            self.measurement_policy,
            MeasurementPolicy,
        ):
            raise TypeError(
                "measurement_policy must be a MeasurementPolicy"
            )

        if not isinstance(
            self.comparison_policy,
            ComparisonPolicy,
        ):
            raise TypeError(
                "comparison_policy must be a ComparisonPolicy"
            )

    def to_default_stage_resources(
        self,
    ) -> DefaultStageResources:
        shared = {
            "profile":
                self.profile,

            "tracker":
                self.tracker,

            "settling_policies":
                self.settling_policies,

            "measurement_policy":
                self.measurement_policy,

            "comparison_policy":
                self.comparison_policy,
        }

        cup2 = {
            "primary_rcds_policy": (
                cup2_primary_rcds_production_policy()
            ),

            # Explicit runtime resources come last and
            # therefore override the default, including None.
            **dict(
                self.cup2
            ),
        }

        cup3 = {
            "rfq_hardware":
                self.rfq_hardware,

            "lc_candidates":
                self.lc_candidates,

            "rfq_matching_policy":
                self.rfq_matching_policy,

            "rfq_q_policy":
                self.rfq_q_policy,

            "target_q":
                float(
                    self.target_q
                ),

            **dict(
                self.cup3
            ),
        }

        cup4 = {
            "primary_rcds_policy": (
                cup4_primary_rcds_production_policy()
            ),

            # Explicit runtime resources come last and
            # therefore override the default, including None.
            **dict(
                self.cup4
            ),
        }

        cup6 = {
            "primary_rcds_policy": (
                cup6_primary_rcds_production_policy()
            ),

            # Explicit runtime resources come last and
            # therefore override the default, including None.
            **dict(
                self.cup6
            ),
        }

        return DefaultStageResources(
            shared=shared,

            cup1=dict(
                self.cup1
            ),

            cup2=cup2,

            cup3=cup3,

            cup4=cup4,

            cup5=dict(
                self.cup5
            ),

            cup6=cup6,

            final_characterization=dict(
                self.final_characterization
            ),
        )


@dataclass(frozen=True)
class SiriusRuntimePreflight:
    requirements: dict[
        str,
        tuple[
            str,
            ...
        ],
    ]

    missing: dict[
        str,
        tuple[
            str,
            ...
        ],
    ]

    @property
    def ready(
        self,
    ) -> bool:
        return not bool(
            self.missing
        )


def preflight_runtime_resources(
    config: SiriusRunConfig,
    resources: SiriusRuntimeResources,
) -> SiriusRuntimePreflight:
    """
    Check every real optimizer signature before any hardware action.

    This is deliberately performed before run_sirius() so an interface or
    configuration error cannot appear halfway through the beamline.
    """

    requirements = (
        inspect_default_stage_requirements()
    )

    stage_resources = (
        resources
        .to_default_stage_resources()
    )

    missing: dict[
        str,
        tuple[
            str,
            ...
        ],
    ] = {}

    for (
        stage_name,
        names,
    ) in requirements.items():
        if (
            stage_name
            == SiriusStage
            .FINAL_CHARACTERIZATION
            .value
            and not config.perform_final_characterization
        ):
            continue

        stage = SiriusStage(
            stage_name
        )

        available = (
            stage_resources.for_stage(
                stage
            )
        )

        absent = tuple(
            name
            for name
            in names
            if name not in available
        )

        if absent:
            missing[
                stage_name
            ] = absent

    return SiriusRuntimePreflight(
        requirements=(
            requirements
        ),
        missing=missing,
    )


def _configure_adapter_safety(
    adapter,
    config: SiriusRunConfig,
) -> None:
    guard = (
        config.hardware_safety
        .hardware_guard_policy
    )

    if guard is None:
        raise SiriusRuntimeConfigurationError(
            "Real SIRIUS execution requires an explicit "
            "HardwareGuardPolicy; no implicit hardware step sizes "
            "are permitted"
        )

    try:
        require_complete_hardware_guard(
            guard
        )
    except Exception as exc:
        raise SiriusRuntimeConfigurationError(
            "Real SIRIUS execution requires complete hardware-guard "
            f"coverage: {exc}"
        ) from exc

    setattr(
        adapter,
        "hardware_guard_policy",
        guard,
    )

    setattr(
        adapter,
        "readback_freshness_policy",
        config.hardware_safety
        .readback_freshness_policy,
    )

    quality_policy = (
        config.hardware_safety
        .readback_quality_policy
    )

    if quality_policy is None:
        raise SiriusRuntimeConfigurationError(
            "Real SIRIUS execution requires an explicit "
            "ReadbackQualityPolicy derived from the actual FLAVIA "
            "readback quality semantics"
        )

    if not quality_policy.strict:
        raise SiriusRuntimeConfigurationError(
            "Real SIRIUS execution requires strict readback quality "
            "checking; missing quality may not be accepted"
        )

    setattr(
        adapter,
        "readback_quality_policy",
        quality_policy,
    )

    setattr(
        adapter,
        "command_cadence_controller",
        CommandCadenceController(),
    )

    setter = getattr(
        adapter,
        "set_cup_selection_policy",
        None,
    )

    if not callable(
        setter
    ):
        raise SiriusRuntimeConfigurationError(
            "Adapter does not expose set_cup_selection_policy(); "
            "run-wide positive cup acknowledgement cannot be guaranteed"
        )

    setter(
        config.hardware_safety
        .cup_selection_policy
    )


@dataclass
class SiriusRuntime:
    """
    Concrete executable SIRIUS runtime.

    This is the boundary between configuration/resource assembly and the
    generic run state machine.
    """

    adapter: Any

    config: SiriusRunConfig

    resources: SiriusRuntimeResources

    logger: Any = None

    def preflight(
        self,
    ) -> SiriusRuntimePreflight:
        return preflight_runtime_resources(
            self.config,
            self.resources,
        )

    def run(
        self,
        initial_state: MachineState,
    ) -> SiriusRunResult:
        preflight = (
            self.preflight()
        )

        if not preflight.ready:
            raise SiriusRuntimeConfigurationError(
                "SIRIUS runtime preflight failed: "
                f"{preflight.missing}"
            )

        _configure_adapter_safety(
            self.adapter,
            self.config,
        )

        stages = (
            build_default_stages(
                self.resources
                .to_default_stage_resources()
            )
        )

        return run_sirius(
            adapter=self.adapter,
            initial_state=initial_state,
            config=self.config,
            stages=stages,
            logger=self.logger,
        )
