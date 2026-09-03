from types import (
    SimpleNamespace,
)

import pytest

import sirius.default_stages as default_module
import sirius.transition as transition_module
from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.cup_ack import (
    CupSelectionPolicy,
    CupSelectionResult,
)
from sirius.default_stages import (
    DefaultStageResources,
)
from sirius.hardware_safety import (
    HardwareSafetyConfig,
)
from sirius.measurement import (
    MeasurementPolicy,
)
from sirius.orchestrator import (
    SiriusRunContext,
    SiriusStage,
)
from sirius.run_config import (
    SiriusRunConfig,
)
from sirius.runtime import (
    SiriusRuntimeResources,
    preflight_runtime_resources,
)
from sirius.state import (
    MachineState,
)


def state(
    cup,
):
    return MachineState(
        mass_u=60.0,
        cup=cup,
        stage=cup,
        parameters={
            "sputter_voltage_v":
                8000.0,
        },
    )


def config():
    return SiriusRunConfig(
        mass_u=60.0,
        hardware_safety=(
            HardwareSafetyConfig(
                cooler_end_max_step_v=100.0,
                qpt_max_step_v=125.0,
            )
        ),
    )


def test_cup1_reference_for_downstream_stage_comes_from_history():
    ctx = SiriusRunContext(
        config=config(),
        adapter=object(),
    )

    cup1 = state(
        1
    )

    current = state(
        3
    )

    ctx.record_completed_state(
        SiriusStage.CUP1,
        cup1,
    )

    supplied, value = (
        default_module._automatic_value(
            "cup1_reference_state",
            stage=SiriusStage.CUP3,
            state=current,
            context=ctx,
        )
    )

    assert supplied is True

    assert value is cup1

    assert value is not current


def test_cup2_input_named_cup1_reference_state_remains_current_state():
    ctx = SiriusRunContext(
        config=config(),
        adapter=object(),
    )

    current = state(
        1
    )

    supplied, value = (
        default_module._automatic_value(
            "cup1_reference_state",
            stage=SiriusStage.CUP2,
            state=current,
            context=ctx,
        )
    )

    assert supplied is True

    assert value is current


def test_requirement_inspection_no_longer_exposes_dynamic_cup1_reference():
    requirements = (
        default_module
        .inspect_default_stage_requirements()
    )

    for stage in (
        "cup3",
        "cup4",
        "cup5",
        "cup6",
    ):
        assert (
            "cup1_reference_state"
            not in requirements[
                stage
            ]
        )


def test_transition_inherits_adapter_level_cup_policy(
    monkeypatch,
):
    policy = CupSelectionPolicy(
        timeout_s=7.0,
        poll_interval_s=0.1,
        minimum_wait_s=0.2,
        consecutive_confirmations=3,
    )

    class Adapter:
        cup_selection_policy = (
            policy
        )

        def select_cup(
            self,
            cup,
        ):
            pass

        def read_selected_cup(
            self,
        ):
            return 2

    captured = {}

    def fake_ack(
        *,
        select_cup,
        read_selected_cup,
        target_cup,
        policy,
        **kwargs,
    ):
        captured[
            "policy"
        ] = policy

        return CupSelectionResult(
            requested_cup=2,
            confirmed_cup=2,
            elapsed_s=0.2,
            confirmation_count=3,
            samples=(),
        )

    monkeypatch.setattr(
        transition_module,
        "select_cup_and_wait",
        fake_ack,
    )

    monkeypatch.setattr(
        transition_module,
        "capture_readbacks",
        lambda adapter, state:
            state,
    )

    current = state(
        1
    )

    target = state(
        2
    )

    transition_module.apply_state(
        Adapter(),
        current=current,
        target=target,
        settling_policies={},
    )

    assert (
        captured[
            "policy"
        ]
        is policy
    )


def test_runtime_resource_preflight_matches_current_optimizer_signatures():
    resources = SiriusRuntimeResources(
        profile=object(),
        tracker=object(),
        settling_policies={},
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        rfq_hardware=object(),
        lc_candidates=(),
        rfq_matching_policy=object(),
        rfq_q_policy=object(),
        target_q=0.45,
    )

    result = (
        preflight_runtime_resources(
            config(),
            resources,
        )
    )

    assert result.ready is True

    assert result.missing == {}


def test_runtime_target_q_respects_rfq_stability_limit():
    with pytest.raises(
        ValueError,
        match="0.9",
    ):
        SiriusRuntimeResources(
            profile=object(),
            tracker=object(),
            settling_policies={},
            measurement_policy=(
                MeasurementPolicy()
            ),
            comparison_policy=(
                ComparisonPolicy()
            ),
            rfq_hardware=object(),
            lc_candidates=(),
            rfq_matching_policy=object(),
            rfq_q_policy=object(),
            target_q=0.95,
        )


def test_runtime_enables_bounded_cup2_rcds_by_default():
    resources = SiriusRuntimeResources(
        profile=object(),
        tracker=object(),
        settling_policies={},
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        rfq_hardware=object(),
        lc_candidates=(),
        rfq_matching_policy=object(),
        rfq_q_policy=object(),
        target_q=0.45,
    )

    stage_resources = (
        resources
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP2
        )
    )

    policy = stage_resources[
        "primary_rcds_policy"
    ]

    assert (
        policy.max_iterations
        == 2
    )

    assert (
        policy.max_evaluations
        == 73
    )

    assert (
        policy.line_samples
        == 7
    )

    assert (
        policy.reuse_cached_evaluations
        is False
    )


def test_runtime_allows_explicit_legacy_cup2_override():
    resources = SiriusRuntimeResources(
        profile=object(),
        tracker=object(),
        settling_policies={},
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        rfq_hardware=object(),
        lc_candidates=(),
        rfq_matching_policy=object(),
        rfq_q_policy=object(),
        target_q=0.45,
        cup2={
            "primary_rcds_policy":
                None,
            "runtime_test_marker":
                "legacy",
        },
    )

    stage_resources = (
        resources
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP2
        )
    )

    assert (
        stage_resources[
            "primary_rcds_policy"
        ]
        is None
    )

    assert (
        stage_resources[
            "runtime_test_marker"
        ]
        == "legacy"
    )


def test_runtime_cup2_rcds_policy_reaches_real_optimizer_call_contract():
    def make_resources(
        *,
        cup2=None,
    ):
        return SiriusRuntimeResources(
            profile=object(),
            tracker=object(),
            settling_policies={},
            measurement_policy=(
                MeasurementPolicy()
            ),
            comparison_policy=(
                ComparisonPolicy()
            ),
            rfq_hardware=object(),
            lc_candidates=(),
            rfq_matching_policy=object(),
            rfq_q_policy=object(),
            target_q=0.45,
            cup2=(
                {}
                if cup2 is None
                else cup2
            ),
        )

    current = state(
        1
    )

    ctx = SiriusRunContext(
        config=config(),
        adapter=object(),
        logger=None,
    )

    real_optimize_cup2 = (
        default_module
        ._load_stage_function(
            SiriusStage.CUP2
        )
    )

    assert (
        real_optimize_cup2.__name__
        == "optimize_cup2"
    )

    # --------------------------------------------------------
    # Runtime default:
    # bounded, cache-free live RCDS policy must reach the
    # REAL optimize_cup2 signature.
    # --------------------------------------------------------

    default_resources = (
        make_resources()
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP2
        )
    )

    default_policy = (
        default_resources[
            "primary_rcds_policy"
        ]
    )

    args, kwargs = (
        default_module
        ._prepare_call(
            real_optimize_cup2,
            stage=SiriusStage.CUP2,
            state=current,
            context=ctx,
            resources=(
                default_resources
            ),
        )
    )

    assert args == []

    assert (
        kwargs[
            "adapter"
        ]
        is ctx.adapter
    )

    assert (
        kwargs[
            "current_state"
        ]
        is current
    )

    # For Cup 2 the saved Cup-1 state is the incoming current
    # state supplied automatically by the stage assembler.
    assert (
        kwargs[
            "cup1_reference_state"
        ]
        is current
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ]
        is default_policy
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].max_iterations
        == 2
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].max_evaluations
        == 73
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].reuse_cached_evaluations
        is False
    )

    # --------------------------------------------------------
    # Explicit legacy override:
    # None must also reach the REAL optimize_cup2 signature,
    # causing its existing legacy branch to remain selectable.
    # --------------------------------------------------------

    legacy_resources = (
        make_resources(
            cup2={
                "primary_rcds_policy":
                    None,
            }
        )
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP2
        )
    )

    legacy_args, legacy_kwargs = (
        default_module
        ._prepare_call(
            real_optimize_cup2,
            stage=SiriusStage.CUP2,
            state=current,
            context=ctx,
            resources=(
                legacy_resources
            ),
        )
    )

    assert legacy_args == []

    assert (
        legacy_kwargs[
            "primary_rcds_policy"
        ]
        is None
    )


def test_runtime_enables_bounded_cup4_rcds_by_default():
    resources = SiriusRuntimeResources(
        profile=object(),
        tracker=object(),
        settling_policies={},
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        rfq_hardware=object(),
        lc_candidates=(),
        rfq_matching_policy=object(),
        rfq_q_policy=object(),
        target_q=0.45,
    )

    stage_resources = (
        resources
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP4
        )
    )

    policy = stage_resources[
        "primary_rcds_policy"
    ]

    assert (
        policy.max_iterations
        == 2
    )

    assert (
        policy.max_evaluations
        == 91
    )

    assert (
        policy.line_samples
        == 7
    )

    assert (
        policy.line_half_width
        == pytest.approx(
            0.35
        )
    )

    assert (
        policy.parabolic_refinement
        is True
    )

    assert (
        policy.reuse_cached_evaluations
        is False
    )


def test_runtime_allows_explicit_legacy_cup4_override():
    resources = SiriusRuntimeResources(
        profile=object(),
        tracker=object(),
        settling_policies={},
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        rfq_hardware=object(),
        lc_candidates=(),
        rfq_matching_policy=object(),
        rfq_q_policy=object(),
        target_q=0.45,
        cup4={
            "primary_rcds_policy":
                None,
            "runtime_test_marker":
                "legacy",
        },
    )

    stage_resources = (
        resources
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP4
        )
    )

    assert (
        stage_resources[
            "primary_rcds_policy"
        ]
        is None
    )

    assert (
        stage_resources[
            "runtime_test_marker"
        ]
        == "legacy"
    )


def test_runtime_cup4_rcds_policy_reaches_real_optimizer_call_contract():
    def make_resources(
        *,
        cup4=None,
    ):
        return SiriusRuntimeResources(
            profile=object(),
            tracker=object(),
            settling_policies={},
            measurement_policy=(
                MeasurementPolicy()
            ),
            comparison_policy=(
                ComparisonPolicy()
            ),
            rfq_hardware=object(),
            lc_candidates=(),
            rfq_matching_policy=object(),
            rfq_q_policy=object(),
            target_q=0.45,
            cup4=(
                {}
                if cup4 is None
                else cup4
            ),
        )

    cup1 = state(
        1
    )

    # Cup 4 is entered from the completed Cup-3 state.
    current = state(
        3
    )

    ctx = SiriusRunContext(
        config=config(),
        adapter=object(),
        logger=None,
    )

    # Cup-4 source-reference maintenance requires the actual
    # completed Cup-1 reference from run history.
    ctx.record_completed_state(
        SiriusStage.CUP1,
        cup1,
    )

    real_optimize_cup4 = (
        default_module
        ._load_stage_function(
            SiriusStage.CUP4
        )
    )

    assert (
        real_optimize_cup4.__name__
        == "optimize_cup4"
    )

    # --------------------------------------------------------
    # Runtime default:
    #
    # bounded, cache-free live RCDS policy must reach the REAL
    # optimize_cup4 signature.
    #
    # At the same time:
    #   current_state          <- incoming Cup-3 state
    #   cup3_reference_state   <- same incoming Cup-3 state
    #   cup1_reference_state   <- completed Cup-1 history
    # --------------------------------------------------------

    default_resources = (
        make_resources()
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP4
        )
    )

    default_policy = (
        default_resources[
            "primary_rcds_policy"
        ]
    )

    args, kwargs = (
        default_module
        ._prepare_call(
            real_optimize_cup4,
            stage=(
                SiriusStage.CUP4
            ),
            state=current,
            context=ctx,
            resources=(
                default_resources
            ),
        )
    )

    assert (
        args
        == []
    )

    assert (
        kwargs[
            "adapter"
        ]
        is ctx.adapter
    )

    assert (
        kwargs[
            "current_state"
        ]
        is current
    )

    assert (
        kwargs[
            "cup3_reference_state"
        ]
        is current
    )

    assert (
        kwargs[
            "cup1_reference_state"
        ]
        is cup1
    )

    assert (
        kwargs[
            "cup1_reference_state"
        ]
        is not current
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ]
        is default_policy
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].max_iterations
        == 2
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].max_evaluations
        == 91
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].line_samples
        == 7
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].line_half_width
        == pytest.approx(
            0.35
        )
    )

    assert (
        kwargs[
            "primary_rcds_policy"
        ].reuse_cached_evaluations
        is False
    )

    # --------------------------------------------------------
    # Explicit legacy override:
    #
    # None must reach the SAME real optimize_cup4 signature.
    # Dynamic Cup-1 and incoming Cup-3 bindings must remain
    # identical.
    # --------------------------------------------------------

    legacy_resources = (
        make_resources(
            cup4={
                "primary_rcds_policy":
                    None,
            }
        )
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP4
        )
    )

    (
        legacy_args,
        legacy_kwargs,
    ) = (
        default_module
        ._prepare_call(
            real_optimize_cup4,
            stage=(
                SiriusStage.CUP4
            ),
            state=current,
            context=ctx,
            resources=(
                legacy_resources
            ),
        )
    )

    assert (
        legacy_args
        == []
    )

    assert (
        legacy_kwargs[
            "adapter"
        ]
        is ctx.adapter
    )

    assert (
        legacy_kwargs[
            "current_state"
        ]
        is current
    )

    assert (
        legacy_kwargs[
            "cup3_reference_state"
        ]
        is current
    )

    assert (
        legacy_kwargs[
            "cup1_reference_state"
        ]
        is cup1
    )

    assert (
        legacy_kwargs[
            "primary_rcds_policy"
        ]
        is None
    )


def test_runtime_enables_bounded_cup6_rcds_by_default():
    resources = SiriusRuntimeResources(
        profile=object(),
        tracker=object(),
        settling_policies={},
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        rfq_hardware=object(),
        lc_candidates=(),
        rfq_matching_policy=object(),
        rfq_q_policy=object(),
        target_q=0.45,
    )

    stage_resources = (
        resources
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP6
        )
    )

    policy = stage_resources[
        "primary_rcds_policy"
    ]

    assert (
        policy.max_iterations
        == 2
    )

    assert (
        policy.max_evaluations
        == 73
    )

    assert (
        policy.line_samples
        == 7
    )

    assert (
        policy.line_half_width
        == pytest.approx(
            0.35
        )
    )

    assert (
        policy.parabolic_refinement
        is True
    )

    assert (
        policy.reuse_cached_evaluations
        is False
    )


def test_runtime_allows_explicit_legacy_cup6_override():
    resources = SiriusRuntimeResources(
        profile=object(),
        tracker=object(),
        settling_policies={},
        measurement_policy=(
            MeasurementPolicy()
        ),
        comparison_policy=(
            ComparisonPolicy()
        ),
        rfq_hardware=object(),
        lc_candidates=(),
        rfq_matching_policy=object(),
        rfq_q_policy=object(),
        target_q=0.45,
        cup6={
            "primary_rcds_policy":
                None,
            "runtime_test_marker":
                "legacy",
        },
    )

    stage_resources = (
        resources
        .to_default_stage_resources()
        .for_stage(
            SiriusStage.CUP6
        )
    )

    assert (
        stage_resources[
            "primary_rcds_policy"
        ]
        is None
    )

    assert (
        stage_resources[
            "runtime_test_marker"
        ]
        == "legacy"
    )
