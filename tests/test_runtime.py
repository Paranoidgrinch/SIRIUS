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