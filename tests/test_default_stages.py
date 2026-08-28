from dataclasses import (
    dataclass,
)
from types import (
    SimpleNamespace,
)

import pytest

import sirius.default_stages as module
from sirius.coupled_transition import (
    cooler_end_transition_policy,
)
from sirius.cup3_coordinates import (
    EndElectrodeCoordinatePolicy,
)
from sirius.hardware_safety import (
    HardwareSafetyConfig,
)
from sirius.orchestrator import (
    SiriusRunContext,
    SiriusStage,
)
from sirius.qpt_scan2d import (
    QPT2DScanPolicy,
)
from sirius.run_config import (
    SiriusRunConfig,
)
from sirius.state import (
    MachineState,
)


def state():
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        parameters={
            "deceleration_voltage_v": 1000.0,
            "acceleration_voltage_v": 1000.0,
            "quadrupole1_voltage_v": 2000.0,
            "quadrupole2_voltage_v": 3000.0,
            "quadrupole3_voltage_v": 2000.0,
        },
    )


def context():
    config = SiriusRunConfig(
        mass_u=60.0,
        hardware_safety=(
            HardwareSafetyConfig(
                cooler_end_max_step_v=100.0,
                qpt_max_step_v=125.0,
            )
        ),
    )

    return SiriusRunContext(
        config=config,
        adapter=object(),
        logger=None,
    )


def test_all_real_stage_functions_resolve():
    for stage in (
        module.STAGE_FUNCTION_SPECS
    ):
        function = (
            module._load_stage_function(
                stage
            )
        )

        assert callable(
            function
        )


def test_automatic_adapter_and_state_injection():
    ctx = context()
    current = state()

    captured = {}

    def fake(
        adapter,
        current_state,
        *,
        policy,
        logger=None,
    ):
        captured[
            "adapter"
        ] = adapter

        captured[
            "state"
        ] = current_state

        captured[
            "policy"
        ] = policy

        captured[
            "logger"
        ] = logger

        return current_state

    policy = object()

    result = module._invoke_stage(
        fake,
        stage=SiriusStage.CUP3,
        state=current,
        context=ctx,
        resources={
            "policy": policy,
        },
    )

    assert result is current

    assert (
        captured[
            "adapter"
        ]
        is ctx.adapter
    )

    assert (
        captured[
            "state"
        ]
        is current
    )

    assert (
        captured[
            "policy"
        ]
        is policy
    )


def test_missing_required_resource_fails_before_function_call():
    ctx = context()

    called = {
        "value": False
    }

    def fake(
        adapter,
        current_state,
        required_physics_model,
    ):
        called[
            "value"
        ] = True

    with pytest.raises(
        module.DefaultStageWiringError,
        match="required_physics_model",
    ):
        module._invoke_stage(
            fake,
            stage=SiriusStage.CUP3,
            state=state(),
            context=ctx,
            resources={},
        )

    assert (
        called[
            "value"
        ]
        is False
    )


def test_stage_specific_resources_override_shared_resources():
    resources = (
        module.DefaultStageResources(
            shared={
                "policy":
                    "shared",
            },
            cup3={
                "policy":
                    "cup3",
            },
        )
    )

    assert (
        resources.for_stage(
            SiriusStage.CUP3
        )[
            "policy"
        ]
        == "cup3"
    )


def test_end_electrode_policy_is_bound_to_run_safety():
    ctx = context()

    original = (
        EndElectrodeCoordinatePolicy()
    )

    bound = (
        module._bind_hardware_safety(
            original,
            ctx,
        )
    )

    assert (
        original.transition_policy
        is None
    )

    assert (
        bound.transition_policy
        is not None
    )

    assert (
        bound.transition_policy
        .max_step_by_parameter[
            "deceleration_voltage_v"
        ]
        == pytest.approx(
            100.0
        )
    )


def test_qpt_policy_is_bound_to_run_safety():
    ctx = context()

    original = (
        QPT2DScanPolicy()
    )

    bound = (
        module._bind_hardware_safety(
            original,
            ctx,
        )
    )

    assert (
        original.transition_policy
        is None
    )

    assert (
        bound.transition_policy
        is not None
    )

    for name in (
        "quadrupole1_voltage_v",
        "quadrupole2_voltage_v",
        "quadrupole3_voltage_v",
    ):
        assert (
            bound.transition_policy
            .max_step_by_parameter[
                name
            ]
            == pytest.approx(
                125.0
            )
        )


def test_nested_stage_policy_is_bound_recursively():
    @dataclass(frozen=True)
    class FakeCup3Policy:
        end_electrodes: EndElectrodeCoordinatePolicy

        untouched: str = "keep"

    original = FakeCup3Policy(
        end_electrodes=(
            EndElectrodeCoordinatePolicy()
        )
    )

    bound = (
        module._bind_hardware_safety(
            original,
            context(),
        )
    )

    assert (
        bound is not original
    )

    assert (
        bound.untouched
        == "keep"
    )

    assert (
        bound.end_electrodes
        .transition_policy
        is not None
    )


def test_unrelated_dataclass_is_not_recreated():
    @dataclass(frozen=True)
    class Unrelated:
        value: int

    original = Unrelated(
        42
    )

    bound = (
        module._bind_hardware_safety(
            original,
            context(),
        )
    )

    assert (
        bound is original
    )


def test_final_characterization_gets_cup6_state_alias():
    current = MachineState(
        mass_u=60.0,
        cup=6,
        stage=6,
        parameters={
            "sputter_voltage_v":
                8000.0,
        },
    )

    captured = {}

    def fake(
        adapter,
        final_cup6_state,
    ):
        captured[
            "state"
        ] = final_cup6_state

        return SimpleNamespace(
            final_state=current
        )

    module._invoke_stage(
        fake,
        stage=(
            SiriusStage
            .FINAL_CHARACTERIZATION
        ),
        state=current,
        context=context(),
        resources={},
    )

    assert (
        captured[
            "state"
        ]
        is current
    )


def test_requirement_inspection_does_not_execute_stage_functions():
    requirements = (
        module.inspect_default_stage_requirements()
    )

    assert set(
        requirements
    ) == {
        "cup1",
        "cup2",
        "cup3",
        "cup4",
        "cup5",
        "cup6",
        "final_characterization",
    }

    for required in (
        requirements.values()
    ):
        assert isinstance(
            required,
            tuple,
        )