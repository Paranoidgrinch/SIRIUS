from types import SimpleNamespace

import pytest

from sirius.hardware_safety import (
    HardwareSafetyConfig,
)
from sirius.orchestrator import (
    SiriusRunError,
    SiriusStage,
    SiriusStageRunners,
    run_sirius,
)
from sirius.run_config import (
    SiriusRunConfig,
)
from sirius.state import (
    MachineState,
)


def state(
    cup,
    *,
    value=1000.0,
    mass_u=60.0,
):
    return MachineState(
        mass_u=mass_u,
        cup=cup,
        stage=(
            cup
            if cup in range(
                1,
                7,
            )
            else None
        ),
        parameters={
            "sputter_voltage_v":
                value,
        },
    )


def config(
    *,
    final=True,
):
    return SiriusRunConfig(
        mass_u=60.0,
        hardware_safety=(
            HardwareSafetyConfig(
                cooler_end_max_step_v=100.0,
                qpt_max_step_v=125.0,
            )
        ),
        perform_final_characterization=(
            final
        ),
    )


def staged_runner(
    name,
    output_cup,
    calls,
):
    def run(
        current_state,
        context,
    ):
        calls.append(
            (
                name,
                current_state.cup,
                context,
            )
        )

        result = state(
            output_cup,
            value=(
                current_state.parameters[
                    "sputter_voltage_v"
                ]
                + 1.0
            ),
        )

        return SimpleNamespace(
            final_state=result
        )

    return run


def complete_stages(
    calls,
):
    return SiriusStageRunners(
        cup1=staged_runner(
            "cup1",
            1,
            calls,
        ),
        cup2=staged_runner(
            "cup2",
            2,
            calls,
        ),
        cup3=staged_runner(
            "cup3",
            3,
            calls,
        ),
        cup4=staged_runner(
            "cup4",
            4,
            calls,
        ),
        cup5=staged_runner(
            "cup5",
            5,
            calls,
        ),
        cup6=staged_runner(
            "cup6",
            6,
            calls,
        ),
        final_characterization=(
            staged_runner(
                "final_characterization",
                1,
                calls,
            )
        ),
    )


def test_complete_run_uses_canonical_order():
    calls = []

    result = run_sirius(
        adapter=object(),
        initial_state=state(
            None
        ),
        config=config(),
        stages=complete_stages(
            calls
        ),
    )

    assert [
        item[
            0
        ]
        for item
        in calls
    ] == [
        "cup1",
        "cup2",
        "cup3",
        "cup4",
        "cup5",
        "cup6",
        "final_characterization",
    ]

    assert [
        stage.stage
        for stage
        in result.completed_stages
    ] == [
        SiriusStage.CUP1,
        SiriusStage.CUP2,
        SiriusStage.CUP3,
        SiriusStage.CUP4,
        SiriusStage.CUP5,
        SiriusStage.CUP6,
        SiriusStage.FINAL_CHARACTERIZATION,
    ]


def test_each_stage_receives_previous_completed_state():
    calls = []

    run_sirius(
        adapter=object(),
        initial_state=state(
            None
        ),
        config=config(),
        stages=complete_stages(
            calls
        ),
    )

    assert [
        item[
            1
        ]
        for item
        in calls
    ] == [
        None,
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_final_characterization_finishes_on_cup1():
    calls = []

    result = run_sirius(
        adapter=object(),
        initial_state=state(
            None
        ),
        config=config(),
        stages=complete_stages(
            calls
        ),
    )

    assert (
        result.final_state.cup
        == 1
    )

    assert (
        result.cup6_state.cup
        == 6
    )


def test_final_characterization_can_be_disabled():
    calls = []

    result = run_sirius(
        adapter=object(),
        initial_state=state(
            None
        ),
        config=config(
            final=False
        ),
        stages=complete_stages(
            calls
        ),
    )

    assert [
        item[
            0
        ]
        for item
        in calls
    ] == [
        "cup1",
        "cup2",
        "cup3",
        "cup4",
        "cup5",
        "cup6",
    ]

    assert (
        result.final_state.cup
        == 6
    )

    assert (
        result.final_characterization_result
        is None
    )


def test_wrong_output_cup_aborts_run():
    calls = []

    stages = complete_stages(
        calls
    )

    stages = SiriusStageRunners(
        cup1=stages.cup1,
        cup2=staged_runner(
            "bad_cup2",
            3,
            calls,
        ),
        cup3=stages.cup3,
        cup4=stages.cup4,
        cup5=stages.cup5,
        cup6=stages.cup6,
        final_characterization=(
            stages.final_characterization
        ),
    )

    with pytest.raises(
        SiriusRunError
    ) as captured:
        run_sirius(
            adapter=object(),
            initial_state=state(
                None
            ),
            config=config(),
            stages=stages,
        )

    error = captured.value

    assert (
        error.failed_stage
        == SiriusStage.CUP2
    )

    # Cup 1 was the last positively completed stage.
    assert (
        error.last_state.cup
        == 1
    )


def test_mass_change_aborts_run():
    calls = []

    stages = complete_stages(
        calls
    )

    def bad_cup3(
        current,
        context,
    ):
        return state(
            3,
            mass_u=61.0,
        )

    stages = SiriusStageRunners(
        cup1=stages.cup1,
        cup2=stages.cup2,
        cup3=bad_cup3,
        cup4=stages.cup4,
        cup5=stages.cup5,
        cup6=stages.cup6,
        final_characterization=(
            stages.final_characterization
        ),
    )

    with pytest.raises(
        SiriusRunError
    ) as captured:
        run_sirius(
            adapter=object(),
            initial_state=state(
                None
            ),
            config=config(),
            stages=stages,
        )

    assert (
        captured.value.failed_stage
        == SiriusStage.CUP3
    )

    assert (
        captured.value.last_state.cup
        == 2
    )


def test_stage_exception_preserves_last_completed_state():
    calls = []

    stages = complete_stages(
        calls
    )

    def failing_cup4(
        current,
        context,
    ):
        raise RuntimeError(
            "simulated beam loss"
        )

    stages = SiriusStageRunners(
        cup1=stages.cup1,
        cup2=stages.cup2,
        cup3=stages.cup3,
        cup4=failing_cup4,
        cup5=stages.cup5,
        cup6=stages.cup6,
        final_characterization=(
            stages.final_characterization
        ),
    )

    with pytest.raises(
        SiriusRunError
    ) as captured:
        run_sirius(
            adapter=object(),
            initial_state=state(
                None
            ),
            config=config(),
            stages=stages,
        )

    error = captured.value

    assert (
        error.failed_stage
        == SiriusStage.CUP4
    )

    assert (
        error.last_state.cup
        == 3
    )

    assert isinstance(
        error.__cause__,
        RuntimeError,
    )


def test_context_exposes_central_hardware_safety():
    seen = []

    stages = complete_stages(
        seen
    )

    run_sirius(
        adapter=object(),
        initial_state=state(
            None
        ),
        config=config(),
        stages=stages,
    )

    first_context = (
        seen[
            0
        ][
            2
        ]
    )

    assert (
        first_context.hardware_safety
        is first_context.config.hardware_safety
    )

    assert (
        first_context.cup_selection_policy
        is first_context.hardware_safety.cup_selection_policy
    )


def test_initial_mass_must_match_config():
    with pytest.raises(
        ValueError,
        match="ion mass",
    ):
        run_sirius(
            adapter=object(),
            initial_state=state(
                None,
                mass_u=61.0,
            ),
            config=config(),
            stages=complete_stages(
                []
            ),
        )


def test_final_characterization_runner_is_required_when_enabled():
    calls = []

    base = complete_stages(
        calls
    )

    stages = SiriusStageRunners(
        cup1=base.cup1,
        cup2=base.cup2,
        cup3=base.cup3,
        cup4=base.cup4,
        cup5=base.cup5,
        cup6=base.cup6,
        final_characterization=None,
    )

    with pytest.raises(
        SiriusRunError
    ) as captured:
        run_sirius(
            adapter=object(),
            initial_state=state(
                None
            ),
            config=config(),
            stages=stages,
        )

    assert (
        captured.value.failed_stage
        == SiriusStage.FINAL_CHARACTERIZATION
    )

    assert (
        captured.value.last_state.cup
        == 6
    )