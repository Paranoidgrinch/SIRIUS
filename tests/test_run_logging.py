import json

import pytest

from sirius.measurement import (
    BeamCurrentSample,
    BeamMeasurement,
)
from sirius.reference import (
    SourceReference,
)
from sirius.run_logging import (
    RunLogger,
)
from sirius.state import (
    MachineState,
)


def measurement():
    return BeamMeasurement(
        mean_a=8.0e-9,
        sigma_a=0.1e-9,
        sem_a=0.02e-9,
        n=3,
        duration_s=0.5,
        relative_sem=0.0025,
        precision_threshold_a=0.08e-9,
        drift_delta_a=0.01e-9,
        stop_reason="precision_reached",
        below_noise_floor=False,
        samples=(
            BeamCurrentSample(
                current_a=7.9e-9,
                source_timestamp=1.0,
                elapsed_s=0.1,
            ),
            BeamCurrentSample(
                current_a=8.0e-9,
                source_timestamp=2.0,
                elapsed_s=0.2,
            ),
            BeamCurrentSample(
                current_a=8.1e-9,
                source_timestamp=3.0,
                elapsed_s=0.3,
            ),
        ),
    )


def machine_state():
    return MachineState(
        mass_u=60.0,
        cup=3,
        stage=3,
        role="stage_best",
        parameters={
            "extraction_voltage_v": 19000.0,
            "ion_cooler_voltage_v": 26950.0,
        },
        readbacks={
            "extraction_voltage_v": 18600.0,
            "ion_cooler_voltage_v": 26600.0,
        },
    )


def read_events(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def test_create_run_writes_manifest_and_start_event(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="test-run",
        created_at_utc="2026-08-26T12:00:00+00:00",
        git_commit="abc123",
    )

    assert logger.paths.manifest.exists()
    assert logger.paths.events.exists()
    assert logger.paths.states.exists()

    manifest = json.loads(
        logger.paths.manifest.read_text(
            encoding="utf-8"
        )
    )

    assert manifest["run_id"] == "test-run"
    assert manifest["mass_u"] == 60.0
    assert manifest["git_commit"] == "abc123"

    events = read_events(
        logger.paths.events
    )

    assert len(events) == 1
    assert events[0]["event_type"] == "run_started"
    assert events[0]["sequence"] == 1


def test_duplicate_run_directory_is_rejected(
    tmp_path,
):
    RunLogger.create(
        tmp_path,
        60.0,
        run_id="same-run",
        git_commit="abc",
    )

    with pytest.raises(
        FileExistsError
    ):
        RunLogger.create(
            tmp_path,
            60.0,
            run_id="same-run",
            git_commit="abc",
        )


def test_events_are_append_only_and_sequential(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="sequence-run",
        git_commit="abc",
    )

    logger.log_event(
        "first_test_event",
        {"value": 1},
    )

    logger.log_event(
        "second_test_event",
        {"value": 2},
    )

    events = read_events(
        logger.paths.events
    )

    assert [
        event["sequence"]
        for event in events
    ] == [
        1,
        2,
        3,
    ]

    assert [
        event["event_type"]
        for event in events
    ] == [
        "run_started",
        "first_test_event",
        "second_test_event",
    ]


def test_machine_state_is_saved_with_commands_and_readbacks(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="state-run",
        git_commit="abc",
    )

    state = machine_state()

    path = logger.save_state(
        state,
        "stage3_best",
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        data["parameters"][
            "extraction_voltage_v"
        ]
        == 19000.0
    )

    assert (
        data["readbacks"][
            "extraction_voltage_v"
        ]
        == 18600.0
    )


def test_measurement_logging_preserves_raw_samples(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="measurement-run",
        git_commit="abc",
    )

    logger.log_measurement(
        measurement(),
        cup=3,
        state_id="state-123",
        purpose="optimization",
    )

    events = read_events(
        logger.paths.events
    )

    event = events[-1]

    assert event["event_type"] == "beam_measurement"

    payload = event["payload"]

    assert payload["cup"] == 3
    assert payload["state_id"] == "state-123"

    logged_measurement = payload[
        "measurement"
    ]

    assert logged_measurement["mean_a"] == 8.0e-9

    assert len(
        logged_measurement["samples"]
    ) == 3

    assert (
        logged_measurement["samples"][0][
            "current_a"
        ]
        == 7.9e-9
    )


def test_reference_logging_preserves_state_link(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="reference-run",
        git_commit="abc",
    )

    reference = SourceReference(
        measurement=measurement(),
        state_id="cup1-reference-state",
        mass_u=60.0,
        monotonic_s=600.0,
        created_at_utc="2026-08-26T12:10:00+00:00",
    )

    logger.log_reference(
        reference
    )

    event = read_events(
        logger.paths.events
    )[-1]

    assert event["event_type"] == "cup1_reference"

    assert (
        event["payload"]["state_id"]
        == "cup1-reference-state"
    )


def test_optimizer_decision_can_be_logged(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="decision-run",
        git_commit="abc",
    )

    logger.log_optimizer_decision(
        stage=2,
        cup=2,
        parameter="lens2_voltage_v",
        decision="better",
        baseline_state_id="old",
        candidate_state_id="new",
        details={
            "delta_a": 0.5e-9,
        },
    )

    event = read_events(
        logger.paths.events
    )[-1]

    assert event["event_type"] == "optimizer_decision"

    assert event["payload"]["decision"] == "better"

    assert (
        event["payload"]["parameter"]
        == "lens2_voltage_v"
    )


def test_nonfinite_values_are_rejected(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="invalid-run",
        git_commit="abc",
    )

    with pytest.raises(
        ValueError
    ):
        logger.log_event(
            "invalid",
            {
                "value": float("nan"),
            },
        )


def test_invalid_mass_is_rejected(
    tmp_path,
):
    with pytest.raises(
        ValueError
    ):
        RunLogger.create(
            tmp_path,
            0.0,
            run_id="bad-run",
            git_commit="abc",
        )


def test_state_file_is_not_silently_overwritten(
    tmp_path,
):
    logger = RunLogger.create(
        tmp_path,
        60.0,
        run_id="overwrite-run",
        git_commit="abc",
    )

    state = machine_state()

    logger.save_state(
        state,
        "best",
    )

    with pytest.raises(
        FileExistsError
    ):
        logger.save_state(
            state,
            "best",
        )