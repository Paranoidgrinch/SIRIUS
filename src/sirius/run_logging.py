from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from sirius.measurement import BeamMeasurement
from sirius.optimizer_api import OptimizationResult
from sirius.reference import SourceReference, TransmissionResult
from sirius.state import MachineState
from sirius.transition import AppliedStateResult


RUN_LOG_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def detect_git_commit(
    repository_root: str | Path = ".",
) -> str | None:
    """
    Return the current Git commit hash when available.

    Logging must still work when SIRIUS is run outside a Git checkout,
    so Git lookup failure is intentionally non-fatal.
    """

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=Path(repository_root),
            capture_output=True,
            text=True,
            check=True,
        )

    except (
        OSError,
        subprocess.CalledProcessError,
    ):
        return None

    commit = result.stdout.strip()

    return commit or None


def _jsonable(value: Any) -> Any:
    """
    Convert SIRIUS objects into strict JSON-compatible data.

    NaN and infinite floating-point values are rejected because they are
    not valid portable JSON and would damage reproducibility.
    """

    if is_dataclass(value):
        return _jsonable(
            asdict(value)
        )

    if isinstance(value, Enum):
        return _jsonable(
            value.value
        )

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _jsonable(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Non-finite floating-point values cannot be logged"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise TypeError(
        f"Unsupported log value type: {type(value)!r}"
    )


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    events: Path
    states: Path


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    mass_u: float

    created_at_utc: str

    git_commit: str | None

    software: str = "SIRIUS"


class RunLogger:
    """
    Persistent append-only logging for one SIRIUS optimization run.

    Every event is immediately flushed to disk so useful diagnostic
    information survives an unexpected program or hardware failure.
    """

    def __init__(
        self,
        paths: RunPaths,
        manifest: RunManifest,
    ):
        self.paths = paths
        self.manifest = manifest

        self._sequence = 0

    @classmethod
    def create(
        cls,
        base_directory: str | Path,
        mass_u: float,
        *,
        repository_root: str | Path = ".",
        run_id: str | None = None,
        created_at_utc: str | None = None,
        git_commit: str | None = None,
    ) -> "RunLogger":
        if mass_u <= 0:
            raise ValueError(
                "Ion mass must be greater than zero"
            )

        if run_id is None:
            run_id = (
                f"{compact_utc_timestamp()}"
                f"_mass{mass_u:g}_"
                f"{uuid4().hex[:8]}"
            )

        root = (
            Path(base_directory)
            / run_id
        )

        if root.exists():
            raise FileExistsError(
                f"Run directory already exists: {root}"
            )

        states = root / "states"

        states.mkdir(
            parents=True,
            exist_ok=False,
        )

        paths = RunPaths(
            root=root,
            manifest=root / "manifest.json",
            events=root / "events.jsonl",
            states=states,
        )

        if git_commit is None:
            git_commit = detect_git_commit(
                repository_root
            )

        manifest = RunManifest(
            schema_version=RUN_LOG_SCHEMA_VERSION,
            run_id=run_id,
            mass_u=float(mass_u),
            created_at_utc=(
                created_at_utc
                if created_at_utc is not None
                else utc_now_iso()
            ),
            git_commit=git_commit,
        )

        logger = cls(
            paths=paths,
            manifest=manifest,
        )

        logger._write_json(
            paths.manifest,
            manifest,
        )

        # Create the event stream immediately.
        paths.events.touch(
            exist_ok=False,
        )

        logger.log_event(
            "run_started",
            {
                "mass_u": float(mass_u),
                "git_commit": git_commit,
            },
        )

        return logger

    def _write_json(
        self,
        path: Path,
        value: Any,
    ) -> None:
        payload = _jsonable(
            value
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")

    def log_event(
        self,
        event_type: str,
        payload: Any,
        *,
        timestamp_utc: str | None = None,
    ) -> dict[str, Any]:
        if not event_type:
            raise ValueError(
                "event_type must not be empty"
            )

        self._sequence += 1

        event = {
            "schema_version": RUN_LOG_SCHEMA_VERSION,
            "sequence": self._sequence,
            "event_id": str(uuid4()),
            "timestamp_utc": (
                timestamp_utc
                if timestamp_utc is not None
                else utc_now_iso()
            ),
            "event_type": event_type,
            "payload": _jsonable(payload),
        }

        with self.paths.events.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                event,
                handle,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()

        return event

    def save_state(
        self,
        state: MachineState,
        label: str,
    ) -> Path:
        """
        Save a complete reproducible machine state and emit an event
        linking the run timeline to the state file.
        """

        state.validate()

        safe_label = "".join(
            character
            if (
                character.isalnum()
                or character in ("-", "_")
            )
            else "_"
            for character in label
        )

        if not safe_label:
            raise ValueError(
                "State label must contain a usable character"
            )

        filename = (
            f"{safe_label}_"
            f"{state.state_id}.json"
        )

        path = (
            self.paths.states
            / filename
        )

        if path.exists():
            raise FileExistsError(
                f"State file already exists: {path}"
            )

        state.to_json(
            path
        )

        self.log_event(
            "machine_state_saved",
            {
                "label": label,
                "state_id": state.state_id,
                "role": state.role,
                "cup": state.cup,
                "stage": state.stage,
                "path": str(
                    path.relative_to(
                        self.paths.root
                    )
                ),
            },
        )

        return path

    def log_measurement(
        self,
        measurement: BeamMeasurement,
        *,
        cup: int,
        state_id: str,
        purpose: str,
    ) -> dict[str, Any]:
        if not 1 <= cup <= 6:
            raise ValueError(
                "Cup must be between 1 and 6"
            )

        return self.log_event(
            "beam_measurement",
            {
                "cup": cup,
                "state_id": state_id,
                "purpose": purpose,
                "measurement": measurement,
            },
        )

    def log_reference(
        self,
        reference: SourceReference,
    ) -> dict[str, Any]:
        return self.log_event(
            "cup1_reference",
            reference,
        )

    def log_transmission(
        self,
        transmission: TransmissionResult,
    ) -> dict[str, Any]:
        return self.log_event(
            "transmission",
            transmission,
        )

    def log_state_transition(
        self,
        result: AppliedStateResult,
    ) -> dict[str, Any]:
        return self.log_event(
            "state_transition",
            {
                "source_state_id": (
                    result.plan.source_state_id
                ),
                "target_state_id": (
                    result.plan.target_state_id
                ),
                "changed_parameters": (
                    result.plan.changed_parameters
                ),
                "changes": result.plan.changes,
                "settling_results": (
                    result.settling_results
                ),
                "selected_cup": (
                    result.selected_cup
                ),
                "observed_readbacks": (
                    result.observed_state.readbacks
                ),
            },
        )

    def log_optimizer_trace(
        self,
        result: OptimizationResult,
        *,
        stage: int,
        cup: int,
    ) -> tuple[
        dict[str, Any],
        ...
    ]:
        """Persist a completed optimizer trace event-by-event."""

        trace = result.metadata.get(
            "trace"
        )

        if trace is None:
            raise ValueError(
                "Optimization result does not contain a trace"
            )

        if not isinstance(
            trace,
            (list, tuple),
        ):
            raise TypeError(
                "Optimization trace must be a list or tuple"
            )

        prepared_payloads: list[
            dict[str, Any]
        ] = []

        for trace_index, trace_event in enumerate(
            trace
        ):
            if not isinstance(
                trace_event,
                dict,
            ):
                raise TypeError(
                    "Optimization trace events must be dictionaries"
                )

            trace_event_type = trace_event.get(
                "event_type"
            )

            if (
                not isinstance(
                    trace_event_type,
                    str,
                )
                or not trace_event_type
            ):
                raise ValueError(
                    "Optimization trace event_type must be a non-empty string"
                )

            details = {
                str(key): value
                for key, value
                in trace_event.items()
                if key != "event_type"
            }

            payload = {
                "stage": stage,
                "cup": cup,
                "optimizer_name": result.optimizer_name,
                "optimizer_version": result.optimizer_version,
                "trace_index": trace_index,
                "trace_event_type": trace_event_type,
                "details": details,
            }

            # Validate the complete trace before the first
            # append-only event is written.
            _jsonable(
                payload
            )

            prepared_payloads.append(
                payload
            )

        return tuple(
            self.log_event(
                "optimizer_trace_event",
                payload,
            )
            for payload
            in prepared_payloads
        )

    def log_optimizer_decision(
        self,
        *,
        stage: int,
        cup: int,
        parameter: str | None,
        decision: str,
        baseline_state_id: str | None,
        candidate_state_id: str | None,
        details: Any = None,
    ) -> dict[str, Any]:
        """
        Generic optimizer-decision event.

        The optimizer itself does not exist yet, but defining this event
        now ensures its future decisions are reproducible from day one.
        """

        return self.log_event(
            "optimizer_decision",
            {
                "stage": stage,
                "cup": cup,
                "parameter": parameter,
                "decision": decision,
                "baseline_state_id": baseline_state_id,
                "candidate_state_id": candidate_state_id,
                "details": details,
            },
        )
