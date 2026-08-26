from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sirius.parameters import PARAMETERS


STATE_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RFQState:
    frequency_hz: float | None = None
    generator_amplitude_vpp: float | None = None

    inductance_uh: float | None = None
    capacitance_pf: float | None = None

    rfq_vpp_measured: float | None = None

    q_target: float | None = None
    q_nominal: float | None = None
    q_measured: float | None = None


@dataclass
class MachineState:
    mass_u: float

    parameters: dict[str, float]

    cup: int | None = None
    stage: int | None = None

    role: str = "working"

    rfq: RFQState = field(default_factory=RFQState)

    fixed_conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    state_id: str = field(default_factory=lambda: str(uuid4()))
    created_at_utc: str = field(default_factory=utc_now_iso)

    schema_version: int = STATE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.mass_u <= 0:
            raise ValueError("Ion mass must be greater than zero")

        if self.cup is not None and not 1 <= self.cup <= 6:
            raise ValueError("SIRIUS cup must be between 1 and 6")

        if self.stage is not None and not 1 <= self.stage <= 6:
            raise ValueError("SIRIUS stage must be between 1 and 6")

        for name, value in self.parameters.items():
            if name not in PARAMETERS:
                raise ValueError(f"Unknown SIRIUS parameter: {name}")

            definition = PARAMETERS[name]

            if not definition.minimum <= value <= definition.maximum:
                raise ValueError(
                    f"{name}={value} outside allowed state range "
                    f"{definition.minimum}..{definition.maximum} "
                    f"{definition.unit}"
                )

        self._validate_rfq()

    def _validate_rfq(self) -> None:
        if self.rfq.frequency_hz is not None and self.rfq.frequency_hz <= 0:
            raise ValueError("RFQ frequency must be greater than zero")

        if (
            self.rfq.generator_amplitude_vpp is not None
            and self.rfq.generator_amplitude_vpp < 0
        ):
            raise ValueError(
                "RFQ generator amplitude must be non-negative"
            )

        if self.rfq.inductance_uh is not None and self.rfq.inductance_uh < 0:
            raise ValueError("RFQ inductance must be non-negative")

        if self.rfq.capacitance_pf is not None and self.rfq.capacitance_pf < 0:
            raise ValueError("RFQ capacitance must be non-negative")

        if (
            self.rfq.rfq_vpp_measured is not None
            and self.rfq.rfq_vpp_measured < 0
        ):
            raise ValueError("Measured RFQ Vpp must be non-negative")

        for name, value in (
            ("q_target", self.rfq.q_target),
            ("q_nominal", self.rfq.q_nominal),
            ("q_measured", self.rfq.q_measured),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.rfq.q_target is not None and self.rfq.q_target > 0.9:
            raise ValueError("SIRIUS q target must not exceed 0.9")

        if self.rfq.q_measured is not None and self.rfq.q_measured > 0.9:
            raise ValueError("Measured RFQ q must not exceed 0.9")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self, path: str | Path) -> Path:
        self.validate()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        return output_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MachineState":
        data = dict(data)

        rfq_data = data.pop("rfq", {})
        rfq = RFQState(**rfq_data)

        state = cls(
            rfq=rfq,
            **data,
        )

        state.validate()
        return state

    @classmethod
    def from_json(cls, path: str | Path) -> "MachineState":
        input_path = Path(path)

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        return cls.from_dict(data)