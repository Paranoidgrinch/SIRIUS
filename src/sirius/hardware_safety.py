from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    field,
    is_dataclass,
    replace,
)
from typing import Any

from sirius.readback_quality import (
    ReadbackQualityPolicy,
)

from sirius.readback_freshness import (
    ReadbackFreshnessPolicy,
)

from sirius.hardware_guard import (
    HardwareGuardPolicy,
)

from sirius.coupled_transition import (
    CoupledTransitionPolicy,
    cooler_end_transition_policy,
    qpt_transition_policy,
)
from sirius.cup_ack import CupSelectionPolicy


DEFAULT_COOLER_END_ORDER = (
    "deceleration_voltage_v",
    "acceleration_voltage_v",
)

DEFAULT_QPT_ORDER = (
    "quadrupole1_voltage_v",
    "quadrupole2_voltage_v",
    "quadrupole3_voltage_v",
)


@dataclass(frozen=True)
class HardwareSafetyConfig:
    """
    Central hardware-transition safety configuration for SIRIUS.

    The coupled-transition step sizes are deliberately mandatory.

    SIRIUS must not silently invent a facility-safe HV or QPT step size.
    These values must be chosen explicitly for the real installation.

    This configuration contains command-transition safety settings only.
    It does not claim that the configured values are intrinsic hardware
    limits or manufacturer specifications.
    """

    cooler_end_max_step_v: float
    qpt_max_step_v: float

    cup_selection_policy: CupSelectionPolicy = field(
        default_factory=CupSelectionPolicy
    )

    readback_freshness_policy: ReadbackFreshnessPolicy = field(
        default_factory=ReadbackFreshnessPolicy
    )

    readback_quality_policy: (
        ReadbackQualityPolicy | None
    ) = None

    hardware_guard_policy: (
        HardwareGuardPolicy | None
    ) = None

    cooler_end_parameter_order: tuple[
        str,
        ...
    ] = DEFAULT_COOLER_END_ORDER

    qpt_parameter_order: tuple[
        str,
        ...
    ] = DEFAULT_QPT_ORDER

    def __post_init__(self) -> None:
        for name, value in (
            (
                "cooler_end_max_step_v",
                self.cooler_end_max_step_v,
            ),
            (
                "qpt_max_step_v",
                self.qpt_max_step_v,
            ),
        ):
            numeric = float(
                value
            )

            if (
                not math.isfinite(
                    numeric
                )
                or numeric <= 0
            ):
                raise ValueError(
                    f"{name} must be finite and greater than zero"
                )

        if (
            set(
                self.cooler_end_parameter_order
            )
            != set(
                DEFAULT_COOLER_END_ORDER
            )
            or len(
                self.cooler_end_parameter_order
            )
            != 2
        ):
            raise ValueError(
                "cooler_end_parameter_order must contain "
                "deceleration_voltage_v and acceleration_voltage_v "
                "exactly once"
            )

        if (
            set(
                self.qpt_parameter_order
            )
            != set(
                DEFAULT_QPT_ORDER
            )
            or len(
                self.qpt_parameter_order
            )
            != 3
        ):
            raise ValueError(
                "qpt_parameter_order must contain QPT1, QPT2, and QPT3 "
                "exactly once"
            )

        if not isinstance(
            self.cup_selection_policy,
            CupSelectionPolicy,
        ):
            raise TypeError(
                "cup_selection_policy must be a CupSelectionPolicy"
            )

        if not isinstance(
            self.readback_freshness_policy,
            ReadbackFreshnessPolicy,
        ):
            raise TypeError(
                "readback_freshness_policy must be "
                "ReadbackFreshnessPolicy"
            )

        if (
            self.readback_quality_policy
            is not None
            and not isinstance(
                self.readback_quality_policy,
                ReadbackQualityPolicy,
            )
        ):
            raise TypeError(
                "readback_quality_policy must be "
                "ReadbackQualityPolicy or None"
            )

        if (
            self.hardware_guard_policy
            is not None
            and not isinstance(
                self.hardware_guard_policy,
                HardwareGuardPolicy,
            )
        ):
            raise TypeError(
                "hardware_guard_policy must be "
                "HardwareGuardPolicy or None"
            )

    def cooler_end_transition_policy(
        self,
    ) -> CoupledTransitionPolicy:
        """
        Build the bounded HV1/HV4 transition policy for this run.
        """

        return cooler_end_transition_policy(
            max_step_v=float(
                self.cooler_end_max_step_v
            ),
            parameter_order=(
                self.cooler_end_parameter_order
            ),
        )

    def qpt_transition_policy(
        self,
    ) -> CoupledTransitionPolicy:
        """
        Build the bounded QPT1/QPT2/QPT3 transition policy for this run.
        """

        return qpt_transition_policy(
            max_step_v=float(
                self.qpt_max_step_v
            ),
            parameter_order=(
                self.qpt_parameter_order
            ),
        )

    def bind_end_electrode_policy(
        self,
        policy: Any,
    ):
        """
        Return a copy of an EndElectrodeCoordinatePolicy-like dataclass
        with the run's bounded HV transition policy installed.

        Keeping this generic avoids making the central hardware-safety
        module depend on the Cup-3 optimizer implementation.
        """

        return _replace_transition_policy(
            policy,
            self.cooler_end_transition_policy(),
            expected_name=(
                "end-electrode policy"
            ),
        )

    def bind_qpt_scan_policy(
        self,
        policy: Any,
    ):
        """
        Return a copy of a QPT2DScanPolicy-like dataclass with the run's
        bounded QPT transition policy installed.
        """

        return _replace_transition_policy(
            policy,
            self.qpt_transition_policy(),
            expected_name=(
                "QPT scan policy"
            ),
        )

    def transition_apply_kwargs(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        """
        Keyword arguments for normal apply_state() calls that may select
        a Faraday cup.
        """

        return {
            "cup_selection_policy": (
                self.cup_selection_policy
            ),
        }

    def to_manifest_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        """
        JSON-safe representation for the SIRIUS run manifest.
        """

        cup = self.cup_selection_policy

        return {
            "cooler_end_max_step_v": float(
                self.cooler_end_max_step_v
            ),
            "qpt_max_step_v": float(
                self.qpt_max_step_v
            ),
            "cooler_end_parameter_order": list(
                self.cooler_end_parameter_order
            ),
            "qpt_parameter_order": list(
                self.qpt_parameter_order
            ),
            "hardware_guard": (
                None
                if self.hardware_guard_policy
                is None
                else {
                    "reject_unconfigured_changes": bool(
                        self.hardware_guard_policy
                        .reject_unconfigured_changes
                    ),
                    "max_total_steps": int(
                        self.hardware_guard_policy
                        .max_total_steps
                    ),
                    "parameter_rules": {
                        name: {
                            "max_step": float(
                                rule.max_step
                            ),
                            "minimum_command_interval_s": float(
                                rule.minimum_command_interval_s
                            ),
                            "require_readback": bool(
                                rule.require_readback
                            ),
                            "require_settling": bool(
                                rule.require_settling
                            ),
                        }
                        for name, rule
                        in self.hardware_guard_policy
                        .parameter_rules.items()
                    },
                }
            ),
            "readback_quality": (
                None
                if self.readback_quality_policy
                is None
                else {
                    "accepted_values": list(
                        self.readback_quality_policy
                        .accepted_values
                    ),
                    "allow_missing": bool(
                        self.readback_quality_policy
                        .allow_missing
                    ),
                    "case_sensitive": bool(
                        self.readback_quality_policy
                        .case_sensitive
                    ),
                }
            ),
            "readback_freshness": {
                "timeout_s": float(
                    self.readback_freshness_policy.timeout_s
                ),
                "poll_interval_s": float(
                    self.readback_freshness_policy.poll_interval_s
                ),
            },
            "cup_selection": {
                "timeout_s": float(
                    cup.timeout_s
                ),
                "poll_interval_s": float(
                    cup.poll_interval_s
                ),
                "minimum_wait_s": float(
                    cup.minimum_wait_s
                ),
                "consecutive_confirmations": int(
                    cup.consecutive_confirmations
                ),
            },
        }


def _replace_transition_policy(
    policy: Any,
    transition_policy: CoupledTransitionPolicy,
    *,
    expected_name: str,
):
    if not is_dataclass(
        policy
    ):
        raise TypeError(
            f"{expected_name} must be a dataclass instance"
        )

    fields = getattr(
        policy,
        "__dataclass_fields__",
        {},
    )

    if "transition_policy" not in fields:
        raise TypeError(
            f"{expected_name} does not expose transition_policy"
        )

    return replace(
        policy,
        transition_policy=(
            transition_policy
        ),
    )