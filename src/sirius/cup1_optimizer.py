from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Mapping

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
    compare_measurements,
)
from sirius.magnet_model import (
    MagnetPrediction,
    predict_magnet,
)
from sirius.mass_profile import MassProfile
from sirius.measurement import (
    BeamMeasurement,
    MeasurementPolicy,
    measure_beam_current,
)
from sirius.parameters import PARAMETERS
from sirius.reference import (
    SourceReference,
    SourceReferenceTracker,
)
from sirius.scan1d import (
    OneDimensionalScanResult,
    ScanPolicy,
    scan_parameter_1d,
)
from sirius.settling import SettlingPolicy
from sirius.state import (
    MachineState,
    utc_now_iso,
)
from sirius.transition import (
    AppliedStateResult,
    capture_readbacks,
)
from sirius.safe_transition import apply_state


CUP1_REQUIRED_PARAMETERS = (
    "sputter_voltage_v",
    "extraction_voltage_v",
    "einzel_lens_voltage_v",
    "magnet_current_a",
)


class Cup1OptimizationError(RuntimeError):
    pass


class Cup1OptimizationNoBeamError(
    Cup1OptimizationError
):
    pass


@dataclass(frozen=True)
class Cup1OptimizationPolicy:
    """
    Initial conservative Cup-1 coordinate-descent policy.

    These values are starting defaults only. They should later be tuned
    from real SIRIUS run data.
    """

    magnet_half_width_a: float = 1.0

    magnet_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                0.25,
                0.05,
            )
        )
    )

    einzel_half_width_v: float = 2500.0

    einzel_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                500.0,
                100.0,
            )
        )
    )

    sputter_half_width_v: float = 1500.0

    sputter_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                500.0,
                100.0,
            )
        )
    )

    extraction_half_width_v: float = 2000.0

    extraction_scan: ScanPolicy = field(
        default_factory=lambda: ScanPolicy(
            steps=(
                500.0,
                100.0,
                25.0,
            )
        )
    )

    source_passes: int = 2

    def __post_init__(self) -> None:
        for name, value in (
            (
                "magnet_half_width_a",
                self.magnet_half_width_a,
            ),
            (
                "einzel_half_width_v",
                self.einzel_half_width_v,
            ),
            (
                "sputter_half_width_v",
                self.sputter_half_width_v,
            ),
            (
                "extraction_half_width_v",
                self.extraction_half_width_v,
            ),
        ):
            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero"
                )

        if self.source_passes < 1:
            raise ValueError(
                "source_passes must be at least 1"
            )


@dataclass(frozen=True)
class Cup1OptimizationResult:
    initial_state: MachineState
    initial_measurement: BeamMeasurement

    magnet_prediction: MagnetPrediction

    predicted_magnet_state: MachineState | None
    predicted_magnet_measurement: (
        BeamMeasurement | None
    )

    magnet_seed_source: str
    magnet_seed_state: MachineState

    magnet_scan: OneDimensionalScanResult

    source_scans: tuple[
        OneDimensionalScanResult,
        ...
    ]

    final_state: MachineState
    final_measurement: BeamMeasurement

    reference: SourceReference


def _commands_equal(
    first: float,
    second: float,
) -> bool:
    return math.isclose(
        float(first),
        float(second),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _replace_parameter(
    state: MachineState,
    parameter_name: str,
    command_value: float,
    *,
    role: str,
) -> MachineState:
    parameters = dict(
        state.parameters
    )

    parameters[
        parameter_name
    ] = float(
        command_value
    )

    readbacks = dict(
        state.readbacks
    )

    # The old physical observation no longer belongs to the new command.
    readbacks.pop(
        parameter_name,
        None,
    )

    new_state = MachineState(
        mass_u=state.mass_u,
        parameters=parameters,
        readbacks=readbacks,
        cup=state.cup,
        stage=state.stage,
        role=role,
        rfq=deepcopy(
            state.rfq
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata=deepcopy(
            state.metadata
        ),
    )

    new_state.validate()

    return new_state


def _retag_state(
    state: MachineState,
    *,
    role: str,
    metadata: dict | None = None,
) -> MachineState:
    combined_metadata = deepcopy(
        state.metadata
    )

    if metadata:
        combined_metadata.update(
            metadata
        )

    result = MachineState(
        mass_u=state.mass_u,
        parameters=dict(
            state.parameters
        ),
        readbacks=dict(
            state.readbacks
        ),
        cup=state.cup,
        stage=state.stage,
        role=role,
        rfq=deepcopy(
            state.rfq
        ),
        fixed_conditions=deepcopy(
            state.fixed_conditions
        ),
        metadata=combined_metadata,
    )

    result.validate()

    return result


def _local_profile(
    profile: MassProfile,
    parameter_name: str,
    center: float,
    half_width: float,
) -> MassProfile:
    """
    Create an ephemeral MassProfile copy containing a local search window.

    Persistent learned bounds are not modified here.

    If the current machine command lies outside an old learned range,
    SIRIUS allows a path from the current value back toward the learned
    region, but does not expand farther away from it.
    """

    local = deepcopy(
        profile
    )

    hard = PARAMETERS[
        parameter_name
    ]

    learned_minimum, learned_maximum = (
        profile.effective_bounds(
            parameter_name
        )
    )

    allowed_minimum = max(
        float(hard.minimum),
        min(
            float(center),
            float(learned_minimum),
        ),
    )

    allowed_maximum = min(
        float(hard.maximum),
        max(
            float(center),
            float(learned_maximum),
        ),
    )

    local_minimum = max(
        allowed_minimum,
        float(center)
        - float(half_width),
    )

    local_maximum = min(
        allowed_maximum,
        float(center)
        + float(half_width),
    )

    local.set_learned_range(
        parameter_name,
        local_minimum,
        local_maximum,
        source="cup1_local_window",
    )

    return local


def _validate_inputs(
    state: MachineState,
    profile: MassProfile,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
) -> None:
    state.validate()
    profile.validate()

    if state.mass_u != profile.mass_u:
        raise ValueError(
            "Machine state and mass profile must use the same ion mass"
        )

    if state.cup != 1:
        raise ValueError(
            "Cup-1 optimization requires cup 1 to be selected"
        )

    if state.stage not in (
        None,
        1,
    ):
        raise ValueError(
            "Cup-1 optimization requires stage 1 or no stage assignment"
        )

    for parameter_name in (
        CUP1_REQUIRED_PARAMETERS
    ):
        if parameter_name not in state.parameters:
            raise ValueError(
                f"Cup-1 state is missing {parameter_name}"
            )

        if (
            parameter_name
            not in settling_policies
        ):
            raise KeyError(
                f"No settling policy configured for {parameter_name}"
            )


def _physics_voltage(
    state: MachineState,
    parameter_name: str,
) -> float:
    """
    Prefer the actual physical readback for beam-physics calculations.

    Fall back to the command only when no readback is available.
    """

    if parameter_name in state.readbacks:
        return float(
            state.readbacks[
                parameter_name
            ]
        )

    return float(
        state.parameters[
            parameter_name
        ]
    )


def _update_profile_from_final_state(
    profile: MassProfile,
    state: MachineState,
) -> None:
    for parameter_name in (
        CUP1_REQUIRED_PARAMETERS
    ):
        profile.set_best_command(
            parameter_name,
            state.parameters[
                parameter_name
            ],
        )

    profile.set_best_state(
        "cup1_reference",
        state.state_id,
    )


def optimize_cup1(
    adapter,
    current_state: MachineState,
    profile: MassProfile,
    tracker: SourceReferenceTracker,
    settling_policies: Mapping[
        str,
        SettlingPolicy,
    ],
    measurement_policy: MeasurementPolicy,
    comparison_policy: ComparisonPolicy,
    *,
    optimization_policy: (
        Cup1OptimizationPolicy | None
    ) = None,
    noise_floor_a: float | None = None,
    logger=None,
    monotonic: Callable[
        [],
        float,
    ] = time.monotonic,
    utc_now: Callable[
        [],
        str,
    ] = utc_now_iso,
) -> Cup1OptimizationResult:
    """
    Optimize source -> analyzing magnet -> Cup 1.

    Strategy:

      1. Capture fresh source-voltage readbacks.
      2. Measure the existing Cup-1 state.
      3. Calculate the FLAVIA-compatible magnet prediction using the
         physical source-voltage readbacks when available.
      4. Test the predicted magnet point against the existing state.
      5. Select a physically sensible magnet seed.
      6. Perform a narrow, slow magnet scan.
      7. Freeze the magnet.
      8. Coordinate-scan einzel lens, sputter and extraction.
      9. Measure the final Cup-1 current.
     10. Declare the final state the Cup-1 100 % reference state.
     11. Update the in-memory mass profile with the coherent final commands.

    The MassProfile is mutated in memory but is not persisted here.
    Persistence remains the responsibility of MassProfileStore.
    """

    policy = (
        optimization_policy
        if optimization_policy is not None
        else Cup1OptimizationPolicy()
    )

    _validate_inputs(
        current_state,
        profile,
        settling_policies,
    )

    # We need fresh physical voltages before calculating magnetic rigidity.
    initial_state = capture_readbacks(
        adapter,
        current_state,
    )

    initial_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
        )
    )

    if logger is not None:
        logger.log_event(
            "cup1_optimization_started",
            {
                "state_id": (
                    initial_state.state_id
                ),
                "mass_u": (
                    initial_state.mass_u
                ),
                "commands": (
                    initial_state.parameters
                ),
                "readbacks": (
                    initial_state.readbacks
                ),
            },
        )

        logger.log_measurement(
            initial_measurement,
            cup=1,
            state_id=(
                initial_state.state_id
            ),
            purpose="cup1_initial",
        )

    sputter_for_physics = (
        _physics_voltage(
            initial_state,
            "sputter_voltage_v",
        )
    )

    extraction_for_physics = (
        _physics_voltage(
            initial_state,
            "extraction_voltage_v",
        )
    )

    magnet_prediction = predict_magnet(
        mass_u=initial_state.mass_u,
        sputter_voltage_v=(
            sputter_for_physics
        ),
        extraction_voltage_v=(
            extraction_for_physics
        ),
    )

    if magnet_prediction.current_clamped:
        raise Cup1OptimizationError(
            "Physics-derived magnet current lies outside the "
            "available 0..120 A range"
        )

    if logger is not None:
        logger.log_event(
            "magnet_prediction",
            {
                "mass_u": (
                    magnet_prediction.mass_u
                ),
                "sputter_voltage_for_physics_v": (
                    sputter_for_physics
                ),
                "extraction_voltage_for_physics_v": (
                    extraction_for_physics
                ),
                "prediction": (
                    magnet_prediction
                ),
            },
        )

    existing_magnet = float(
        initial_state.parameters[
            "magnet_current_a"
        ]
    )

    predicted_magnet = float(
        magnet_prediction.command_current_a
    )

    predicted_state = None
    predicted_measurement = None

    physical_state = initial_state

    if _commands_equal(
        existing_magnet,
        predicted_magnet,
    ):
        magnet_seed_source = (
            "existing_matches_physics"
        )

        magnet_seed_state = (
            initial_state
        )

    else:
        predicted_state = (
            _replace_parameter(
                initial_state,
                "magnet_current_a",
                predicted_magnet,
                role="magnet_prediction_test",
            )
        )

        transition = apply_state(
            adapter,
            current=initial_state,
            target=predicted_state,
            settling_policies=(
                settling_policies
            ),
            select_target_cup=False,
        )

        physical_state = (
            transition.observed_state
        )

        if logger is not None:
            logger.log_state_transition(
                transition
            )

        predicted_measurement = (
            measure_beam_current(
                adapter,
                measurement_policy,
                noise_floor_a=noise_floor_a,
            )
        )

        if logger is not None:
            logger.log_measurement(
                predicted_measurement,
                cup=1,
                state_id=(
                    physical_state.state_id
                ),
                purpose=(
                    "cup1_physics_magnet_test"
                ),
            )

        comparison = (
            compare_measurements(
                initial_measurement,
                predicted_measurement,
                comparison_policy,
            )
        )

        # Cold-start rule:
        #
        # If the existing state has no detectable beam, retain the
        # physics-derived magnet position even if both measurements are
        # below the noise floor. It is a more meaningful center for the
        # subsequent local magnet search.
        if initial_measurement.below_noise_floor:
            use_prediction = True

        else:
            use_prediction = (
                comparison.decision
                == ComparisonDecision.BETTER
            )

        if use_prediction:
            magnet_seed_source = (
                "physics_prediction"
            )

            magnet_seed_state = (
                physical_state
            )

        else:
            magnet_seed_source = (
                "existing_state"
            )

            restoration = apply_state(
                adapter,
                current=physical_state,
                target=initial_state,
                settling_policies=(
                    settling_policies
                ),
                select_target_cup=False,
            )

            physical_state = (
                restoration.observed_state
            )

            magnet_seed_state = (
                physical_state
            )

            if logger is not None:
                logger.log_state_transition(
                    restoration
                )

        if logger is not None:
            logger.log_optimizer_decision(
                stage=1,
                cup=1,
                parameter="magnet_current_a",
                decision=(
                    magnet_seed_source
                ),
                baseline_state_id=(
                    initial_state.state_id
                ),
                candidate_state_id=(
                    predicted_state.state_id
                ),
                details={
                    "existing_command_a": (
                        existing_magnet
                    ),
                    "predicted_command_a": (
                        predicted_magnet
                    ),
                    "comparison": (
                        comparison
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Narrow magnet optimization.
    # ------------------------------------------------------------------

    magnet_center = float(
        magnet_seed_state.parameters[
            "magnet_current_a"
        ]
    )

    magnet_profile = (
        _local_profile(
            profile,
            "magnet_current_a",
            magnet_center,
            policy.magnet_half_width_a,
        )
    )

    magnet_scan = scan_parameter_1d(
        adapter,
        magnet_seed_state,
        magnet_profile,
        "magnet_current_a",
        policy.magnet_scan,
        settling_policies,
        measurement_policy,
        comparison_policy,
        noise_floor_a=noise_floor_a,
        logger=logger,
    )

    working_state = (
        magnet_scan.final_state
    )

    frozen_magnet_command = float(
        working_state.parameters[
            "magnet_current_a"
        ]
    )

    if logger is not None:
        logger.log_event(
            "cup1_magnet_frozen",
            {
                "state_id": (
                    working_state.state_id
                ),
                "magnet_command_a": (
                    frozen_magnet_command
                ),
                "magnet_readback_a": (
                    working_state.readbacks.get(
                        "magnet_current_a"
                    )
                ),
            },
        )

    # ------------------------------------------------------------------
    # Source / einzel coordinate descent.
    #
    # The magnet is never a scan parameter again after this point.
    # ------------------------------------------------------------------

    source_scans: list[
        OneDimensionalScanResult
    ] = []

    source_parameters = (
        (
            "einzel_lens_voltage_v",
            policy.einzel_half_width_v,
            policy.einzel_scan,
        ),
        (
            "sputter_voltage_v",
            policy.sputter_half_width_v,
            policy.sputter_scan,
        ),
        (
            "extraction_voltage_v",
            policy.extraction_half_width_v,
            policy.extraction_scan,
        ),
    )

    for pass_index in range(
        1,
        policy.source_passes + 1,
    ):
        if logger is not None:
            logger.log_event(
                "cup1_source_pass_started",
                {
                    "pass": pass_index,
                    "magnet_frozen_command_a": (
                        frozen_magnet_command
                    ),
                },
            )

        for (
            parameter_name,
            half_width,
            scan_policy,
        ) in source_parameters:
            center = float(
                working_state.parameters[
                    parameter_name
                ]
            )

            local_profile = (
                _local_profile(
                    profile,
                    parameter_name,
                    center,
                    half_width,
                )
            )

            scan = scan_parameter_1d(
                adapter,
                working_state,
                local_profile,
                parameter_name,
                scan_policy,
                settling_policies,
                measurement_policy,
                comparison_policy,
                noise_floor_a=(
                    noise_floor_a
                ),
                logger=logger,
            )

            working_state = (
                scan.final_state
            )

            source_scans.append(
                scan
            )

            current_magnet = float(
                working_state.parameters[
                    "magnet_current_a"
                ]
            )

            if not _commands_equal(
                current_magnet,
                frozen_magnet_command,
            ):
                raise Cup1OptimizationError(
                    "Magnet command changed after the magnet was frozen"
                )

    # Capture final physical values before the definitive Cup-1 reference.
    working_state = (
        capture_readbacks(
            adapter,
            working_state,
        )
    )

    final_state = _retag_state(
        working_state,
        role="cup1_reference",
        metadata={
            "optimized_stage": 1,
            "magnet_seed_source": (
                magnet_seed_source
            ),
            "magnet_frozen_command_a": (
                frozen_magnet_command
            ),
        },
    )

    final_measurement = (
        measure_beam_current(
            adapter,
            measurement_policy,
            noise_floor_a=noise_floor_a,
        )
    )

    if (
        final_measurement.below_noise_floor
        or final_measurement.mean_a <= 0
    ):
        raise Cup1OptimizationNoBeamError(
            "Final Cup-1 optimization did not produce a valid "
            "reference beam current"
        )

    reference = SourceReference(
        measurement=final_measurement,
        state_id=final_state.state_id,
        mass_u=final_state.mass_u,
        monotonic_s=monotonic(),
        created_at_utc=utc_now(),
    )

    tracker.add(
        reference
    )

    _update_profile_from_final_state(
        profile,
        final_state,
    )

    if logger is not None:
        logger.save_state(
            final_state,
            "cup1_reference",
        )

        logger.log_measurement(
            final_measurement,
            cup=1,
            state_id=(
                final_state.state_id
            ),
            purpose=(
                "cup1_final_reference"
            ),
        )

        logger.log_reference(
            reference
        )

        logger.log_event(
            "cup1_optimization_completed",
            {
                "state_id": (
                    final_state.state_id
                ),
                "reference_current_a": (
                    final_measurement.mean_a
                ),
                "reference_sem_a": (
                    final_measurement.sem_a
                ),
                "commands": (
                    final_state.parameters
                ),
                "readbacks": (
                    final_state.readbacks
                ),
                "source_scan_count": (
                    len(source_scans)
                ),
            },
        )

    return Cup1OptimizationResult(
        initial_state=initial_state,
        initial_measurement=(
            initial_measurement
        ),
        magnet_prediction=(
            magnet_prediction
        ),
        predicted_magnet_state=(
            predicted_state
        ),
        predicted_magnet_measurement=(
            predicted_measurement
        ),
        magnet_seed_source=(
            magnet_seed_source
        ),
        magnet_seed_state=(
            magnet_seed_state
        ),
        magnet_scan=magnet_scan,
        source_scans=tuple(
            source_scans
        ),
        final_state=final_state,
        final_measurement=(
            final_measurement
        ),
        reference=reference,
    )