from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
)

from sirius.comparison import (
    ComparisonDecision,
    ComparisonPolicy,
)
from sirius.objective import (
    ScalarEstimate,
    compare_estimates,
)


@dataclass(frozen=True)
class OptimizationAxis:
    name: str
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError(
                "Optimization axis name must not be empty"
            )

        minimum = float(
            self.minimum
        )

        maximum = float(
            self.maximum
        )

        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
        ):
            raise ValueError(
                f"Bounds for {self.name} must be finite"
            )

        if maximum <= minimum:
            raise ValueError(
                f"Maximum must exceed minimum for {self.name}"
            )

    @property
    def span(
        self,
    ) -> float:
        return (
            float(self.maximum)
            - float(self.minimum)
        )


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """
    One real or simulated machine evaluation.

    value:
        Scalar optimization objective.

    sem:
        Standard error of that objective estimate.

    safe:
        Whether the observed point remained operationally valid.

    The optimizer should normally avoid evaluating points that fail the
    problem's a-priori safety predicate. safe=False additionally allows an
    evaluator to report an unexpected runtime constraint violation.
    """

    point: tuple[
        float,
        ...
    ]

    value: float
    sem: float

    safe: bool = True

    below_noise_floor: bool = False

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.point:
            raise ValueError(
                "Evaluation point must not be empty"
            )

        if not all(
            math.isfinite(
                float(value)
            )
            for value
            in self.point
        ):
            raise ValueError(
                "Evaluation point must contain only finite values"
            )

        if not math.isfinite(
            float(
                self.value
            )
        ):
            raise ValueError(
                "Objective value must be finite"
            )

        if (
            not math.isfinite(
                float(
                    self.sem
                )
            )
            or float(
                self.sem
            ) < 0
        ):
            raise ValueError(
                "Objective SEM must be finite and non-negative"
            )


EvaluationFunction = Callable[
    [
        tuple[
            float,
            ...
        ]
    ],
    ObjectiveEvaluation,
]

SafetyPredicate = Callable[
    [
        tuple[
            float,
            ...
        ]
    ],
    bool,
]

ComparisonFunction = Callable[
    [
        ObjectiveEvaluation,
        ObjectiveEvaluation,
    ],
    bool,
]


@dataclass(frozen=True)
class OptimizationProblem:
    axes: tuple[
        OptimizationAxis,
        ...
    ]

    initial_point: tuple[
        float,
        ...
    ]

    maximize: bool = True

    safety_predicate: (
        SafetyPredicate | None
    ) = None

    comparison: (
        ComparisonFunction | None
    ) = None

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError(
                "Optimization problem requires at least one axis"
            )

        names = [
            axis.name
            for axis
            in self.axes
        ]

        if len(
            names
        ) != len(
            set(
                names
            )
        ):
            raise ValueError(
                "Optimization axis names must be unique"
            )

        self.validate_point(
            self.initial_point
        )

        if not self.is_allowed(
            self.initial_point
        ):
            raise ValueError(
                "Initial optimization point is not allowed"
            )

    @property
    def dimension(
        self,
    ) -> int:
        return len(
            self.axes
        )

    def validate_point(
        self,
        point: tuple[
            float,
            ...
        ],
    ) -> None:
        if len(
            point
        ) != self.dimension:
            raise ValueError(
                "Optimization point dimension does not match problem"
            )

        for axis, value in zip(
            self.axes,
            point,
        ):
            value = float(
                value
            )

            if not math.isfinite(
                value
            ):
                raise ValueError(
                    f"{axis.name} must be finite"
                )

            if not (
                axis.minimum
                <= value
                <= axis.maximum
            ):
                raise ValueError(
                    f"{axis.name}={value} outside "
                    f"{axis.minimum}..{axis.maximum}"
                )

    def is_allowed(
        self,
        point: tuple[
            float,
            ...
        ],
    ) -> bool:
        try:
            self.validate_point(
                point
            )
        except ValueError:
            return False

        if self.safety_predicate is None:
            return True

        return bool(
            self.safety_predicate(
                point
            )
        )

    def is_better(
        self,
        candidate: ObjectiveEvaluation,
        incumbent: ObjectiveEvaluation,
    ) -> bool:
        if not candidate.safe:
            return False

        if not incumbent.safe:
            return True

        if self.comparison is not None:
            return bool(
                self.comparison(
                    candidate,
                    incumbent,
                )
            )

        if self.maximize:
            return (
                candidate.value
                > incumbent.value
            )

        return (
            candidate.value
            < incumbent.value
        )


@dataclass(frozen=True)
class OptimizationResult:
    optimizer_name: str

    initial_evaluation: ObjectiveEvaluation
    best_evaluation: ObjectiveEvaluation

    history: tuple[
        ObjectiveEvaluation,
        ...
    ]

    iterations: int
    termination_reason: str

    optimizer_version: str | None = None

    metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    @property
    def evaluations(
        self,
    ) -> int:
        return len(
            self.history
        )


class Optimizer(Protocol):
    name: str

    def optimize(
        self,
        problem: OptimizationProblem,
        evaluator: EvaluationFunction,
    ) -> OptimizationResult:
        ...


def uncertainty_aware_comparator(
    *,
    maximize: bool,
    sigma_factor: float = 2.0,
    absolute_margin: float = 0.0,
) -> ComparisonFunction:
    """
    Construct a simple uncertainty-aware optimizer comparator.

    A candidate must improve the objective by more than the combined
    uncertainty margin:

        margin =
            absolute_margin
            + sigma_factor * sqrt(
                sem_candidate^2
                + sem_incumbent^2
            )

    Later, SIRIUS hardware adapters can instead bridge directly to the
    existing ComparisonPolicy / compare_estimates() implementation.
    """

    sigma_factor = float(
        sigma_factor
    )

    absolute_margin = float(
        absolute_margin
    )

    if (
        not math.isfinite(
            sigma_factor
        )
        or sigma_factor < 0
    ):
        raise ValueError(
            "sigma_factor must be finite and non-negative"
        )

    if (
        not math.isfinite(
            absolute_margin
        )
        or absolute_margin < 0
    ):
        raise ValueError(
            "absolute_margin must be finite and non-negative"
        )

    def compare(
        candidate: ObjectiveEvaluation,
        incumbent: ObjectiveEvaluation,
    ) -> bool:
        if not candidate.safe:
            return False

        if not incumbent.safe:
            return True

        margin = (
            absolute_margin
            + sigma_factor
            * math.hypot(
                float(
                    candidate.sem
                ),
                float(
                    incumbent.sem
                ),
            )
        )

        if maximize:
            return (
                candidate.value
                - incumbent.value
                > margin
            )

        return (
            incumbent.value
            - candidate.value
            > margin
        )

    return compare


def comparison_policy_comparator(
    *,
    policy: ComparisonPolicy,
) -> ComparisonFunction:
    """
    Adapt SIRIUS' canonical scalar ComparisonPolicy to the generic
    optimizer comparison interface.

    This comparator is intended for higher-is-better scalar objectives,
    especially Cup-1-normalized beam transmission.

    Reusing compare_estimates() ensures that generic optimizers and the
    existing transmission scanners make identical decisions about
    measurement uncertainty, practical improvement thresholds, and the
    below-noise-floor state.

    Hardware safety remains authoritative outside this comparison layer.
    """

    def compare(
        candidate: ObjectiveEvaluation,
        incumbent: ObjectiveEvaluation,
    ) -> bool:
        if not candidate.safe:
            return False

        if not incumbent.safe:
            return True

        baseline = ScalarEstimate(
            value=float(
                incumbent.value
            ),
            sem=float(
                incumbent.sem
            ),
            below_noise_floor=bool(
                incumbent.below_noise_floor
            ),
        )

        challenger = ScalarEstimate(
            value=float(
                candidate.value
            ),
            sem=float(
                candidate.sem
            ),
            below_noise_floor=bool(
                candidate.below_noise_floor
            ),
        )

        result = compare_estimates(
            baseline,
            challenger,
            policy,
        )

        return (
            result.decision
            == ComparisonDecision.BETTER
        )

    return compare

