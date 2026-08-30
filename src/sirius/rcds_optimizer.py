from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from sirius.optimizer_api import (
    EvaluationFunction,
    ObjectiveEvaluation,
    OptimizationProblem,
    OptimizationResult,
)


class OptimizationBudgetExhausted(
    RuntimeError
):
    pass


@dataclass(frozen=True)
class RCDSPolicy:
    """
    Robust conjugate-direction search policy.

    Coordinates are internally normalized to [0, 1].

    The line search uses several distributed samples followed, where
    justified, by a local parabolic refinement. This makes the search
    substantially less dependent on one noisy function evaluation than
    naive hill climbing.
    """

    max_iterations: int = 12

    max_evaluations: int = 300

    line_samples: int = 7

    line_half_width: float = 0.35

    minimum_direction_norm: float = 1e-6

    stall_iterations: int = 2

    parabolic_refinement: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1"
            )

        if self.max_evaluations < 2:
            raise ValueError(
                "max_evaluations must be at least 2"
            )

        if (
            self.line_samples < 3
            or self.line_samples % 2 == 0
        ):
            raise ValueError(
                "line_samples must be an odd integer >= 3"
            )

        if (
            not math.isfinite(
                float(
                    self.line_half_width
                )
            )
            or self.line_half_width <= 0
        ):
            raise ValueError(
                "line_half_width must be finite and positive"
            )

        if (
            not math.isfinite(
                float(
                    self.minimum_direction_norm
                )
            )
            or self.minimum_direction_norm <= 0
        ):
            raise ValueError(
                "minimum_direction_norm must be finite and positive"
            )

        if self.stall_iterations < 1:
            raise ValueError(
                "stall_iterations must be at least 1"
            )


def _norm(
    vector: tuple[
        float,
        ...
    ],
) -> float:
    return math.sqrt(
        sum(
            value * value
            for value
            in vector
        )
    )


def _normalize_direction(
    vector: tuple[
        float,
        ...
    ],
) -> tuple[
    float,
    ...
]:
    magnitude = _norm(
        vector
    )

    if magnitude <= 0:
        raise ValueError(
            "Direction has zero norm"
        )

    return tuple(
        value / magnitude
        for value
        in vector
    )


def _identity_directions(
    dimension: int,
) -> list[
    tuple[
        float,
        ...
    ]
]:
    return [
        tuple(
            1.0
            if index == axis
            else 0.0
            for index
            in range(
                dimension
            )
        )
        for axis
        in range(
            dimension
        )
    ]


def _physical_to_normalized(
    problem: OptimizationProblem,
    point: tuple[
        float,
        ...
    ],
) -> tuple[
    float,
    ...
]:
    return tuple(
        (
            float(value)
            - float(axis.minimum)
        )
        / axis.span
        for axis, value
        in zip(
            problem.axes,
            point,
        )
    )


def _normalized_to_physical(
    problem: OptimizationProblem,
    point: tuple[
        float,
        ...
    ],
) -> tuple[
    float,
    ...
]:
    return tuple(
        float(axis.minimum)
        + float(value)
        * axis.span
        for axis, value
        in zip(
            problem.axes,
            point,
        )
    )


def _alpha_bounds(
    point: tuple[
        float,
        ...
    ],
    direction: tuple[
        float,
        ...
    ],
) -> tuple[
    float,
    float,
]:
    lower = -math.inf
    upper = math.inf

    for coordinate, component in zip(
        point,
        direction,
    ):
        if abs(
            component
        ) <= 1e-15:
            continue

        first = (
            0.0
            - coordinate
        ) / component

        second = (
            1.0
            - coordinate
        ) / component

        local_lower = min(
            first,
            second,
        )

        local_upper = max(
            first,
            second,
        )

        lower = max(
            lower,
            local_lower,
        )

        upper = min(
            upper,
            local_upper,
        )

    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or upper < lower
    ):
        raise RuntimeError(
            "Could not construct bounded line-search interval"
        )

    return (
        lower,
        upper,
    )


def _point_along(
    point: tuple[
        float,
        ...
    ],
    direction: tuple[
        float,
        ...
    ],
    alpha: float,
) -> tuple[
    float,
    ...
]:
    return tuple(
        coordinate
        + alpha * component
        for coordinate, component
        in zip(
            point,
            direction,
        )
    )


def _line_grid(
    lower: float,
    upper: float,
    samples: int,
) -> tuple[
    float,
    ...
]:
    if math.isclose(
        lower,
        upper,
        abs_tol=1e-15,
    ):
        return (
            float(
                lower
            ),
        )

    values = [
        lower
        + index
        * (
            upper
            - lower
        )
        / (
            samples
            - 1
        )
        for index
        in range(
            samples
        )
    ]

    if (
        lower <= 0 <= upper
        and not any(
            math.isclose(
                value,
                0.0,
                abs_tol=1e-12,
            )
            for value
            in values
        )
    ):
        values.append(
            0.0
        )

    return tuple(
        sorted(
            set(
                round(
                    value,
                    15,
                )
                for value
                in values
            )
        )
    )


def _quadratic_vertex(
    first: tuple[
        float,
        float,
    ],
    second: tuple[
        float,
        float,
    ],
    third: tuple[
        float,
        float,
    ],
    *,
    maximize: bool,
) -> float | None:
    x1, y1 = first
    x2, y2 = second
    x3, y3 = third

    denominator = (
        (x1 - x2)
        * (x1 - x3)
        * (x2 - x3)
    )

    if abs(
        denominator
    ) <= 1e-18:
        return None

    a = (
        x3 * (y2 - y1)
        + x2 * (y1 - y3)
        + x1 * (y3 - y2)
    ) / denominator

    b = (
        x3 * x3 * (y1 - y2)
        + x2 * x2 * (y3 - y1)
        + x1 * x1 * (y2 - y3)
    ) / denominator

    if abs(
        a
    ) <= 1e-18:
        return None

    if maximize and a >= 0:
        return None

    if not maximize and a <= 0:
        return None

    vertex = (
        -b
        / (
            2.0
            * a
        )
    )

    if not (
        min(
            x1,
            x3,
        )
        < vertex
        < max(
            x1,
            x3,
        )
    ):
        return None

    return float(
        vertex
    )


class RobustConjugateDirectionOptimizer:
    """
    Online, derivative-free robust conjugate-direction optimizer.

    No historical training data are required.

    The optimizer operates only through the generic OptimizationProblem /
    EvaluationFunction interface and therefore has no knowledge of FLAVIA,
    Faraday cups, voltages, or RFQ hardware.
    """

    name = "rcds"
    version = "1.0"

    def __init__(
        self,
        policy: RCDSPolicy | None = None,
    ):
        self.policy = (
            policy
            if policy is not None
            else RCDSPolicy()
        )

    def optimize(
        self,
        problem: OptimizationProblem,
        evaluator: EvaluationFunction,
    ) -> OptimizationResult:
        policy = (
            self.policy
        )

        history: list[
            ObjectiveEvaluation
        ] = []

        learned_directions: list[
            tuple[
                float,
                ...
            ]
        ] = []

        cache: dict[
            tuple[
                float,
                ...
            ],
            ObjectiveEvaluation,
        ] = {}

        def cache_key(
            normalized_point,
        ):
            return tuple(
                round(
                    float(value),
                    14,
                )
                for value
                in normalized_point
            )

        def evaluate_normalized(
            normalized_point: tuple[
                float,
                ...
            ],
        ) -> ObjectiveEvaluation | None:
            key = cache_key(
                normalized_point
            )

            if key in cache:
                return cache[
                    key
                ]

            physical_point = (
                _normalized_to_physical(
                    problem,
                    normalized_point,
                )
            )

            if not problem.is_allowed(
                physical_point
            ):
                return None

            if (
                len(
                    history
                )
                >= policy.max_evaluations
            ):
                raise OptimizationBudgetExhausted

            evaluation = evaluator(
                physical_point
            )

            if len(
                evaluation.point
            ) != problem.dimension:
                raise ValueError(
                    "Evaluator returned wrong point dimension"
                )

            for requested, observed in zip(
                physical_point,
                evaluation.point,
            ):
                if not math.isclose(
                    float(requested),
                    float(observed),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "Evaluator returned an evaluation for a "
                        "different optimization point"
                    )

            history.append(
                evaluation
            )

            cache[
                key
            ] = evaluation

            return evaluation

        current_point = (
            _physical_to_normalized(
                problem,
                problem.initial_point,
            )
        )

        initial_evaluation = (
            evaluate_normalized(
                current_point
            )
        )

        if initial_evaluation is None:
            raise ValueError(
                "Initial point was unexpectedly rejected"
            )

        if not initial_evaluation.safe:
            raise ValueError(
                "Initial evaluation is not safe"
            )

        current_evaluation = (
            initial_evaluation
        )

        best_point = (
            current_point
        )

        best_evaluation = (
            current_evaluation
        )

        directions = (
            _identity_directions(
                problem.dimension
            )
        )

        def line_search(
            point: tuple[
                float,
                ...
            ],
            starting_evaluation: ObjectiveEvaluation,
            direction: tuple[
                float,
                ...
            ],
        ) -> tuple[
            tuple[
                float,
                ...
            ],
            ObjectiveEvaluation,
        ]:
            lower, upper = (
                _alpha_bounds(
                    point,
                    direction,
                )
            )

            lower = max(
                lower,
                -policy.line_half_width,
            )

            upper = min(
                upper,
                policy.line_half_width,
            )

            if upper < lower:
                return (
                    point,
                    starting_evaluation,
                )

            alpha_values = (
                _line_grid(
                    lower,
                    upper,
                    policy.line_samples,
                )
            )

            evaluated: list[
                tuple[
                    float,
                    ObjectiveEvaluation,
                    tuple[
                        float,
                        ...
                    ],
                ]
            ] = []

            line_best_point = (
                point
            )

            line_best_evaluation = (
                starting_evaluation
            )

            for alpha in (
                alpha_values
            ):
                candidate_point = (
                    _point_along(
                        point,
                        direction,
                        alpha,
                    )
                )

                candidate = (
                    evaluate_normalized(
                        candidate_point
                    )
                )

                if candidate is None:
                    continue

                evaluated.append(
                    (
                        float(alpha),
                        candidate,
                        candidate_point,
                    )
                )

                if problem.is_better(
                    candidate,
                    line_best_evaluation,
                ):
                    line_best_point = (
                        candidate_point
                    )

                    line_best_evaluation = (
                        candidate
                    )

            if (
                policy.parabolic_refinement
                and len(
                    evaluated
                ) >= 3
            ):
                ordered = sorted(
                    evaluated,
                    key=lambda item:
                        item[
                            0
                        ],
                )

                best_index = min(
                    range(
                        len(
                            ordered
                        )
                    ),
                    key=lambda index:
                        abs(
                            ordered[
                                index
                            ][
                                0
                            ]
                            - _alpha_for_point(
                                point,
                                direction,
                                line_best_point,
                            )
                        ),
                )

                if (
                    0
                    < best_index
                    < len(
                        ordered
                    ) - 1
                ):
                    left = (
                        ordered[
                            best_index - 1
                        ]
                    )

                    center = (
                        ordered[
                            best_index
                        ]
                    )

                    right = (
                        ordered[
                            best_index + 1
                        ]
                    )

                    vertex = (
                        _quadratic_vertex(
                            (
                                left[
                                    0
                                ],
                                left[
                                    1
                                ].value,
                            ),
                            (
                                center[
                                    0
                                ],
                                center[
                                    1
                                ].value,
                            ),
                            (
                                right[
                                    0
                                ],
                                right[
                                    1
                                ].value,
                            ),
                            maximize=(
                                problem.maximize
                            ),
                        )
                    )

                    if (
                        vertex is not None
                        and lower
                        <= vertex
                        <= upper
                    ):
                        vertex_point = (
                            _point_along(
                                point,
                                direction,
                                vertex,
                            )
                        )

                        vertex_evaluation = (
                            evaluate_normalized(
                                vertex_point
                            )
                        )

                        if (
                            vertex_evaluation
                            is not None
                            and problem.is_better(
                                vertex_evaluation,
                                line_best_evaluation,
                            )
                        ):
                            line_best_point = (
                                vertex_point
                            )

                            line_best_evaluation = (
                                vertex_evaluation
                            )

            return (
                line_best_point,
                line_best_evaluation,
            )

        iterations_completed = 0

        stall_count = 0

        termination_reason = (
            "max_iterations"
        )

        try:
            for iteration in range(
                1,
                policy.max_iterations + 1,
            ):
                iterations_completed = (
                    iteration
                )

                iteration_start_point = (
                    current_point
                )

                iteration_start_evaluation = (
                    current_evaluation
                )

                largest_raw_gain = (
                    -math.inf
                )

                replace_direction_index = 0

                for direction_index, direction in enumerate(
                    directions
                ):
                    before = (
                        current_evaluation
                    )

                    (
                        candidate_point,
                        candidate_evaluation,
                    ) = line_search(
                        current_point,
                        current_evaluation,
                        direction,
                    )

                    if problem.is_better(
                        candidate_evaluation,
                        current_evaluation,
                    ):
                        current_point = (
                            candidate_point
                        )

                        current_evaluation = (
                            candidate_evaluation
                        )

                    raw_gain = (
                        current_evaluation.value
                        - before.value
                    )

                    if not problem.maximize:
                        raw_gain = (
                            -raw_gain
                        )

                    if (
                        raw_gain
                        > largest_raw_gain
                    ):
                        largest_raw_gain = (
                            raw_gain
                        )

                        replace_direction_index = (
                            direction_index
                        )

                    if problem.is_better(
                        current_evaluation,
                        best_evaluation,
                    ):
                        best_point = (
                            current_point
                        )

                        best_evaluation = (
                            current_evaluation
                        )

                displacement = tuple(
                    current
                    - start
                    for current, start
                    in zip(
                        current_point,
                        iteration_start_point,
                    )
                )

                displacement_norm = (
                    _norm(
                        displacement
                    )
                )

                if (
                    displacement_norm
                    >= policy.minimum_direction_norm
                ):
                    new_direction = (
                        _normalize_direction(
                            displacement
                        )
                    )

                    (
                        conjugate_point,
                        conjugate_evaluation,
                    ) = line_search(
                        current_point,
                        current_evaluation,
                        new_direction,
                    )

                    if problem.is_better(
                        conjugate_evaluation,
                        current_evaluation,
                    ):
                        current_point = (
                            conjugate_point
                        )

                        current_evaluation = (
                            conjugate_evaluation
                        )

                    directions[
                        replace_direction_index
                    ] = (
                        new_direction
                    )

                    learned_directions.append(
                        new_direction
                    )

                    if problem.is_better(
                        current_evaluation,
                        best_evaluation,
                    ):
                        best_point = (
                            current_point
                        )

                        best_evaluation = (
                            current_evaluation
                        )

                improved_iteration = (
                    problem.is_better(
                        current_evaluation,
                        iteration_start_evaluation,
                    )
                )

                if improved_iteration:
                    stall_count = 0
                else:
                    stall_count += 1

                if (
                    stall_count
                    >= policy.stall_iterations
                ):
                    termination_reason = (
                        "stalled"
                    )

                    break

        except OptimizationBudgetExhausted:
            termination_reason = (
                "max_evaluations"
            )

        return OptimizationResult(
            optimizer_name=(
                self.name
            ),
            initial_evaluation=(
                initial_evaluation
            ),
            best_evaluation=(
                best_evaluation
            ),
            history=tuple(
                history
            ),
            iterations=(
                iterations_completed
            ),
            termination_reason=(
                termination_reason
            ),
            optimizer_version=(
                self.version
            ),
            metadata={
                "axis_names": tuple(
                    axis.name
                    for axis
                    in problem.axes
                ),
                "learned_directions_normalized": tuple(
                    learned_directions
                ),
                "final_directions_normalized": tuple(
                    directions
                ),
            },
        )


def _alpha_for_point(
    origin: tuple[
        float,
        ...
    ],
    direction: tuple[
        float,
        ...
    ],
    point: tuple[
        float,
        ...
    ],
) -> float:
    numerator = sum(
        (
            target
            - source
        )
        * component
        for source, target, component
        in zip(
            origin,
            point,
            direction,
        )
    )

    denominator = sum(
        component
        * component
        for component
        in direction
    )

    if denominator <= 0:
        return 0.0

    return (
        numerator
        / denominator
    )
