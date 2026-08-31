import math

import pytest

from sirius.comparison import (
    ComparisonPolicy,
)
from sirius.optimizer_api import (
    ObjectiveEvaluation,
    OptimizationAxis,
    OptimizationProblem,
    comparison_policy_comparator,
    uncertainty_aware_comparator,
)
from sirius.rcds_optimizer import (
    RCDSPolicy,
    RobustConjugateDirectionOptimizer,
)


def test_uncertainty_comparator_requires_significant_improvement():
    compare = (
        uncertainty_aware_comparator(
            maximize=True,
            sigma_factor=2.0,
        )
    )

    incumbent = ObjectiveEvaluation(
        point=(0.0,),
        value=1.0,
        sem=0.1,
    )

    small_change = ObjectiveEvaluation(
        point=(0.1,),
        value=1.1,
        sem=0.1,
    )

    large_change = ObjectiveEvaluation(
        point=(0.2,),
        value=1.5,
        sem=0.1,
    )

    assert (
        compare(
            small_change,
            incumbent,
        )
        is False
    )

    assert (
        compare(
            large_change,
            incumbent,
        )
        is True
    )


def test_rcds_finds_rotated_two_dimensional_optimum():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
            OptimizationAxis(
                "y",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.15,
            0.85,
        ),
        maximize=True,
        comparison=(
            uncertainty_aware_comparator(
                maximize=True,
                sigma_factor=0.0,
            )
        ),
    )

    optimum = (
        0.75,
        0.25,
    )

    def evaluator(
        point,
    ):
        x, y = point

        # Rotated elliptical objective.
        u = (
            (
                x
                - optimum[
                    0
                ]
            )
            + 0.65
            * (
                y
                - optimum[
                    1
                ]
            )
        )

        v = (
            -0.65
            * (
                x
                - optimum[
                    0
                ]
            )
            + (
                y
                - optimum[
                    1
                ]
            )
        )

        value = (
            1.0
            - 2.0
            * u * u
            - 0.6
            * v * v
        )

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=value,
            sem=0.0,
        )

    optimizer = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=8,
                max_evaluations=250,
                line_samples=7,
                line_half_width=1.0,
                stall_iterations=2,
            )
        )
    )

    result = optimizer.optimize(
        problem,
        evaluator,
    )

    x, y = (
        result.best_evaluation.point
    )

    assert x == pytest.approx(
        optimum[
            0
        ],
        abs=0.08,
    )

    assert y == pytest.approx(
        optimum[
            1
        ],
        abs=0.08,
    )

    assert (
        result.best_evaluation.value
        > result.initial_evaluation.value
    )


def test_rcds_never_evaluates_static_unsafe_candidate():
    evaluated = []

    def safe(
        point,
    ):
        x, y = point

        return (
            x + y
            <= 1.10
            + 1e-12
        )

    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
            OptimizationAxis(
                "y",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.2,
            0.2,
        ),
        maximize=True,
        safety_predicate=safe,
    )

    def evaluator(
        point,
    ):
        assert safe(
            point
        )

        evaluated.append(
            tuple(
                point
            )
        )

        x, y = point

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                x + y
            ),
            sem=0.0,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=5,
                max_evaluations=150,
                line_samples=7,
                line_half_width=1.0,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    assert evaluated

    assert all(
        safe(
            point
        )
        for point
        in evaluated
    )

    assert (
        sum(
            result.best_evaluation.point
        )
        <= 1.10
        + 1e-12
    )

    skipped = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "candidate_skipped"
        )
    )

    assert skipped

    assert all(
        event[
            "reason"
        ]
        == "problem_rejected"
        for event
        in skipped
    )

    assert all(
        not safe(
            event[
                "physical_point"
            ]
        )
        for event
        in skipped
    )

    for event in skipped:
        assert (
            len(
                event[
                    "normalized_point"
                ]
            )
            == problem.dimension
        )

        assert (
            len(
                event[
                    "physical_point"
                ]
            )
            == problem.dimension
        )

        assert all(
            0.0
            <= coordinate
            <= 1.0
            for coordinate
            in event[
                "normalized_point"
            ]
        )


def test_differently_scaled_physical_axes_are_normalized():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "hv_v",
                0.0,
                6000.0,
            ),
            OptimizationAxis(
                "steerer_v",
                -250.0,
                250.0,
            ),
        ),
        initial_point=(
            500.0,
            -200.0,
        ),
        maximize=True,
    )

    optimum = (
        4200.0,
        75.0,
    )

    def evaluator(
        point,
    ):
        hv, steering = point

        normalized_hv_error = (
            (
                hv
                - optimum[
                    0
                ]
            )
            / 6000.0
        )

        normalized_steering_error = (
            (
                steering
                - optimum[
                    1
                ]
            )
            / 500.0
        )

        value = (
            1.0
            - normalized_hv_error ** 2
            - normalized_steering_error ** 2
        )

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=value,
            sem=0.0,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=8,
                max_evaluations=250,
                line_samples=7,
                line_half_width=1.0,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    hv, steering = (
        result.best_evaluation.point
    )

    assert hv == pytest.approx(
        optimum[
            0
        ],
        abs=350.0,
    )

    assert steering == pytest.approx(
        optimum[
            1
        ],
        abs=30.0,
    )


def test_evaluation_budget_is_hard_limit():
    calls = {
        "n": 0
    }

    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
            OptimizationAxis(
                "y",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.5,
            0.5,
        ),
    )

    def evaluator(
        point,
    ):
        calls[
            "n"
        ] += 1

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=sum(
                point
            ),
            sem=0.0,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=20,
                max_evaluations=20,
                line_samples=7,
                line_half_width=1.0,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    assert calls[
        "n"
    ] <= 20

    assert (
        result.evaluations
        <= 20
    )


def test_invalid_initial_point_is_rejected():
    with pytest.raises(
        ValueError
    ):
        OptimizationProblem(
            axes=(
                OptimizationAxis(
                    "x",
                    0.0,
                    1.0,
                ),
            ),
            initial_point=(
                2.0,
            ),
        )


def test_optimizer_requires_no_training_data():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.1,
        ),
    )

    def evaluator(
        point,
    ):
        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                -(
                    x
                    - 0.7
                ) ** 2
            ),
            sem=0.0,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=4,
                line_half_width=1.0,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    assert (
        result.best_evaluation.value
        > result.initial_evaluation.value
    )


def test_comparison_policy_comparator_uses_canonical_uncertainty_semantics():
    compare = comparison_policy_comparator(
        policy=ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_absolute_improvement_a=0.0,
            minimum_relative_improvement=0.0,
        )
    )

    incumbent = ObjectiveEvaluation(
        point=(0.0,),
        value=1.0,
        sem=0.1,
    )

    statistically_indistinguishable = (
        ObjectiveEvaluation(
            point=(0.1,),
            value=1.1,
            sem=0.1,
        )
    )

    clear_improvement = (
        ObjectiveEvaluation(
            point=(0.2,),
            value=1.5,
            sem=0.1,
        )
    )

    assert (
        compare(
            statistically_indistinguishable,
            incumbent,
        )
        is False
    )

    assert (
        compare(
            clear_improvement,
            incumbent,
        )
        is True
    )


def test_comparison_policy_comparator_preserves_noise_floor_semantics():
    compare = comparison_policy_comparator(
        policy=ComparisonPolicy(
            uncertainty_multiple=2.0,
            minimum_absolute_improvement_a=0.0,
            minimum_relative_improvement=0.0,
        )
    )

    below_noise = ObjectiveEvaluation(
        point=(0.0,),
        value=0.001,
        sem=0.0001,
        below_noise_floor=True,
    )

    detected_beam = ObjectiveEvaluation(
        point=(0.1,),
        value=0.01,
        sem=0.0001,
        below_noise_floor=False,
    )

    another_below_noise = ObjectiveEvaluation(
        point=(0.2,),
        value=0.002,
        sem=0.0001,
        below_noise_floor=True,
    )

    assert (
        compare(
            detected_beam,
            below_noise,
        )
        is True
    )

    assert (
        compare(
            another_below_noise,
            below_noise,
        )
        is False
    )


def test_rcds_does_not_accept_statistically_indistinguishable_gain():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.5,
        ),
        maximize=True,
        comparison=(
            comparison_policy_comparator(
                policy=ComparisonPolicy(
                    uncertainty_multiple=2.0,
                    minimum_absolute_improvement_a=0.0,
                    minimum_relative_improvement=0.0,
                )
            )
        ),
    )

    def evaluator(
        point,
    ):
        x = point[0]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                1.0
                + 0.05
                * (
                    x
                    - 0.5
                )
            ),
            sem=0.10,
        )

    optimizer = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=3,
                max_evaluations=50,
                line_samples=7,
                line_half_width=1.0,
                stall_iterations=2,
            )
        )
    )

    result = optimizer.optimize(
        problem,
        evaluator,
    )

    assert (
        result.best_evaluation.point
        == pytest.approx(
            (0.5,)
        )
    )

    assert (
        result.best_evaluation.value
        == pytest.approx(
            result.initial_evaluation.value
        )
    )

def test_rcds_exports_learned_directions():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
            OptimizationAxis(
                "y",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.15,
            0.85,
        ),
        maximize=True,
    )

    optimum = (
        0.75,
        0.25,
    )

    def evaluator(
        point,
    ):
        x, y = point

        u = (
            (
                x
                - optimum[0]
            )
            + 0.65
            * (
                y
                - optimum[1]
            )
        )

        v = (
            -0.65
            * (
                x
                - optimum[0]
            )
            + (
                y
                - optimum[1]
            )
        )

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                1.0
                - 2.0
                * u
                * u
                - 0.6
                * v
                * v
            ),
            sem=0.0,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=8,
                max_evaluations=250,
                line_samples=7,
                line_half_width=1.0,
                stall_iterations=2,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    assert (
        result.optimizer_version
        == "1.0"
    )

    assert (
        result.metadata[
            "axis_names"
        ]
        == (
            "x",
            "y",
        )
    )

    learned = result.metadata[
        "learned_directions_normalized"
    ]

    final_directions = result.metadata[
        "final_directions_normalized"
    ]

    assert learned

    assert (
        len(
            final_directions
        )
        == problem.dimension
    )

    for direction in learned:
        assert (
            len(
                direction
            )
            == problem.dimension
        )

        norm = math.sqrt(
            sum(
                component
                * component
                for component
                in direction
            )
        )

        assert norm == pytest.approx(
            1.0,
            abs=1e-12,
        )

    for direction in final_directions:
        assert (
            len(
                direction
            )
            == problem.dimension
        )

        norm = math.sqrt(
            sum(
                component
                * component
                for component
                in direction
            )
        )

        assert norm == pytest.approx(
            1.0,
            abs=1e-12,
        )

    direction_events = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "direction_replaced"
        )
    )

    assert (
        len(
            direction_events
        )
        == len(
            learned
        )
    )

    for event, direction in zip(
        direction_events,
        learned,
    ):
        assert event[
            "new_direction"
        ] == pytest.approx(
            direction
        )

        assert (
            0
            <= event[
                "direction_index"
            ]
            < problem.dimension
        )

        assert (
            event[
                "iteration"
            ]
            >= 1
        )

        assert (
            event[
                "displacement_norm"
            ]
            > 0.0
        )


def test_rcds_trace_groundwork_records_real_evaluations():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "scaled",
                10.0,
                20.0,
            ),
        ),
        initial_point=(
            12.0,
        ),
        maximize=True,
    )

    def evaluator(
        point,
    ):
        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                -(
                    x
                    - 14.0
                ) ** 2
            ),
            sem=0.05,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=3,
                max_evaluations=50,
                line_samples=5,
                line_half_width=1.0,
                stall_iterations=2,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    trace = result.metadata[
        "trace"
    ]

    assert isinstance(
        trace,
        tuple,
    )

    assert (
        trace[
            0
        ][
            "event_type"
        ]
        == "optimizer_started"
    )

    assert (
        trace[
            -1
        ][
            "event_type"
        ]
        == "optimizer_terminated"
    )

    started = trace[
        0
    ]

    assert (
        started[
            "optimizer_name"
        ]
        == result.optimizer_name
    )

    assert (
        started[
            "optimizer_version"
        ]
        == result.optimizer_version
    )

    assert (
        started[
            "axis_names"
        ]
        == (
            "scaled",
        )
    )

    assert started[
        "initial_normalized_point"
    ] == pytest.approx(
        (
            0.2,
        )
    )

    assert started[
        "initial_physical_point"
    ] == pytest.approx(
        (
            12.0,
        )
    )

    evaluation_events = tuple(
        event
        for event
        in trace
        if (
            event[
                "event_type"
            ]
            == "evaluation"
        )
    )

    assert (
        len(
            evaluation_events
        )
        == result.evaluations
    )

    first_evaluation = (
        evaluation_events[
            0
        ]
    )

    assert first_evaluation[
        "normalized_point"
    ] == pytest.approx(
        (
            0.2,
        )
    )

    assert first_evaluation[
        "physical_point"
    ] == pytest.approx(
        (
            12.0,
        )
    )

    assert (
        first_evaluation[
            "objective"
        ]
        == pytest.approx(
            result.initial_evaluation.value
        )
    )

    assert (
        first_evaluation[
            "uncertainty"
        ]
        == pytest.approx(
            0.05
        )
    )

    assert (
        first_evaluation[
            "safe"
        ]
        is True
    )

    assert (
        first_evaluation[
            "below_noise_floor"
        ]
        is False
    )

    terminated = trace[
        -1
    ]

    assert (
        terminated[
            "termination_reason"
        ]
        == result.termination_reason
    )

    assert (
        terminated[
            "evaluations"
        ]
        == result.evaluations
    )

    assert (
        terminated[
            "iterations"
        ]
        == result.iterations
    )

    assert terminated[
        "physical_best_point"
    ] == pytest.approx(
        result.best_evaluation.point
    )

    assert (
        terminated[
            "objective"
        ]
        == pytest.approx(
            result.best_evaluation.value
        )
    )


def test_rcds_trace_records_grid_line_search_decisions():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.2,
        ),
        maximize=True,
    )

    def evaluator(
        point,
    ):
        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                -(
                    x
                    - 0.8
                ) ** 2
            ),
            sem=0.01,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=3,
                max_evaluations=50,
                line_samples=5,
                line_half_width=1.0,
                stall_iterations=2,
                parabolic_refinement=False,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    samples = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "line_search_sample"
        )
    )

    assert samples

    assert all(
        event[
            "source"
        ]
        == "grid"
        for event
        in samples
    )

    for event in samples:
        assert (
            len(
                event[
                    "normalized_candidate"
                ]
            )
            == 1
        )

        assert (
            len(
                event[
                    "physical_candidate"
                ]
            )
            == 1
        )

        assert (
            len(
                event[
                    "direction"
                ]
            )
            == 1
        )

        assert isinstance(
            event[
                "accepted"
            ],
            bool,
        )

        assert (
            event[
                "uncertainty"
            ]
            == pytest.approx(
                0.01
            )
        )

        assert (
            "incumbent_objective"
            in event
        )

        assert (
            "incumbent_uncertainty"
            in event
        )


def test_rcds_trace_records_parabolic_refinement_decision():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.2,
        ),
        maximize=True,
    )

    def evaluator(
        point,
    ):
        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                -(
                    x
                    - 0.53
                ) ** 2
            ),
            sem=0.0,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=1,
                max_evaluations=30,
                line_samples=5,
                line_half_width=0.5,
                stall_iterations=1,
                parabolic_refinement=True,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    samples = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "line_search_sample"
            and event[
                "source"
            ]
            == "parabolic_refinement"
        )
    )

    assert samples

    first = samples[
        0
    ]

    assert first[
        "normalized_candidate"
    ][
        0
    ] == pytest.approx(
        0.53,
        abs=1e-10,
    )

    assert first[
        "physical_candidate"
    ][
        0
    ] == pytest.approx(
        0.53,
        abs=1e-10,
    )

    assert first[
        "alpha"
    ] == pytest.approx(
        0.33,
        abs=1e-10,
    )

    assert first[
        "objective"
    ] == pytest.approx(
        0.0,
        abs=1e-14,
    )

    assert first[
        "uncertainty"
    ] == pytest.approx(
        0.0,
        abs=1e-14,
    )

    assert (
        first[
            "accepted"
        ]
        is True
    )

    assert (
        "incumbent_objective"
        in first
    )

    assert (
        "incumbent_uncertainty"
        in first
    )


def test_rcds_trace_brackets_line_searches():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "scaled",
                10.0,
                20.0,
            ),
        ),
        initial_point=(
            12.0,
        ),
        maximize=True,
    )

    def evaluator(
        point,
    ):
        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                -(
                    x
                    - 16.0
                ) ** 2
            ),
            sem=0.02,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=2,
                max_evaluations=100,
                line_samples=5,
                line_half_width=0.5,
                stall_iterations=2,
                parabolic_refinement=True,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    trace = result.metadata[
        "trace"
    ]

    starts = tuple(
        event
        for event
        in trace
        if (
            event[
                "event_type"
            ]
            == "line_search_started"
        )
    )

    completions = tuple(
        event
        for event
        in trace
        if (
            event[
                "event_type"
            ]
            == "line_search_completed"
        )
    )

    assert starts
    assert completions

    assert (
        len(
            starts
        )
        == len(
            completions
        )
    )

    for started, completed in zip(
        starts,
        completions,
    ):
        assert (
            len(
                started[
                    "normalized_origin"
                ]
            )
            == 1
        )

        assert (
            len(
                started[
                    "physical_origin"
                ]
            )
            == 1
        )

        assert (
            len(
                started[
                    "direction"
                ]
            )
            == 1
        )

        normalized = started[
            "normalized_origin"
        ][
            0
        ]

        physical = started[
            "physical_origin"
        ][
            0
        ]

        assert physical == pytest.approx(
            10.0
            + 10.0
            * normalized
        )

        assert (
            started[
                "alpha_lower"
            ]
            <= 0.0
            <= started[
                "alpha_upper"
            ]
        )

        assert completed[
            "direction"
        ] == pytest.approx(
            started[
                "direction"
            ]
        )

        assert (
            len(
                completed[
                    "normalized_best_point"
                ]
            )
            == 1
        )

        assert (
            len(
                completed[
                    "physical_best_point"
                ]
            )
            == 1
        )

        assert (
            completed[
                "grid_samples_evaluated"
            ]
            >= 1
        )

        assert (
            completed[
                "reason"
            ]
            == "completed"
        )

        assert isinstance(
            completed[
                "objective"
            ],
            float,
        )

        assert (
            completed[
                "uncertainty"
            ]
            == pytest.approx(
                0.02
            )
        )


def test_rcds_trace_records_cache_hits():
    evaluator_calls = []

    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.5,
        ),
        maximize=True,
    )

    def evaluator(
        point,
    ):
        evaluator_calls.append(
            tuple(
                point
            )
        )

        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                -(
                    x
                    - 0.8
                ) ** 2
            ),
            sem=0.02,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=1,
                max_evaluations=20,
                line_samples=3,
                line_half_width=0.5,
                stall_iterations=1,
                parabolic_refinement=False,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    cache_hits = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "cache_hit"
        )
    )

    assert cache_hits

    # Cache reuse must not count as another real evaluation.
    assert (
        len(
            evaluator_calls
        )
        == result.evaluations
    )

    initial_hits = tuple(
        event
        for event
        in cache_hits
        if event[
            "physical_point"
        ] == pytest.approx(
            (
                0.5,
            )
        )
    )

    assert initial_hits

    initial_hit = initial_hits[
        0
    ]

    assert initial_hit[
        "normalized_point"
    ] == pytest.approx(
        (
            0.5,
        )
    )

    assert initial_hit[
        "cache_key"
    ] == pytest.approx(
        (
            0.5,
        )
    )

    assert (
        initial_hit[
            "objective"
        ]
        == pytest.approx(
            result.initial_evaluation.value
        )
    )

    assert (
        initial_hit[
            "uncertainty"
        ]
        == pytest.approx(
            0.02
        )
    )

    assert (
        initial_hit[
            "safe"
        ]
        is True
    )

    assert (
        initial_hit[
            "below_noise_floor"
        ]
        is False
    )

    for event in cache_hits:
        assert (
            len(
                event[
                    "cache_key"
                ]
            )
            == problem.dimension
        )

        assert (
            len(
                event[
                    "normalized_point"
                ]
            )
            == problem.dimension
        )

        assert (
            len(
                event[
                    "physical_point"
                ]
            )
            == problem.dimension
        )


def test_rcds_trace_records_iteration_completion():
    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "scaled",
                10.0,
                20.0,
            ),
        ),
        initial_point=(
            12.0,
        ),
        maximize=True,
    )

    def evaluator(
        point,
    ):
        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=tuple(
                point
            ),
            value=(
                -(
                    x
                    - 16.0
                ) ** 2
            ),
            sem=0.02,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=3,
                max_evaluations=200,
                line_samples=5,
                line_half_width=0.5,
                stall_iterations=2,
                parabolic_refinement=False,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    events = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "iteration_completed"
        )
    )

    assert events

    assert (
        len(
            events
        )
        == result.iterations
    )

    assert tuple(
        event[
            "iteration"
        ]
        for event
        in events
    ) == tuple(
        range(
            1,
            result.iterations + 1,
        )
    )

    for event in events:
        assert (
            len(
                event[
                    "normalized_point"
                ]
            )
            == problem.dimension
        )

        assert (
            len(
                event[
                    "physical_point"
                ]
            )
            == problem.dimension
        )

        normalized = event[
            "normalized_point"
        ][
            0
        ]

        physical = event[
            "physical_point"
        ][
            0
        ]

        assert physical == pytest.approx(
            10.0
            + 10.0
            * normalized
        )

        assert isinstance(
            event[
                "improved"
            ],
            bool,
        )

        assert (
            event[
                "stall_count"
            ]
            >= 0
        )

        assert (
            event[
                "evaluations"
            ]
            >= 1
        )

        assert (
            event[
                "uncertainty"
            ]
            == pytest.approx(
                0.02
            )
        )

    assert (
        events[
            -1
        ][
            "evaluations"
        ]
        == result.evaluations
    )

    assert (
        events[
            -1
        ][
            "physical_point"
        ]
        == pytest.approx(
            result.best_evaluation.point
        )
    )

    assert (
        events[
            -1
        ][
            "objective"
        ]
        == pytest.approx(
            result.best_evaluation.value
        )
    )


def test_rcds_can_disable_cached_evaluation_reuse():
    evaluator_calls = []

    problem = OptimizationProblem(
        axes=(
            OptimizationAxis(
                "x",
                0.0,
                1.0,
            ),
        ),
        initial_point=(
            0.5,
        ),
        maximize=True,
    )

    def evaluator(
        point,
    ):
        point = tuple(
            point
        )

        evaluator_calls.append(
            point
        )

        x = point[
            0
        ]

        return ObjectiveEvaluation(
            point=point,
            value=(
                -(
                    x
                    - 0.8
                ) ** 2
            ),
            sem=0.02,
        )

    result = (
        RobustConjugateDirectionOptimizer(
            RCDSPolicy(
                max_iterations=1,
                max_evaluations=20,
                line_samples=3,
                line_half_width=0.5,
                stall_iterations=1,
                parabolic_refinement=False,
                reuse_cached_evaluations=False,
            )
        ).optimize(
            problem,
            evaluator,
        )
    )

    # The line grid contains alpha=0, so the current physical
    # point is deliberately measured again when reuse is off.
    repeated_initial = tuple(
        point
        for point
        in evaluator_calls
        if point == (
            0.5,
        )
    )

    assert (
        len(
            repeated_initial
        )
        >= 2
    )

    assert (
        len(
            evaluator_calls
        )
        == result.evaluations
    )

    cache_hits = tuple(
        event
        for event
        in result.metadata[
            "trace"
        ]
        if (
            event[
                "event_type"
            ]
            == "cache_hit"
        )
    )

    assert (
        cache_hits
        == ()
    )


def test_rcds_cache_reuse_policy_requires_bool():
    with pytest.raises(
        TypeError,
        match="reuse_cached_evaluations",
    ):
        RCDSPolicy(
            reuse_cached_evaluations=1,
        )
