import math

import pytest

from sirius.optimizer_api import (
    ObjectiveEvaluation,
    OptimizationAxis,
    OptimizationProblem,
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