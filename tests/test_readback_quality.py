import pytest

from sirius.readback_quality import (
    MissingReadbackQualityError,
    ReadbackQualityPolicy,
    RejectedReadbackQualityError,
)


def policy():
    # TEST VALUE ONLY.
    # This does not define FLAVIA production semantics.
    return ReadbackQualityPolicy(
        accepted_values=(
            "test-valid",
        )
    )


def test_explicit_accepted_quality_passes():
    result = policy().require_accepted(
        "test-valid",
        parameter_name="lens2_voltage_v",
    )

    assert result == "test-valid"


def test_matching_is_case_insensitive_by_default():
    assert policy().accepts(
        "TEST-VALID"
    )


def test_unknown_quality_fails_closed():
    with pytest.raises(
        RejectedReadbackQualityError
    ):
        policy().require_accepted(
            "test-bad",
            parameter_name="lens2_voltage_v",
        )


def test_missing_quality_fails_closed():
    with pytest.raises(
        MissingReadbackQualityError
    ):
        policy().require_accepted(
            None,
            parameter_name="lens2_voltage_v",
        )


def test_empty_quality_fails_closed():
    with pytest.raises(
        MissingReadbackQualityError
    ):
        policy().require_accepted(
            "   ",
            parameter_name="lens2_voltage_v",
        )


def test_policy_requires_explicit_allowlist():
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        ReadbackQualityPolicy(
            accepted_values=()
        )


def test_duplicate_values_after_normalization_are_rejected():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        ReadbackQualityPolicy(
            accepted_values=(
                "VALID",
                "valid",
            )
        )


def test_allow_missing_policy_is_not_strict():
    weak = ReadbackQualityPolicy(
        accepted_values=(
            "test-valid",
        ),
        allow_missing=True,
    )

    assert weak.strict is False

    assert weak.accepts(
        None
    )