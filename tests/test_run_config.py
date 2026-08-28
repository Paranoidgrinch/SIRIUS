import json

import pytest

from sirius.hardware_safety import (
    HardwareSafetyConfig,
)
from sirius.run_config import (
    SiriusRunConfig,
)


def hardware():
    return HardwareSafetyConfig(
        cooler_end_max_step_v=100.0,
        qpt_max_step_v=125.0,
    )


def test_run_config_requires_valid_mass():
    with pytest.raises(
        ValueError
    ):
        SiriusRunConfig(
            mass_u=0.0,
            hardware_safety=hardware(),
        )


def test_run_config_requires_hardware_safety():
    with pytest.raises(
        TypeError
    ):
        SiriusRunConfig(
            mass_u=60.0,
            hardware_safety=None,
        )


def test_manifest_is_json_serializable():
    config = SiriusRunConfig(
        mass_u=60.0,
        hardware_safety=hardware(),
        metadata={
            "operator": "test",
        },
    )

    payload = (
        config.to_manifest_dict()
    )

    assert (
        payload[
            "mass_u"
        ]
        == pytest.approx(
            60.0
        )
    )

    assert (
        payload[
            "hardware_safety"
        ][
            "qpt_max_step_v"
        ]
        == pytest.approx(
            125.0
        )
    )

    assert isinstance(
        json.dumps(
            payload
        ),
        str,
    )