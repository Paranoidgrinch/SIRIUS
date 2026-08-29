import ast
from pathlib import Path


ALLOWED_RAW_APPLY_STATE_IMPORTERS = {
    "safe_transition.py",
}


def _imports_raw_apply_state(
    path: Path,
) -> bool:
    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        ),
        filename=str(
            path
        ),
    )

    for node in ast.walk(
        tree
    ):
        if not isinstance(
            node,
            ast.ImportFrom,
        ):
            continue

        if (
            node.module
            != "sirius.transition"
        ):
            continue

        for alias in node.names:
            if (
                alias.name
                == "apply_state"
            ):
                return True

    return False


def test_only_safe_transition_may_import_raw_apply_state():
    root = Path(
        "src/sirius"
    )

    offenders = []

    for path in root.glob(
        "*.py"
    ):
        if (
            not _imports_raw_apply_state(
                path
            )
        ):
            continue

        if (
            path.name
            not in
            ALLOWED_RAW_APPLY_STATE_IMPORTERS
        ):
            offenders.append(
                path.name
            )

    assert offenders == [], (
        "Modules bypass safe_transition and import raw apply_state: "
        + ", ".join(
            sorted(
                offenders
            )
        )
    )


def test_safe_transition_is_the_raw_boundary():
    path = Path(
        "src/sirius/safe_transition.py"
    )

    assert (
        _imports_raw_apply_state(
            path
        )
        is True
    )