from pathlib import Path


def test_python_sources_have_no_utf8_bom():
    roots = (
        Path("src/sirius"),
        Path("tests"),
    )

    offenders = []

    for root in roots:
        for path in root.rglob("*.py"):
            data = path.read_bytes()

            if data.startswith(
                b"\xef\xbb\xbf"
            ):
                offenders.append(
                    str(path)
                )

    assert offenders == [], (
        "Python source files contain UTF-8 BOM: "
        + ", ".join(
            sorted(
                offenders
            )
        )
    )