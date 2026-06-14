import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("filename", "required_source", "validation_marker"),
    [
        (
            "4.0-classical-features.ipynb",
            ("Data_Pattern", '"patt"', "window_seconds"),
            "CLASSICAL_FEATURES_VERIFIED",
        ),
        (
            "4.1-local-patterns.ipynb",
            ("Data_Pattern", '"patt"', "compute_lndp_codes", "compute_lgp_codes"),
            "LOCAL_PATTERNS_VERIFIED",
        ),
    ],
)
def test_feature_notebook_is_executed_and_validated(
    filename: str,
    required_source: tuple[str, ...],
    validation_marker: str,
) -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / filename
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    output_text = "\n".join(
        "".join(output.get("text", []))
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert all(required_text in source for required_text in required_source)
    assert validation_marker in output_text
