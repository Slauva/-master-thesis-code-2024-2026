import json
from pathlib import Path


def test_full_imagery_sweep_comparison_notebook_is_executed_and_validated() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "6.2-full-imagery-model-feature-sweep.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    output_text = "\n".join(
        "".join(output.get("text", []))
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    image_outputs = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
    )

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert all(
        required_text in source
        for required_text in (
            "## tl;dr",
            "## Data And Coverage",
            "## Protocol-Separated Leaders",
            "## Paired Subject-Cluster Bootstrap",
            "## Feature And Method Families",
            "## Limitations",
            "write_full_imagery_comparison_summary",
            "stage5_comparison_summary.json",
            "not multiplicity-adjusted",
            "never average cross-subject and within-subject",
            "geometric+random",
            "Cross-subject",
            "Within-subject",
        )
    )
    assert image_outputs >= 4
    assert "FULL_IMAGERY_SWEEP_COMPARISON_VERIFIED" in output_text
