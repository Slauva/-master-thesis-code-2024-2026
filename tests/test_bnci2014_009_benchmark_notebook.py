import json
from pathlib import Path


def test_bnci2014_009_benchmark_notebook_is_executed_and_validated() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "7.3-bnci2014-009-benchmark.ipynb"
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
            "## Source Artifacts",
            "## Contract Checks",
            "## Aggregate Ranking",
            "## Family View",
            "## Held-Out Subject Variability",
            "## Target Recall And Score Metrics",
            "## Aggregate Confusion Matrices",
            "## Limitations",
            "BNCI2014_009_BENCHMARK_VERIFIED",
            "single post-stimulus P300 epoch",
            "leave-one-subject-out",
            "morlet",
            "superlet",
            "stft",
            "raw_spectral_split_aligned",
            "stage7_benchmark_summary.json",
        )
    )
    assert image_outputs >= 4
    assert "BNCI2014_009_BENCHMARK_VERIFIED" in output_text
