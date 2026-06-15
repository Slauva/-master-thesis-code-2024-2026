import json
from pathlib import Path


def test_logistic_regression_notebook_is_executed_and_validated() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "5.0-logistic-regression-random-pixels.ipynb"
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
            "f515948b6bf5af55",
            "load_experiment_run",
            "build_non_eeg_baselines",
            "evaluate_prediction_matrix",
            "bootstrap_subject_mean_balanced_accuracy",
            "selected_feature_names",
            "test_subject_ids",
        )
    )
    assert image_outputs >= 5
    assert "LOGISTIC_REGRESSION_RANDOM_PIXELS_VERIFIED" in output_text
