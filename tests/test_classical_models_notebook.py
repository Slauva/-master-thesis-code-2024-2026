import json
from pathlib import Path


def test_classical_models_training_notebook_is_executed_and_validated() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "5.2-classical-models-training.ipynb"
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
            "MODEL_CONFIG_PATHS",
            "PLANNED_MODEL_IDS",
            "PROTOCOLS",
            "REUSE_EXISTING = True",
            "execute_model_protocol",
            "load_model_run",
            "split_audit",
            "pipeline_count",
            "(141, 39)",
            "(81, 81)",
            "len(runs_df) == 27",
            "pure model",
        )
    )
    assert image_outputs >= 1
    assert "CLASSICAL_MODELS_TRAINING_VERIFIED" in output_text
