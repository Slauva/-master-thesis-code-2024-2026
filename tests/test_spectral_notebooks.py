import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "notebook_name",
    [
        "2.1-fft.ipynb",
        "2.2-morlet.ipynb",
        "2.3-superlet.ipynb",
        "2.4-stft.ipynb",
        "2.5-spectral-methods-comparison.ipynb",
    ],
)
def test_spectral_notebook_is_executed_for_both_recording_families(
    notebook_name: str,
) -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    assert "Data_Train" in source
    assert "Data_Pattern" in source
    assert '"exec"' in source
    assert '"patt"' in source
