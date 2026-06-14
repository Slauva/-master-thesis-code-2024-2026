import json
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("notebook_name", "required_source_tokens"),
    [
        (
            "3.0-torch-dataset-gpu.ipynb",
            ("TorchDataset", "collate_torch_samples", "Conv1d"),
        ),
        (
            "3.1-torch-preprocessed-dataset-gpu.ipynb",
            (
                "TorchPreprocessedDataset",
                "collate_torch_spectral_samples",
                "Conv2d",
                '"fft"',
                '"morlet"',
                '"superlet"',
                '"stft"',
            ),
        ),
    ],
)
def test_torch_notebook_is_executed_on_cuda_for_both_recording_families(
    notebook_name: str,
    required_source_tokens: tuple[str, ...],
) -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    outputs = "\n".join(
        json.dumps(output, sort_keys=True)
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
    assert "torch.cuda.is_available()" in source
    assert 'torch.device("cuda")' in source
    assert '"status": "CUDA_VERIFIED"' in source
    assert "CUDA_VERIFIED" in outputs
    assert "Data_Train" in source
    assert "Data_Pattern" in source
    assert '"exec"' in source
    assert '"patt"' in source
    assert all(token in source for token in required_source_tokens)
