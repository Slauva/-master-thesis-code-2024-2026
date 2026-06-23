import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.random_imagery_torch.cli import build_parser, main
from experiments.random_imagery_torch.workflow import TorchProtocolWorkflowResult


def _empty_summary() -> dict[str, object]:
    return {"runs": [], "combined": None}


def test_random_imagery_torch_parser_accepts_all_commands() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "run",
            "--model",
            "eegnet-fft-multilabel",
            "--protocol",
            "within-subject",
            "--set",
            "training.maximum_epochs=1",
            "--reuse-existing",
        ]
    )
    matrix_plan = parser.parse_args(["matrix-plan", "--model", "eegnet-fft-multilabel", "--json"])
    matrix_run = parser.parse_args(
        [
            "matrix-run",
            "--model",
            "eegnet-fft-multilabel",
            "--protocol",
            "cross-subject",
            "--set",
            "training.maximum_epochs=1",
            "--no-reuse-existing",
            "--fail-fast",
            "--json",
        ]
    )
    evaluate = parser.parse_args(["evaluate", "run-a", "run-b", "--json"])
    compare = parser.parse_args(["compare", "run-a", "run-b", "--json"])

    assert run.model == "eegnet-fft-multilabel"
    assert run.protocol == "within-subject"
    assert run.reuse_existing is True
    assert matrix_plan.model == ["eegnet-fft-multilabel"]
    assert matrix_run.protocol == ["cross-subject"]
    assert matrix_run.no_reuse_existing is True
    assert matrix_run.fail_fast is True
    assert evaluate.run_dirs == [Path("run-a"), Path("run-b")]
    assert compare.run_dirs == [Path("run-a"), Path("run-b")]


def test_random_imagery_torch_console_and_module_entry_points_are_equivalent() -> None:
    console_script = Path(sys.executable).with_name("random-imagery-torch")
    commands = (
        (str(console_script), "--help"),
        (sys.executable, "-m", "experiments.random_imagery_torch", "--help"),
    )
    outputs = [
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
        for command in commands
    ]

    assert outputs[0] == outputs[1]
    assert "{run,matrix-plan,matrix-run,evaluate,compare}" in outputs[0]


def test_random_imagery_torch_run_command_reports_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.random_imagery_torch import cli as cli_module

    def execute(*args: object, **kwargs: object) -> TorchProtocolWorkflowResult:
        assert kwargs["reuse_existing"] is True
        return TorchProtocolWorkflowResult(
            model_id="eegnet-fft-multilabel",
            protocol="cross-subject",
            run_dirs=(tmp_path / "run",),
            runs=(),
            summary=_empty_summary(),
            reused=True,
        )

    monkeypatch.setattr(cli_module, "execute_torch_protocol", execute)
    exit_code = main(
        [
            "run",
            "--model",
            "eegnet-fft-multilabel",
            "--protocol",
            "cross-subject",
            "--set",
            f"artifacts.root={tmp_path / 'runs'}",
            "--reuse-existing",
            "--json",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["reused"] is True


def test_random_imagery_torch_matrix_run_command_delegates_and_reports_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.random_imagery_torch import cli as cli_module

    def execute(*args: object, **kwargs: object) -> dict[str, object]:
        specs = tuple(kwargs["specs"])  # type: ignore[arg-type]
        assert len(specs) == 1
        assert specs[0].model_id == "eegnet-fft-multilabel"
        assert kwargs["reuse_existing"] is False
        assert kwargs["continue_on_error"] is False
        assert kwargs["output_path"] == tmp_path / "summary.json"
        assert kwargs["failure_log_path"] == tmp_path / "failures.json"
        assert kwargs["extra_overrides"] == {"training": {"maximum_epochs": 1}}
        assert kwargs["verbose"] is False
        return {
            "planned_protocol_run_count": 1,
            "completed_protocol_run_count": 0,
            "failed_protocol_run_count": 1,
        }

    monkeypatch.setattr(cli_module, "execute_torch_matrix_sweep", execute)
    exit_code = main(
        [
            "matrix-run",
            "--model",
            "eegnet-fft-multilabel",
            "--protocol",
            "cross-subject",
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--failure-log",
            str(tmp_path / "failures.json"),
            "--set",
            "training.maximum_epochs=1",
            "--no-reuse-existing",
            "--fail-fast",
            "--json",
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["failed_protocol_run_count"] == 1


def test_random_imagery_torch_compare_command_and_failure_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.random_imagery_torch import cli as cli_module

    payload = {
        "protocol": "cross-subject",
        "direction": "cross-subject",
        "n_test_rows": 2,
        "runs": [],
    }
    monkeypatch.setattr(cli_module, "compare_torch_runs", lambda paths: payload)
    assert main(["compare", "a", "b", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload

    monkeypatch.setattr(
        cli_module,
        "compare_torch_runs",
        lambda paths: (_ for _ in ()).throw(ValueError("incompatible")),
    )
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b")]) == 1
    assert "incompatible" in capsys.readouterr().err
