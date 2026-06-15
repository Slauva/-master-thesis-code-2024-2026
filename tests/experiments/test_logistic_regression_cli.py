import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.logistic_regression import (
    ProtocolWorkflowResult,
)
from experiments.logistic_regression.cli import build_parser, main


def _empty_summary() -> dict[str, object]:
    return {"runs": [], "combined": None}


def test_parser_accepts_run_and_evaluate_contracts() -> None:
    parser = build_parser()

    run_args = parser.parse_args(
        [
            "run",
            "--protocol",
            "within-subject",
            "--set",
            "grid_search.n_jobs=1",
            "--reuse-existing",
        ]
    )
    evaluate_args = parser.parse_args(
        ["evaluate", "run-a", "run-b", "--json"]
    )

    assert run_args.protocol == "within-subject"
    assert run_args.overrides == ["grid_search.n_jobs=1"]
    assert run_args.reuse_existing is True
    assert evaluate_args.run_dirs == [Path("run-a"), Path("run-b")]
    assert evaluate_args.json_output is True


def test_console_script_and_module_entry_points_are_equivalent() -> None:
    console_script = Path(sys.executable).with_name("logistic-regression")
    commands = (
        (str(console_script), "--help"),
        (sys.executable, "-m", "experiments.logistic_regression", "--help"),
    )

    outputs = [
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for command in commands
    ]

    assert outputs[0] == outputs[1]
    assert "{run,evaluate}" in outputs[0]


def test_run_command_executes_and_writes_new_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.logistic_regression import cli as cli_module

    def execute(*args: object, **kwargs: object) -> ProtocolWorkflowResult:
        assert kwargs["reuse_existing"] is False
        return ProtocolWorkflowResult(
            protocol="cross-subject",
            run_dirs=(tmp_path / "written-run",),
            runs=(),
            summary=_empty_summary(),
            reused=False,
        )

    monkeypatch.setattr(cli_module, "execute_evaluation_protocol", execute)

    exit_code = main(
        [
            "run",
            "--protocol",
            "cross-subject",
            "--set",
            f"artifacts.root={tmp_path / 'runs'}",
            "--json",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["reused"] is False


def test_reuse_existing_validates_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.logistic_regression import cli as cli_module

    def execute(*args: object, **kwargs: object) -> ProtocolWorkflowResult:
        assert kwargs["reuse_existing"] is True
        return ProtocolWorkflowResult(
            protocol="within-subject",
            run_dirs=(tmp_path / "run-a", tmp_path / "run-b"),
            runs=(),
            summary=_empty_summary(),
            reused=True,
        )

    monkeypatch.setattr(cli_module, "execute_evaluation_protocol", execute)
    exit_code = main(
        [
            "run",
            "--protocol",
            "within-subject",
            "--set",
            f"artifacts.root={tmp_path / 'runs'}",
            "--reuse-existing",
            "--json",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["reused"] is True


def test_existing_run_is_refused_without_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.logistic_regression import cli as cli_module

    def reject_existing(*args: object, **kwargs: object) -> object:
        raise FileExistsError("Experiment run already exists and is immutable")

    monkeypatch.setattr(
        cli_module,
        "execute_evaluation_protocol",
        reject_existing,
    )
    assert (
        main(
            [
                "run",
                "--protocol",
                "cross-subject",
                "--set",
                f"artifacts.root={tmp_path / 'runs'}",
            ]
        )
        == 1
    )
    assert "immutable" in capsys.readouterr().err


def test_evaluate_json_reads_schema_v1_reference_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = Path(
        "artifacts/experiments/logistic-regression/f515948b6bf5af55"
    )

    assert main(["evaluate", str(run_dir), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["artifact_schema_version"] == 1
    assert payload["runs"][0]["protocol"] == "cross-subject"
    assert payload["combined"] is None


def test_evaluate_failure_returns_nonzero_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["evaluate", str(tmp_path / "missing")]) == 1
    assert "manifest does not exist" in capsys.readouterr().err
