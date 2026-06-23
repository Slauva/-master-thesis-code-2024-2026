import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.random_imagery.cli import build_parser, main
from experiments.random_imagery.workflow import ModelProtocolWorkflowResult


def _empty_summary() -> dict[str, object]:
    return {"runs": [], "combined": None}


def test_random_imagery_parser_accepts_all_commands() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "run",
            "--model",
            "ridge-regression-independent",
            "--protocol",
            "within-subject",
            "--set",
            "grid_search.n_jobs=1",
            "--reuse-existing",
        ]
    )
    evaluate = parser.parse_args(["evaluate", "run-a", "run-b", "--json"])
    compare = parser.parse_args(["compare", "run-a", "run-b", "--json"])
    matrix = parser.parse_args(
        [
            "matrix-plan",
            "--model",
            "logistic-regression-independent",
            "--feature-family",
            "time+spectral",
            "--protocol",
            "cross-subject",
            "--json",
        ]
    )
    matrix_run = parser.parse_args(
        [
            "matrix-run",
            "--model",
            "pls-regression-multioutput",
            "--feature-family",
            "lbp",
            "--protocol",
            "within-subject",
            "--fail-fast",
            "--no-reuse-existing",
        ]
    )

    assert run.model == "ridge-regression-independent"
    assert run.protocol == "within-subject"
    assert run.reuse_existing is True
    assert evaluate.run_dirs == [Path("run-a"), Path("run-b")]
    assert compare.run_dirs == [Path("run-a"), Path("run-b")]
    assert matrix.model == ["logistic-regression-independent"]
    assert matrix.feature_family == ["time+spectral"]
    assert matrix.protocol == ["cross-subject"]
    assert matrix_run.model == ["pls-regression-multioutput"]
    assert matrix_run.feature_family == ["lbp"]
    assert matrix_run.protocol == ["within-subject"]
    assert matrix_run.fail_fast is True
    assert matrix_run.no_reuse_existing is True


def test_random_imagery_console_and_module_entry_points_are_equivalent() -> None:
    console_script = Path(sys.executable).with_name("random-imagery-models")
    commands = (
        (str(console_script), "--help"),
        (sys.executable, "-m", "experiments.random_imagery", "--help"),
    )
    outputs = [
        subprocess.run(command, check=True, capture_output=True, text=True).stdout
        for command in commands
    ]

    assert outputs[0] == outputs[1]
    assert "{run,matrix-plan,matrix-run,evaluate,compare}" in outputs[0]


def test_random_imagery_run_command_reports_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.random_imagery import cli as cli_module

    def execute(*args: object, **kwargs: object) -> ModelProtocolWorkflowResult:
        assert kwargs["reuse_existing"] is True
        return ModelProtocolWorkflowResult(
            model_id="ridge-regression-independent",
            protocol="cross-subject",
            run_dirs=(tmp_path / "run",),
            runs=(),
            summary=_empty_summary(),
            reused=True,
        )

    monkeypatch.setattr(cli_module, "execute_model_protocol", execute)
    exit_code = main(
        [
            "run",
            "--model",
            "ridge-regression-independent",
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


def test_random_imagery_matrix_plan_command_is_enumeration_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.random_imagery import cli as cli_module

    def fail_if_training(*args: object, **kwargs: object) -> object:
        raise AssertionError("matrix-plan must not execute training")

    monkeypatch.setattr(cli_module, "execute_model_protocol", fail_if_training)
    exit_code = main(
        [
            "matrix-plan",
            "--model",
            "logistic-regression-independent",
            "--feature-family",
            "time",
            "--protocol",
            "within-subject",
            "--artifact-root",
            str(tmp_path),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["run_count"] == 1
    assert payload["expected_direction_run_count"] == 2
    assert payload["runs"][0]["runner"] == "logistic-regression"
    assert payload["runs"][0]["overrides"]["dataset"]["pattern_type"] is None
    assert payload["runs"][0]["overrides"]["feature_screening"]["candidates"] == [["time"]]


def test_random_imagery_matrix_run_command_delegates_to_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.random_imagery import cli as cli_module

    calls: list[dict[str, object]] = []

    def execute(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            "complete": True,
            "planned_protocol_run_count": 1,
            "completed_protocol_run_count": 1,
            "failed_protocol_run_count": 0,
            "completed_direction_run_count": 1,
            "results": [],
            "failures": [],
        }

    monkeypatch.setattr(cli_module, "execute_classical_matrix_sweep", execute)
    exit_code = main(
        [
            "matrix-run",
            "--model",
            "logistic-regression-independent",
            "--feature-family",
            "time",
            "--protocol",
            "cross-subject",
            "--artifact-root",
            str(tmp_path / "runs"),
            "--summary-output",
            str(tmp_path / "summary.json"),
            "--failure-log",
            str(tmp_path / "failures.json"),
            "--set",
            "bootstrap_iterations=10",
        ]
    )

    assert exit_code == 0
    assert "Completed protocol runs: 1 / 1" in capsys.readouterr().out
    assert len(calls) == 1
    assert calls[0]["reuse_existing"] is True
    assert calls[0]["continue_on_error"] is True
    assert calls[0]["output_path"] == tmp_path / "summary.json"
    assert calls[0]["failure_log_path"] == tmp_path / "failures.json"
    assert calls[0]["extra_overrides"] == {"bootstrap_iterations": 10}
    assert calls[0]["verbose"] is True
    specs = calls[0]["specs"]
    assert len(specs) == 1
    assert specs[0].model_id == "logistic-regression-independent"


def test_random_imagery_compare_command_and_failure_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments.random_imagery import cli as cli_module

    payload = {
        "protocol": "cross-subject",
        "direction": "cross-subject",
        "n_test_rows": 2,
        "runs": [],
    }
    monkeypatch.setattr(cli_module, "compare_runs", lambda paths: payload)
    assert main(["compare", "a", "b", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload

    monkeypatch.setattr(
        cli_module,
        "compare_runs",
        lambda paths: (_ for _ in ()).throw(ValueError("incompatible")),
    )
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b")]) == 1
    assert "incompatible" in capsys.readouterr().err
