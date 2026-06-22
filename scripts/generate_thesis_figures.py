from __future__ import annotations

# ruff: noqa: E402
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-thesis-figures")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THESIS_IMAGES = PROJECT_ROOT.parent / "latex" / "images"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

from experiments.logistic_regression.artifacts import load_evaluation_run
from experiments.random_imagery.artifacts import load_model_run
from experiments.random_imagery.metrics import evaluate_prediction_matrix
from experiments.random_imagery.registry import (
    PLANNED_MODEL_IDS,
    REFERENCE_MODEL_ID,
    get_model_spec,
)
from experiments.random_imagery_torch.artifacts import load_torch_run
from experiments.random_imagery_torch.config import PRIMARY_TORCH_MODEL_IDS, parse_torch_model_id

REFERENCE_RUN_DIRS = {
    "cross-subject": (PROJECT_ROOT / "artifacts/experiments/logistic-regression/4fcdf3c4fa5ef75a",),
    "within-subject": (
        PROJECT_ROOT / "artifacts/experiments/logistic-regression/ea7f8aa10a39cea0",
        PROJECT_ROOT / "artifacts/experiments/logistic-regression/0ab4cb2a7512ab19",
    ),
}
CLASSICAL_ROOT = PROJECT_ROOT / "artifacts/experiments/random-imagery"
TORCH_ROOT = PROJECT_ROOT / "artifacts/experiments/random-imagery-torch"
RECONSTRUCTION_RUN = (
    TORCH_ROOT / "shallow-convnet-morlet-multilabel/678f75c694c69eb2/arrays"
)
DIRECTION_ORDER = {
    "cross-subject": ("cross-subject",),
    "within-subject": ("trial-1-to-trial-2", "trial-2-to-trial-1"),
}
PROTOCOL_LABELS = {
    "cross-subject": "cross-subject",
    "within-subject": "bidirectional cross-trial",
}
N_RESAMPLES = 2_000
RANDOM_STATE = 42
METRIC_COLUMNS = (
    "mean_balanced_accuracy",
    "mean_score_mse",
    "mean_sample_iou",
    "micro_iou",
    "hamming_loss",
    "exact_match_accuracy",
)
FAMILY_COLORS = {
    "Reference": "#222222",
    "Classical": "#3E6FA8",
    "Torch": "#C86B35",
}


@dataclass(frozen=True, slots=True)
class CombinedArrays:
    model_id: str
    label: str
    family: str
    task: str
    topology: str
    score_semantics: str
    test_sample_keys: tuple[tuple[int, int, int], ...]
    targets: np.ndarray
    scores: np.ndarray
    predictions: np.ndarray
    subject_ids: np.ndarray
    metadata: dict[str, object]
    baseline_scores: dict[str, np.ndarray] | None = None
    baseline_predictions: dict[str, np.ndarray] | None = None


def main() -> None:
    THESIS_IMAGES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )

    comparison_df, ba_contrasts = build_comparison_frames()
    plot_rankings(comparison_df)
    plot_deltas(ba_contrasts)
    plot_reconstructions()
    plot_pipeline()
    plot_architectures()
    verify_key_numbers(comparison_df, ba_contrasts)


def build_comparison_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol_arrays = {protocol: load_protocol_arrays(protocol) for protocol in DIRECTION_ORDER}
    comparison_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []

    for protocol, arrays_list in protocol_arrays.items():
        reference = arrays_list[0]
        rows = build_bootstrap_rows(reference.targets, reference.subject_ids)
        reference_samples = bootstrap_metric_matrix(reference, rows)
        reference_metrics = evaluate_prediction_matrix(
            reference.targets,
            reference.predictions,
            reference.scores,
        )
        reference_values = metric_vector(reference_metrics)
        raw_p_values: list[float] = []
        raw_p_indices: list[int] = []

        for arrays in arrays_list:
            metrics = evaluate_prediction_matrix(arrays.targets, arrays.predictions, arrays.scores)
            values = metric_vector(metrics)
            samples = bootstrap_metric_matrix(arrays, rows)
            ba_low, ba_high = np.quantile(samples[:, 0], (0.025, 0.975), method="linear")
            comparison_rows.append(
                {
                    "protocol": protocol,
                    "protocol_label": PROTOCOL_LABELS[protocol],
                    "model_id": arrays.model_id,
                    "model": arrays.label,
                    "family": arrays.family,
                    "task": arrays.task,
                    "topology": arrays.topology,
                    "score_semantics": arrays.score_semantics,
                    "balanced_accuracy": values[0],
                    "ba_ci_low": float(ba_low),
                    "ba_ci_high": float(ba_high),
                    "score_mse": values[1],
                    "sample_iou": values[2],
                    "micro_iou": values[3],
                    "hamming_loss": values[4],
                    "exact_match": values[5],
                    "n_test_rows": int(arrays.targets.shape[0]),
                    "n_subjects": int(np.unique(arrays.subject_ids).size),
                }
            )
            if arrays.model_id == REFERENCE_MODEL_ID:
                continue

            ba_improvement_samples = samples[:, 0] - reference_samples[:, 0]
            raw_p = min(
                1.0,
                2.0
                * min(
                    float(np.mean(ba_improvement_samples <= 0.0)),
                    float(np.mean(ba_improvement_samples >= 0.0)),
                ),
            )
            raw_p_indices.append(len(contrast_rows))
            raw_p_values.append(raw_p)

            for metric_index, metric in enumerate(METRIC_COLUMNS):
                sign = -1.0 if metric in {"mean_score_mse", "hamming_loss"} else 1.0
                improvement_samples = sign * (samples[:, metric_index] - reference_samples[:, metric_index])
                low, high = np.quantile(improvement_samples, (0.025, 0.975), method="linear")
                contrast_rows.append(
                    {
                        "protocol": protocol,
                        "model_id": arrays.model_id,
                        "model": arrays.label,
                        "family": arrays.family,
                        "metric": metric,
                        "model_estimate": values[metric_index],
                        "reference_estimate": reference_values[metric_index],
                        "improvement": sign * (values[metric_index] - reference_values[metric_index]),
                        "pointwise_low": float(low),
                        "pointwise_high": float(high),
                        "raw_bootstrap_p": raw_p if metric == "mean_balanced_accuracy" else np.nan,
                        "holm_p": np.nan,
                    }
                )

        adjusted = holm_adjust(np.asarray(raw_p_values, dtype=np.float64))
        for row_index, adjusted_p in zip(raw_p_indices, adjusted, strict=True):
            contrast_rows[row_index]["holm_p"] = float(adjusted_p)

    comparison_df = pd.DataFrame(comparison_rows)
    contrast_df = pd.DataFrame(contrast_rows)
    ba_contrasts = contrast_df[contrast_df["metric"] == "mean_balanced_accuracy"].copy()

    assert comparison_df.shape[0] == 44
    assert ba_contrasts.shape[0] == 42
    assert comparison_df["exact_match"].eq(0.0).all()
    assert not (ba_contrasts["holm_p"] < 0.05).any()
    return comparison_df, ba_contrasts


def load_protocol_arrays(protocol: str) -> list[CombinedArrays]:
    reference = load_reference(protocol)
    arrays = [reference]
    arrays.extend(load_classical(protocol, model_id) for model_id in PLANNED_MODEL_IDS)
    arrays.extend(load_torch(protocol, model_id) for model_id in PRIMARY_TORCH_MODEL_IDS)
    for candidate in arrays[1:]:
        if candidate.test_sample_keys != reference.test_sample_keys:
            raise ValueError(f"{candidate.model_id} uses different ordered test sample keys")
        if not np.array_equal(candidate.targets, reference.targets):
            raise ValueError(f"{candidate.model_id} uses different test targets")
        if not np.array_equal(candidate.subject_ids, reference.subject_ids):
            raise ValueError(f"{candidate.model_id} uses different test subjects")
    return arrays


def load_reference(protocol: str) -> CombinedArrays:
    runs = order_runs(protocol, [load_evaluation_run(path) for path in REFERENCE_RUN_DIRS[protocol]])
    spec = get_model_spec(REFERENCE_MODEL_ID)
    return combine_runs(
        model_id=REFERENCE_MODEL_ID,
        label=spec.label,
        family="Reference",
        task=spec.task,
        topology=spec.topology,
        score_semantics=spec.score_semantics,
        runs=runs,
        score_attribute="probabilities",
        metadata={"parameter_count": np.nan, "training_seconds": np.nan},
        baseline_score_attribute="baseline_probabilities",
    )


def load_classical(protocol: str, model_id: str) -> CombinedArrays:
    runs = []
    for run_dir in sorted((CLASSICAL_ROOT / model_id).iterdir()):
        run = load_model_run(run_dir)
        if run.evaluation["protocol"] == protocol:
            runs.append(run)
    spec = get_model_spec(model_id)
    return combine_runs(
        model_id=model_id,
        label=spec.label,
        family="Classical",
        task=spec.task,
        topology=spec.topology,
        score_semantics=spec.score_semantics,
        runs=order_runs(protocol, runs),
        score_attribute="scores",
        metadata={"parameter_count": np.nan, "training_seconds": np.nan},
    )


def load_torch(protocol: str, model_id: str) -> CombinedArrays:
    runs = []
    for run_dir in sorted((TORCH_ROOT / model_id).iterdir()):
        run = load_torch_run(run_dir)
        if run.evaluation["protocol"] == protocol:
            runs.append(run)
    ordered = order_runs(protocol, runs)
    architecture, method = parse_torch_model_id(model_id)
    return combine_runs(
        model_id=model_id,
        label=torch_label(model_id),
        family="Torch",
        task="classifier",
        topology="multilabel",
        score_semantics="native_probability",
        runs=ordered,
        score_attribute="scores",
        metadata={
            "architecture": architecture,
            "method": method,
            "parameter_count": int(ordered[0].training["parameter_count"]),
            "training_seconds": float(sum(run.training["training_seconds"] for run in ordered)),
        },
    )


def order_runs(protocol: str, runs: list[object]) -> list[object]:
    expected = DIRECTION_ORDER[protocol]
    by_direction = {run.evaluation["direction"]["name"]: run for run in runs}
    if len(by_direction) != len(runs) or set(by_direction) != set(expected):
        raise ValueError(f"{protocol} requires directions {expected}; received {tuple(by_direction)}")
    ordered = [by_direction[direction] for direction in expected]
    if any(run.evaluation["protocol"] != protocol for run in ordered):
        raise ValueError("Run protocol differs from requested protocol")
    return ordered


def combine_runs(
    *,
    model_id: str,
    label: str,
    family: str,
    task: str,
    topology: str,
    score_semantics: str,
    runs: list[object],
    score_attribute: str,
    metadata: dict[str, object],
    baseline_score_attribute: str | None = None,
) -> CombinedArrays:
    keys: list[tuple[int, int, int]] = []
    targets = []
    scores = []
    predictions = []
    subject_ids = []
    baseline_scores: dict[str, list[np.ndarray]] | None = None
    baseline_predictions: dict[str, list[np.ndarray]] | None = None

    for run in runs:
        run_keys = tuple(tuple(key) for key in run.split["test_sample_keys"])
        if set(keys) & set(run_keys):
            raise ValueError(f"{model_id} has overlapping test sample keys across directions")
        keys.extend(run_keys)
        targets.append(run.test_targets)
        scores.append(getattr(run, score_attribute))
        predictions.append(run.predictions)
        subject_ids.append(run.test_subject_ids)
        if baseline_score_attribute is not None:
            run_baseline_scores = getattr(run, baseline_score_attribute)
            run_baseline_predictions = run.baseline_predictions
            if baseline_scores is None:
                baseline_scores = {name: [] for name in run_baseline_scores}
                baseline_predictions = {name: [] for name in run_baseline_predictions}
            if baseline_predictions is None or set(run_baseline_scores) != set(baseline_scores):
                raise ValueError("Reference baseline score names differ across directions")
            if set(run_baseline_predictions) != set(baseline_predictions):
                raise ValueError("Reference baseline prediction names differ across directions")
            for name in baseline_scores:
                baseline_scores[name].append(run_baseline_scores[name])
                baseline_predictions[name].append(run_baseline_predictions[name])

    return CombinedArrays(
        model_id=model_id,
        label=label,
        family=family,
        task=task,
        topology=topology,
        score_semantics=score_semantics,
        test_sample_keys=tuple(keys),
        targets=np.asarray(np.concatenate(targets, axis=0), dtype=np.int8),
        scores=np.asarray(np.concatenate(scores, axis=0), dtype=np.float64),
        predictions=np.asarray(np.concatenate(predictions, axis=0), dtype=np.int8),
        subject_ids=np.asarray(np.concatenate(subject_ids), dtype=np.int64),
        metadata=metadata,
        baseline_scores=None
        if baseline_scores is None
        else {name: np.concatenate(parts, axis=0) for name, parts in baseline_scores.items()},
        baseline_predictions=None
        if baseline_predictions is None
        else {name: np.concatenate(parts, axis=0) for name, parts in baseline_predictions.items()},
    )


def metric_vector(metrics: object) -> np.ndarray:
    return np.asarray(
        (
            metrics.mean_balanced_accuracy,
            metrics.mean_score_mse,
            metrics.mean_sample_iou,
            metrics.micro_iou,
            metrics.hamming_loss,
            metrics.exact_match_accuracy,
        ),
        dtype=np.float64,
    )


def build_bootstrap_rows(targets: np.ndarray, subject_ids: np.ndarray) -> list[np.ndarray]:
    unique_subjects = np.unique(subject_ids)
    subject_rows = {int(subject): np.flatnonzero(subject_ids == subject) for subject in unique_subjects}
    rng = np.random.default_rng(RANDOM_STATE)
    rows: list[np.ndarray] = []
    attempts = 0
    max_attempts = N_RESAMPLES * 100
    while len(rows) < N_RESAMPLES and attempts < max_attempts:
        attempts += 1
        drawn = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
        indices = np.concatenate([subject_rows[int(subject)] for subject in drawn])
        sampled_targets = targets[indices]
        positives = sampled_targets.sum(axis=0, dtype=np.int64)
        if np.any(positives == 0) or np.any(positives == sampled_targets.shape[0]):
            continue
        rows.append(indices.astype(np.int64))
    if len(rows) != N_RESAMPLES:
        raise RuntimeError("Could not draw enough class-complete subject bootstrap samples")
    return rows


def bootstrap_metric_matrix(arrays: CombinedArrays, rows: list[np.ndarray]) -> np.ndarray:
    values = np.empty((len(rows), len(METRIC_COLUMNS)), dtype=np.float64)
    for index, row_indices in enumerate(rows):
        values[index] = metric_vector(
            evaluate_prediction_matrix(
                arrays.targets[row_indices],
                arrays.predictions[row_indices],
                arrays.scores[row_indices],
            )
        )
    return values


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=np.float64)
    running = 0.0
    count = p_values.size
    for rank, original_index in enumerate(order):
        value = (count - rank) * float(p_values[original_index])
        running = max(running, value)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def torch_label(model_id: str) -> str:
    architecture, method = parse_torch_model_id(model_id)
    architecture_label = {
        "eegnet": "EEGNet",
        "deep-convnet": "DeepConvNet",
        "shallow-convnet": "ShallowConvNet",
    }[architecture]
    return f"{architecture_label} / {method.upper()}"


def plot_rankings(comparison_df: pd.DataFrame) -> None:
    for protocol, filename, title in (
        ("cross-subject", "experiment_cross_subject_ranking.pdf", "Cross-subject"),
        ("within-subject", "experiment_within_subject_ranking.pdf", "Bidirectional cross-trial"),
    ):
        protocol_df = comparison_df[comparison_df["protocol"] == protocol].sort_values("balanced_accuracy")
        fig, ax = plt.subplots(figsize=(7.0, 6.6))
        colors = protocol_df["family"].map(FAMILY_COLORS)
        ax.barh(protocol_df["model"], protocol_df["balanced_accuracy"], color=colors, height=0.72)
        ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Средняя per-pixel balanced accuracy")
        ax.set_title(f"Ранжирование моделей: {title}")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(0.47, 0.525)
        add_family_legend(ax)
        fig.tight_layout()
        fig.savefig(THESIS_IMAGES / filename)
        plt.close(fig)


def plot_deltas(ba_contrasts: pd.DataFrame) -> None:
    for protocol, filename, title in (
        ("cross-subject", "experiment_cross_subject_delta_logistic.pdf", "Cross-subject"),
        ("within-subject", "experiment_within_subject_delta_logistic.pdf", "Bidirectional cross-trial"),
    ):
        protocol_df = ba_contrasts[ba_contrasts["protocol"] == protocol].sort_values("improvement")
        fig, ax = plt.subplots(figsize=(7.0, 6.4))
        colors = protocol_df["family"].map(FAMILY_COLORS)
        xerr = np.vstack(
            [
                protocol_df["improvement"] - protocol_df["pointwise_low"],
                protocol_df["pointwise_high"] - protocol_df["improvement"],
            ]
        )
        ax.errorbar(
            protocol_df["improvement"],
            protocol_df["model"],
            xerr=xerr,
            fmt="none",
            ecolor="#777777",
            capsize=2.0,
            linewidth=1.0,
        )
        ax.scatter(protocol_df["improvement"], protocol_df["model"], c=colors, s=22, zorder=3)
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Изменение balanced accuracy относительно Logistic Regression")
        ax.set_title(f"Paired bootstrap-контраст: {title}")
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.055, 0.055)
        add_family_legend(ax, include_reference=False)
        fig.tight_layout()
        fig.savefig(THESIS_IMAGES / filename)
        plt.close(fig)


def plot_reconstructions() -> None:
    targets = np.load(RECONSTRUCTION_RUN / "test_targets.npy").astype(np.int8)
    predictions = np.load(RECONSTRUCTION_RUN / "predictions.npy").astype(np.int8)
    distances = np.abs(targets - predictions).sum(axis=1)
    order = np.argsort(distances)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    row_labels = ("Минимальная ошибка", "Медианная ошибка", "Максимальная ошибка")

    fig, axes = plt.subplots(len(picks), 2, figsize=(4.6, 6.1))
    for row_index, sample_index in enumerate(picks):
        for col_index, matrix in enumerate((targets[sample_index], predictions[sample_index])):
            ax = axes[row_index, col_index]
            ax.imshow(matrix.reshape(6, 6), cmap="Greys", vmin=0, vmax=1)
            ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
            ax.grid(which="minor", color="#BBBBBB", linewidth=0.5)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            if col_index == 0:
                ax.set_ylabel(f"{row_labels[row_index]}\n{int(distances[sample_index])} из 36", fontsize=9)
            ax.set_title("Истинная" if col_index == 0 else "Предсказанная", fontsize=10)
    fig.suptitle("Реальные тестовые реконструкции ShallowConvNet / Morlet", fontsize=11)
    fig.tight_layout()
    fig.savefig(THESIS_IMAGES / "experiment_reconstruction_examples_real.pdf")
    plt.close(fig)


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.03, 0.56, 0.16, 0.22, "EEG-эпоха\n15 с"),
        (0.24, 0.56, 0.17, 0.22, "Представления\nпризнаки / спектр"),
        (0.46, 0.56, 0.18, 0.22, "Модели\nклассика / CNN"),
        (0.70, 0.56, 0.13, 0.22, "36\nлогитов"),
        (0.88, 0.56, 0.09, 0.22, "6 x 6"),
        (0.24, 0.17, 0.17, 0.18, "нормировка\nпо обучению"),
        (0.46, 0.17, 0.18, 0.18, "групповая\nвалидация"),
        (0.70, 0.17, 0.20, 0.18, "кластерный\nbootstrap"),
    ]
    for x, y, width, height, text in boxes:
        draw_box(ax, x, y, width, height, text)
    arrows = [
        ((0.19, 0.67), (0.24, 0.67)),
        ((0.41, 0.67), (0.46, 0.67)),
        ((0.64, 0.67), (0.70, 0.67)),
        ((0.83, 0.67), (0.88, 0.67)),
        ((0.325, 0.56), (0.325, 0.35)),
        ((0.55, 0.56), (0.55, 0.35)),
        ((0.765, 0.56), (0.80, 0.35)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1})
    fig.tight_layout()
    fig.savefig(THESIS_IMAGES / "experiment_pipeline.pdf")
    plt.close(fig)


def plot_architectures() -> None:
    architecture_specs = {
        "eegnet_architecture.pdf": (
            "EEGNet",
            [
                "Вход\nB x P x C x W",
                "Темпоральная\nсвёртка",
                "Depthwise\nсвёртка",
                "ELU + avg pool\n+ dropout",
                "Separable\nсвёртка",
                "Глобальный\npool + 36",
            ],
        ),
        "deepconvnet_architecture.pdf": (
            "DeepConvNet",
            [
                "Вход\nB x P x C x W",
                "Темпоральная\nсвёртка 25",
                "Пространственная\nсвёртка 25",
                "Свёрточный\nблок 50",
                "Свёрточный\nблок 100",
                "Свёрточный\nблок 200",
                "Глобальный\npool + 36",
            ],
        ),
        "shallowconvnet_architecture.pdf": (
            "ShallowConvNet",
            [
                "Вход\nB x P x C x W",
                "Темпоральная\nсвёртка 40",
                "Пространственная\nсвёртка 40",
                "Квадрат",
                "Avg pool",
                "Логарифм\n+ dropout",
                "Глобальный\npool + 36",
            ],
        ),
    }
    for filename, (title, labels) in architecture_specs.items():
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        width = 0.12 if len(labels) > 6 else 0.14
        gap = (0.94 - len(labels) * width) / (len(labels) - 1)
        y = 0.44
        x = 0.03
        previous_right = None
        for index, label in enumerate(labels):
            facecolor = "#EEF3F7" if index == 0 else "#F6F2EA"
            if index == len(labels) - 1:
                facecolor = "#E9F1E7"
            draw_box(ax, x, y, width, 0.24, label, facecolor=facecolor)
            if previous_right is not None:
                ax.annotate(
                    "",
                    xy=(x, y + 0.12),
                    xytext=(previous_right, y + 0.12),
                    arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
                )
            previous_right = x + width
            x += width + gap
        ax.text(0.5, 0.83, title, ha="center", va="center", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(THESIS_IMAGES / filename)
        plt.close(fig)


def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#F3F5F7",
) -> None:
    rect = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor="#4D5964",
        facecolor=facecolor,
    )
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8.5)


def add_family_legend(ax: plt.Axes, *, include_reference: bool = True) -> None:
    families = ["Classical", "Torch"]
    if include_reference:
        families = ["Reference", *families]
    labels = {
        "Reference": "Reference",
        "Classical": "Classical",
        "Torch": "Torch",
    }
    handles = [
        patches.Patch(color=FAMILY_COLORS[family], label=labels[family])
        for family in families
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95)


def verify_key_numbers(comparison_df: pd.DataFrame, ba_contrasts: pd.DataFrame) -> None:
    cross_leader = comparison_df[comparison_df["protocol"] == "cross-subject"].sort_values(
        "balanced_accuracy",
        ascending=False,
    ).iloc[0]
    within_leader = comparison_df[comparison_df["protocol"] == "within-subject"].sort_values(
        "balanced_accuracy",
        ascending=False,
    ).iloc[0]
    min_holm = ba_contrasts["holm_p"].min()
    assert cross_leader["model_id"] == "ridge-regression-independent"
    assert np.isclose(cross_leader["balanced_accuracy"], 0.518381514752213)
    assert within_leader["model_id"] == "deep-convnet-stft-multilabel"
    assert np.isclose(within_leader["balanced_accuracy"], 0.512011, atol=5e-7)
    assert np.isclose(min_holm, 0.273000)
    print(
        "Generated thesis figures: "
        f"cross_subject_leader={cross_leader['model_id']}:{cross_leader['balanced_accuracy']:.6f} "
        f"within_subject_leader={within_leader['model_id']}:{within_leader['balanced_accuracy']:.6f} "
        f"min_holm_p={min_holm:.6f}"
    )


if __name__ == "__main__":
    main()
