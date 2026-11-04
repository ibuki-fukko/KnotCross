"""Figures from saved measurements; never used for model selection."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from code.geometry.vectorized_basis import basis_matrix
from code.evaluation.set_metrics import match_metrics

STUDY = ROOT/"results/learned_study"
OUTPUT = STUDY/"figures"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 11.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    chosen = Path(read(STUDY/"selection.json")["selected"])
    trials = [("Baseline", STUDY/"depth_0p0"), ("Deep supervision", STUDY/"depth_0p5"),
              ("Positive weight 5", STUDY/"positive_5p0"), ("Refresh 20 epochs", STUDY/"dynamic_True"),
              ("Refresh 80 epochs", STUDY/"refresh_duration_80"), ("Selected", chosen)]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 4))
    axes[0].barh([name for name, path in trials], [read(path/"metrics.json")["validation_metrics"]["f1"] for name, path in trials], color="#34668f")
    axes[0].set(xlim=(0.80, 0.95), xlabel="Validation F1, parameter tolerance 0.035")
    axes[0].set_xticks([0.80, 0.85, 0.90, 0.95])
    for name, path in [trials[0], trials[1], trials[-1]]:
        history = read(path/"history.json")
        axes[1].plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label=name)
    axes[1].set(xlabel="Epoch", ylabel="Validation objective within each run", yscale="log")
    axes[1].legend(fontsize=10)
    axes[1].text(0.03, 0.03, "Different loss weights:\ncompare trends, not levels", transform=axes[1].transAxes, fontsize=9)
    figure.tight_layout()
    figure.savefig(OUTPUT/"validation_experiments.png", dpi=180)
    plt.close(figure)

    report = read(STUDY/"final_evaluation.json")["fresh_holdout"]["families"]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for family, color in [("baseline", "#8c8c8c"), ("selected", "#34668f")]:
        runs = report[family]["runs"]
        values = np.array([[run["f1_by_parameter_tolerance"][str(tolerance)] for tolerance in (0.005, 0.01, 0.02, 0.035)] for run in runs])
        axes[0].errorbar([0.005, 0.01, 0.02, 0.035], values.mean(axis=0), yerr=values.std(axis=0), marker="o", label=family, color=color)
        residuals = [run["all_predictions_mean_geometry_residual"] for run in runs]
        axes[1].bar(family, np.mean(residuals), yerr=np.std(residuals), color=color, capsize=4)
    axes[0].set(xlabel="Parameter matching tolerance", ylabel="Fresh holdout F1", ylim=(0, 1))
    axes[0].legend()
    axes[1].set(ylabel="Mean planar residual, all returned predictions")
    figure.tight_layout()
    figure.savefig(OUTPUT/"fresh_holdout_summary.png", dpi=180)
    plt.close(figure)

    archive = np.load(ROOT/"data/learned_study_locked_test.npz")
    predictions = read(chosen/"fresh_holdout_predictions.json")
    records = []
    for index, (prediction, label, count) in enumerate(zip(predictions, archive["test_labels"], archive["test_counts"])):
        metrics = match_metrics([prediction], label[None], np.array([count]))
        records.append({"index": index, "count": int(count), "metrics": metrics})
    picks = []
    for count in (0, 1, 2, 3):
        candidates = [row for row in records if row["count"] == count]
        picks.append(max(candidates, key=lambda row: row["metrics"]["f1"])["index"])
    remaining = [row for row in records if row["index"] not in picks]
    picks.extend(row["index"] for row in sorted(remaining, key=lambda row: row["metrics"]["false_positive"]+row["metrics"]["false_negative"], reverse=True)[:2])
    figure, axes = plt.subplots(3, 2, figsize=(8.4, 8.1))
    parameter = np.linspace(0, 1, 250)
    for axis, index in zip(axes.ravel(), picks):
        curve_a, curve_b = archive["test_curve_a"][index], archive["test_curve_b"][index]
        points_a, points_b = basis_matrix(parameter)@curve_a, basis_matrix(parameter)@curve_b
        axis.plot(*points_a.T, color="#366c96", label="Curve A", linewidth=1.3)
        axis.plot(*points_b.T, color="#bd7a33", label="Curve B", linewidth=1.3)
        count = archive["test_counts"][index]
        if count:
            axis.scatter(*(basis_matrix(archive["test_labels"][index, :count, 0])@curve_a).T, facecolors="none", edgecolors="#25914b", s=50, label="Reference")
        if predictions[index]:
            parameters = np.array(predictions[index])
            axis.scatter(*(basis_matrix(parameters[:, 0])@curve_a).T, marker="x", color="#c44242", s=30, label="Predicted A")
            axis.scatter(*(basis_matrix(parameters[:, 1])@curve_b).T, marker="+", color="#854299", s=30, label="Predicted B")
        metrics = records[index]["metrics"]
        axis.set_title(f'Pair {index}: TP {metrics["true_positive"]}, FP {metrics["false_positive"]}, FN {metrics["false_negative"]}', fontsize=10.5)
        axis.set_aspect("equal", adjustable="datalim")
        axis.tick_params(labelsize=9.5)
    axes[0, 1].legend(fontsize=10, loc="best")
    figure.tight_layout()
    figure.savefig(OUTPUT/"fresh_holdout_examples.png", dpi=180)
    plt.close(figure)
    (OUTPUT/"example_selection.json").write_text(json.dumps({"indices": picks,
        "selection": "one high-F1 example per count then two largest error counts; final-only illustration"}, indent=2), encoding="utf-8")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
