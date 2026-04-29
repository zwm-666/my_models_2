"""Plot figures from official CAPT-UniShape train/evaluate outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


CLASS_NAMES = ["正常", "过湿", "过干"]


def _read_summary_csv(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _safe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def _load_metrics(results_dir: Path) -> dict[str, Any]:
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _confusion_from_metrics(metrics: dict[str, Any]) -> np.ndarray[Any, Any]:
    if "test" in metrics and isinstance(metrics["test"], dict):
        return np.asarray(metrics["test"]["confusion_matrix"], dtype=np.int64)
    return np.asarray(metrics["confusion_matrix"], dtype=np.int64)


def _score_from_metrics(metrics: dict[str, Any], key: str) -> float:
    if "test" in metrics and isinstance(metrics["test"], dict) and key in metrics["test"]:
        return float(metrics["test"][key])
    return float(metrics[key])


def plot_confusion_matrix(results_dir: Path, output_dir: Path, title: str | None = None) -> Path:
    metrics = _load_metrics(results_dir)
    cm = _confusion_from_metrics(metrics)
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sum
    labels = CLASS_NAMES[: cm.shape[0]]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    image = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, f"{cm[row, col]}\n{cm_norm[row, col] * 100:.1f}%", ha="center", va="center")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title or f"混淆矩阵：{results_dir.name}")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{results_dir.name}_confusion_matrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(results_dirs: list[Path], output_dir: Path) -> Path:
    names: list[str] = []
    acc: list[float] = []
    macro_f1: list[float] = []
    params: list[float] = []
    for results_dir in results_dirs:
        metrics = _load_metrics(results_dir)
        names.append(results_dir.name.replace("official_unishape_", ""))
        acc.append(_score_from_metrics(metrics, "accuracy"))
        macro_f1.append(_score_from_metrics(metrics, "macro_f1"))
        params.append(float(metrics.get("parameter_count", 0)) / 1_000_000.0)

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7.0, len(names) * 2.4), 5.0))
    bars_acc = ax.bar(x - width / 2, acc, width, label="准确率", color="#4e79a7")
    bars_f1 = ax.bar(x + width / 2, macro_f1, width, label="宏平均F1", color="#e15759")
    for bars in (bars_acc, bars_f1):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("得分")
    ax.set_title("官方 UniShape 版 CAPT-UniShape 模型对比")
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(x, params, marker="o", color="#59a14f", label="参数量（百万）")
    ax2.set_ylabel("参数量（百万）")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "official_model_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_split_summary(
    summary_path: Path,
    output_dir: Path,
    title: str = "不同训练/测试比例下的模型对比",
    output_name: str = "official_split_ratio_summary.png",
) -> Path:
    rows = _read_summary_csv(summary_path)
    ratio_order = {"8_2": 0, "7_3": 1, "6_4": 2, "5_5": 3}
    model_order = {
        "proposed": 0,
        "logreg": 1,
        "svm": 2,
        "random_forest": 3,
        "mlp": 4,
        "cnn1d": 5,
        "lstm": 6,
        "transformer": 7,
        "itransformer": 8,
        "rbf": 9,
        "no_rbf": 10,
    }
    ratios = sorted({row["ratio"] for row in rows}, key=lambda item: ratio_order.get(item.replace(":", "_"), 99))
    models = sorted({row["model"] for row in rows}, key=lambda item: model_order.get(item, 99))
    x = np.arange(len(ratios))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(9.0, len(ratios) * max(2.2, len(models) * 0.55)), 5.6))
    colors = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab", "#76b7b2", "#edc948"]
    for index, model in enumerate(models):
        values: list[float] = []
        for ratio in ratios:
            matched = [row for row in rows if row["ratio"] == ratio and row["model"] == model]
            values.append(_safe_float(matched[-1], "test_macro_f1") if matched else 0.0)
        offset = (index - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[index % len(colors)])
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([ratio.replace("_", ":") for ratio in ratios])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("训练集:测试集比例")
    ax.set_ylabel("测试集宏平均F1")
    ax.set_title(title)
    ax.legend(ncol=min(4, max(len(models), 1)), fontsize=9)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_baseline_summary(summary_path: Path, output_dir: Path) -> Path:
    return plot_split_summary(
        summary_path,
        output_dir,
        title="提出模型与传统机器学习/深度学习/Transformer 基准对比",
        output_name="official_baseline_comparison_summary.png",
    )


def plot_noise_summary(summary_path: Path, output_dir: Path) -> Path:
    rows = _read_summary_csv(summary_path)
    model_order = {"rbf": 0, "no_rbf": 1}
    models = sorted({row["model"] for row in rows}, key=lambda item: model_order.get(item, 99))
    noise_stds = sorted({_safe_float(row, "noise_std") for row in rows})
    colors = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for index, model in enumerate(models):
        values: list[float] = []
        for noise_std in noise_stds:
            matched = [row for row in rows if row["model"] == model and abs(_safe_float(row, "noise_std") - noise_std) < 1e-12]
            values.append(_safe_float(matched[-1], "test_macro_f1") if matched else 0.0)
        ax.plot(noise_stds, values, marker="o", linewidth=2.0, label=model, color=colors[index % len(colors)])
        for x_value, y_value in zip(noise_stds, values):
            ax.text(x_value, y_value + 0.015, f"{y_value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("测试集相对高斯噪声标准差")
    ax.set_ylabel("测试集宏平均F1")
    ax.set_title("单一训练/测试比例下的噪声鲁棒性")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "official_noise_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation_summary(summary_path: Path, output_dir: Path) -> Path:
    rows = _read_summary_csv(summary_path)
    rows = sorted(rows, key=lambda row: _safe_float(row, "test_macro_f1"), reverse=True)
    names = [row["variant"] for row in rows]
    macro_f1 = [_safe_float(row, "test_macro_f1") for row in rows]
    accuracy = [_safe_float(row, "test_accuracy") for row in rows]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8.8, max(5.2, len(names) * 0.48)))
    ax.barh(y - 0.18, macro_f1, 0.36, label="宏平均F1", color="#e15759")
    ax.barh(y + 0.18, accuracy, 0.36, label="准确率", color="#4e79a7")
    for index, value in enumerate(macro_f1):
        ax.text(value + 0.012, index - 0.18, f"{value:.3f}", va="center", fontsize=9)
    for index, value in enumerate(accuracy):
        ax.text(value + 0.012, index + 0.18, f"{value:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("测试集得分")
    ax.set_title("官方 CAPT-UniShape 消融实验汇总")
    ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "official_ablation_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制官方 UniShape 版 CAPT-UniShape 结果图")
    parser.add_argument("--results", nargs="+", default=[], help="一个或多个包含 metrics.json 的结果目录")
    parser.add_argument("--baseline-summary", help="基准模型对比实验输出的 summary.csv")
    parser.add_argument("--ablation-summary", help="消融实验输出的 summary.csv")
    parser.add_argument("--noise-summary", help="噪声鲁棒性实验输出的 summary.csv")
    parser.add_argument("--output-dir", default="figures/official_unishape")
    args = parser.parse_args()

    result_dirs = [Path(item) for item in args.results]
    output_dir = Path(args.output_dir)
    written: list[Path] = [plot_confusion_matrix(path, output_dir) for path in result_dirs]
    if len(result_dirs) >= 2:
        written.append(plot_model_comparison(result_dirs, output_dir))
    if args.baseline_summary:
        written.append(plot_baseline_summary(Path(args.baseline_summary), output_dir))
    if args.ablation_summary:
        written.append(plot_ablation_summary(Path(args.ablation_summary), output_dir))
    if args.noise_summary:
        written.append(plot_noise_summary(Path(args.noise_summary), output_dir))
    if not written:
        parser.error("请至少提供 --results、--baseline-summary、--ablation-summary 或 --noise-summary 其中之一")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
