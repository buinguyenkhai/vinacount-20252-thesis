from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("artifacts/detector_ablation/summary_validation_draft")


@dataclass(frozen=True)
class Wave4DetectorAblationSummaryResult:
    status: str
    output_root: Path
    summary_json_path: Path
    summary_markdown_path: Path
    errors: list[str]


def run_detector_ablation_summary(
    *,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    variants: list[tuple[str, Path | str]],
) -> Wave4DetectorAblationSummaryResult:
    if not variants:
        raise ValueError("at least one --variant label=artifact_root is required")
    output_path = Path(output_root)
    summary_json_path = output_path / "summary.json"
    summary_markdown_path = output_path / "summary.md"
    rows = [_variant_summary(label, Path(artifact_root)) for label, artifact_root in variants]
    split_roles = sorted({row["split"] for row in rows if row.get("split")})
    summary = {
        "status": "passed",
        "summary_type": "detector_ablation",
        "metric_version": "detector_ablation_summary_v1",
        "split_roles": split_roles,
        "variants": rows,
        "notes": [
            "Rows are included only for variants explicitly passed through --variant.",
            "Rows use the synthetic detector validation split and support method selection, not protected final performance claims.",
            "False-positive rejection is measured only on gold not-supported and insufficient-evidence records.",
            "RefNorm rows repair only uniquely recoverable local evidence IDs and do not change support or severity labels.",
        ],
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _write_json(summary_json_path, summary)
    summary_markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    return Wave4DetectorAblationSummaryResult(
        status="passed",
        output_root=output_path,
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        errors=[],
    )


def _variant_summary(label: str, artifact_root: Path) -> dict[str, Any]:
    manifest = _read_json(artifact_root / "manifest.json")
    metrics = _read_json(artifact_root / "metrics.json")
    predictions = _read_predictions(artifact_root / "predictions.jsonl")
    num_examples = metrics["num_examples"]
    weak_metrics = _support_class_metric(metrics, "weakly_supported")
    invalid_evidence_ref_count = sum(
        count
        for reason, count in metrics.get("invalid_reason_counts", {}).items()
        if reason.startswith("invalid_evidence")
    )
    return {
        "variant": label,
        "artifact_root": str(artifact_root),
        "source_variant": manifest.get("variant"),
        "base_model": manifest.get("base_model"),
        "adapter_dir": manifest.get("adapter_dir"),
        "split": metrics.get("split") or manifest.get("split"),
        "num_examples": num_examples,
        "valid_output_rate": _rate(metrics.get("num_valid_predictions", 0), num_examples),
        "num_invalid_responses": metrics.get("num_invalid_responses", 0),
        "support_level_accuracy": metrics.get("support_level_exact_match_rate"),
        "support_level_macro_f1": _support_macro_f1(metrics),
        "weakly_supported_precision": weak_metrics.get("precision"),
        "weakly_supported_recall": weak_metrics.get("recall"),
        "weakly_supported_f1": weak_metrics.get("f1"),
        "false_positive_rejection_rate": metrics.get("false_positive_rejection", {}).get("rate"),
        "invalid_evidence_ref_rate": _rate(invalid_evidence_ref_count, num_examples),
        "severity_accuracy": metrics.get("severity_exact_match_rate"),
        "severity_macro_f1": _severity_macro_f1(metrics, predictions),
    }


def _support_class_metric(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    if "support_level_class_metrics" in metrics:
        return metrics["support_level_class_metrics"].get(label, _empty_class_metric())
    return _computed_support_class_metrics(metrics).get(label, _empty_class_metric())


def _support_macro_f1(metrics: dict[str, Any]) -> float | None:
    if metrics.get("support_level_macro_f1") is not None:
        return metrics["support_level_macro_f1"]
    class_metrics = _computed_support_class_metrics(metrics)
    f1_values = [metric["f1"] for metric in class_metrics.values() if metric["f1"] is not None]
    if not f1_values:
        return None
    return sum(f1_values) / len(f1_values)


def _severity_macro_f1(metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> float | None:
    if metrics.get("severity_macro_f1") is not None:
        return metrics["severity_macro_f1"]
    if not predictions:
        return None
    class_metrics = _classification_metrics_from_predictions(
        predictions,
        gold_key="gold_severity",
        predicted_key="predicted_severity",
    )
    f1_values = [metric["f1"] for metric in class_metrics.values() if metric["f1"] is not None]
    if not f1_values:
        return None
    return sum(f1_values) / len(f1_values)


def _classification_metrics_from_predictions(
    predictions: list[dict[str, Any]],
    *,
    gold_key: str,
    predicted_key: str,
) -> dict[str, dict[str, Any]]:
    labels = sorted(
        {
            prediction[gold_key]
            for prediction in predictions
            if prediction.get(gold_key) is not None
        }
        | {
            prediction[predicted_key]
            for prediction in predictions
            if prediction.get("prediction_status") == "accepted" and prediction.get(predicted_key) is not None
        }
    )
    class_metrics: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = 0
        predicted_total = 0
        gold_total = 0
        for prediction in predictions:
            gold = prediction.get(gold_key)
            predicted = prediction.get(predicted_key) if prediction.get("prediction_status") == "accepted" else None
            if gold == label:
                gold_total += 1
            if predicted == label:
                predicted_total += 1
            if gold == label and predicted == label:
                true_positive += 1
        precision = _rate(true_positive, predicted_total)
        recall = _rate(true_positive, gold_total)
        class_metrics[label] = {
            "true_positive": true_positive,
            "predicted_total": predicted_total,
            "gold_total": gold_total,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall, predicted_total=predicted_total, gold_total=gold_total),
        }
    return class_metrics


def _computed_support_class_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    confusion = metrics.get("support_level_confusion_matrix_counts", {})
    by_gold = metrics.get("by_gold_support_level", {})
    labels = sorted(set(by_gold) | set(confusion) | {pred for counts in confusion.values() for pred in counts})
    class_metrics: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = confusion.get(label, {}).get(label, 0)
        predicted_total = sum(counts.get(label, 0) for counts in confusion.values())
        gold_total = by_gold.get(label, {}).get("total", sum(confusion.get(label, {}).values()))
        precision = _rate(true_positive, predicted_total)
        recall = _rate(true_positive, gold_total)
        class_metrics[label] = {
            "true_positive": true_positive,
            "predicted_total": predicted_total,
            "gold_total": gold_total,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall, predicted_total=predicted_total, gold_total=gold_total),
        }
    return class_metrics


def _empty_class_metric() -> dict[str, Any]:
    return {
        "true_positive": 0,
        "predicted_total": 0,
        "gold_total": 0,
        "precision": None,
        "recall": None,
        "f1": None,
    }


def _f1(
    precision: float | None,
    recall: float | None,
    *,
    predicted_total: int,
    gold_total: int,
) -> float | None:
    if predicted_total == 0 and gold_total == 0:
        return None
    if precision is None or recall is None:
        return 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Detector Validation Ablation",
        "",
        "Synthetic detector validation split. Higher is better except for invalid evidence-reference rate.",
        "",
        "| Model | Valid output rate | Support acc. | Support macro F1 | Weak F1 | FP rejection | Invalid evidence ref rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    _fmt(row["valid_output_rate"]),
                    _fmt(row["support_level_accuracy"]),
                    _fmt(row["support_level_macro_f1"]),
                    _fmt(row["weakly_supported_f1"]),
                    _fmt(row["false_positive_rejection_rate"]),
                    _fmt(row["invalid_evidence_ref_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Validation rows are development evidence for method selection, not protected final real-world performance claims.",
            "- False-positive rejection is measured only on gold not-supported and insufficient-evidence records.",
            "- RefNorm rows repair only uniquely recoverable local evidence IDs and do not change support or severity labels.",
            "- Severity metrics and per-class details remain in summary.json and the source metrics artifacts.",
            "- Empty cells mean the source artifact did not record that metric and it could not be reconstructed safely.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_variant_argument(argument: str) -> tuple[str, Path]:
    if "=" not in argument:
        raise ValueError("--variant must use label=artifact_root")
    label, artifact_root = argument.split("=", 1)
    if not label.strip() or not artifact_root.strip():
        raise ValueError("--variant must use non-empty label=artifact_root")
    return label.strip(), Path(artifact_root.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a detector ablation summary table from metrics artifacts.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant in the form label=artifact_root, where artifact_root contains metrics.json and manifest.json.",
    )
    args = parser.parse_args(argv)
    result = run_detector_ablation_summary(
        output_root=args.output_root,
        variants=[_parse_variant_argument(argument) for argument in args.variant],
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_root": str(result.output_root),
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
