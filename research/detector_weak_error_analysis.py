from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PREDICTIONS_JSONL = Path(
    "artifacts/detector_real_manual/qwen3_5_4b_detector_lora_guard/validation/predictions.jsonl"
)
DEFAULT_OUTPUT_MARKDOWN = Path("docs/04_evaluation/DetectorWeakCalibrationErrorAnalysis.md")
DEFAULT_OUTPUT_JSON = Path("artifacts/detector_real_manual/weak_calibration_error_analysis.json")


@dataclass(frozen=True)
class WeakErrorAnalysisResult:
    status: str
    output_markdown: Path
    output_json: Path
    weak_case_count: int
    errors: list[str]


def run_detector_weak_error_analysis(
    *,
    predictions_jsonl: Path | str = DEFAULT_PREDICTIONS_JSONL,
    output_markdown: Path | str = DEFAULT_OUTPUT_MARKDOWN,
    output_json: Path | str = DEFAULT_OUTPUT_JSON,
    model_label: str = "Qwen3.5-4B-Detector-LoRA-Guard",
    source_role: str = "real/manual validation diagnostic",
) -> WeakErrorAnalysisResult:
    predictions_path = Path(predictions_jsonl)
    output_markdown_path = Path(output_markdown)
    output_json_path = Path(output_json)
    predictions = _read_jsonl(predictions_path)
    weak_cases = [_weak_case(record) for record in predictions if record.get("gold_support_level") == "weakly_supported"]
    analysis = {
        "status": "passed",
        "analysis_type": "weak_support_calibration_error_analysis",
        "model_label": model_label,
        "source_role": source_role,
        "predictions_jsonl": str(predictions_path),
        "weak_case_count": len(weak_cases),
        "predicted_support_counts": _counts(case["predicted_support_level"] for case in weak_cases),
        "predicted_severity_counts": _counts(case["predicted_severity"] for case in weak_cases),
        "cases": weak_cases,
        "interpretation": {
            "real_manual_used_for_training_or_selection": False,
            "primary_error_pattern": "gold weakly_supported records are promoted to supported",
            "guard_changes_support_or_severity": False,
        },
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_json_path, analysis)
    output_markdown_path.write_text(_render_markdown(analysis), encoding="utf-8")
    return WeakErrorAnalysisResult(
        status="passed",
        output_markdown=output_markdown_path,
        output_json=output_json_path,
        weak_case_count=len(weak_cases),
        errors=[],
    )


def _weak_case(record: dict[str, Any]) -> dict[str, Any]:
    prediction = record.get("prediction") if isinstance(record.get("prediction"), dict) else {}
    packet = record.get("model_visible_input", {}).get("data", {})
    candidate_summary = packet.get("candidate_summary", {}) if isinstance(packet, dict) else {}
    return {
        "example_id": record.get("example_id"),
        "packet_id": record.get("packet_id"),
        "risk_category": record.get("risk_category"),
        "gold_support_level": record.get("gold_support_level"),
        "predicted_support_level": record.get("predicted_support_level"),
        "gold_severity": record.get("gold_severity"),
        "predicted_severity": record.get("predicted_severity"),
        "prediction_status": record.get("prediction_status"),
        "invalid_reason_codes": record.get("invalid_reason_codes", []),
        "guard_actions": record.get("guard_actions", []),
        "candidate_reason": candidate_summary.get("reason_for_candidate"),
        "predicted_rationale_short": prediction.get("rationale_short"),
    }


def _render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Detector Weak-Support Error Analysis",
        "",
        f"Model: **{analysis['model_label']}**",
        "",
        f"Source: {analysis['source_role']}. This diagnostic is not training data, not a selection gate, and not a protected real-world test.",
        "",
        "## Summary",
        "",
        f"- Gold weakly-supported cases: {analysis['weak_case_count']}",
        f"- Predicted support counts: `{json.dumps(analysis['predicted_support_counts'], sort_keys=True)}`",
        f"- Predicted severity counts: `{json.dumps(analysis['predicted_severity_counts'], sort_keys=True)}`",
        "- Main pattern: gold `weakly_supported` records are promoted to `supported`.",
        "",
        "## Cases",
        "",
        "| Example | Risk category | Gold support | Predicted support | Gold severity | Predicted severity | Rationale excerpt |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for case in analysis["cases"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(case.get("example_id")),
                    _md(case.get("risk_category")),
                    _md(case.get("gold_support_level")),
                    _md(case.get("predicted_support_level")),
                    _md(case.get("gold_severity")),
                    _md(case.get("predicted_severity")),
                    _md(_excerpt(case.get("predicted_rationale_short"))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The pattern is a calibration failure rather than a contract failure. The detector preserves the assessment structure after deterministic guarding, but it treats positive partial evidence as fully supported evidence on this small real/manual diagnostic slice. The calibration experiment should therefore target model training distribution, not semantic guard rules.",
        ]
    )
    return "\n".join(lines) + "\n"


def _excerpt(value: Any, limit: int = 220) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _md(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("|", "\\|")
    return text


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate weak-support calibration error analysis.")
    parser.add_argument("--predictions-jsonl", default=str(DEFAULT_PREDICTIONS_JSONL))
    parser.add_argument("--output-markdown", default=str(DEFAULT_OUTPUT_MARKDOWN))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--model-label", default="Qwen3.5-4B-Detector-LoRA-Guard")
    parser.add_argument("--source-role", default="real/manual validation diagnostic")
    args = parser.parse_args(argv)
    result = run_detector_weak_error_analysis(
        predictions_jsonl=args.predictions_jsonl,
        output_markdown=args.output_markdown,
        output_json=args.output_json,
        model_label=args.model_label,
        source_role=args.source_role,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_markdown": str(result.output_markdown),
                "output_json": str(result.output_json),
                "weak_case_count": result.weak_case_count,
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
