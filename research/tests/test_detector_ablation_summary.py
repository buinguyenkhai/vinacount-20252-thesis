import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.detector_ablation_summary import run_detector_ablation_summary


class Wave4DetectorAblationSummaryTest(unittest.TestCase):
    def test_summary_combines_rule_and_existing_model_metrics_into_markdown_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rule_root = _write_variant(
                root / "rule",
                manifest={"variant": "rule_only_detector_baseline", "split": "validation"},
                metrics={
                    "num_examples": 10,
                    "num_valid_predictions": 10,
                    "num_invalid_responses": 0,
                    "support_level_exact_match_rate": 0.3,
                    "support_level_macro_f1": 0.1,
                    "weakly_supported_detection": {
                        "precision": None,
                        "recall": 0.0,
                        "f1": 0.0,
                        "true_positive": 0,
                        "predicted_total": 0,
                        "gold_total": 2,
                    },
                    "false_positive_rejection": {"rate": 1.0},
                    "invalid_reason_counts": {},
                    "severity_exact_match_rate": 0.4,
                    "severity_macro_f1": 0.2,
                },
            )
            model_root = _write_variant(
                root / "model",
                manifest={"adapter_dir": "artifacts/model/adapter", "base_model": "unsloth/Qwen3.5-4B", "split": "validation"},
                metrics={
                    "num_examples": 4,
                    "num_valid_predictions": 3,
                    "num_invalid_responses": 1,
                    "support_level_exact_match_rate": 0.5,
                    "support_level_confusion_matrix_counts": {
                        "supported": {"supported": 1},
                        "weakly_supported": {"supported": 1},
                        "not_supported": {"not_supported": 1},
                    },
                    "by_gold_support_level": {
                        "supported": {"total": 1},
                        "weakly_supported": {"total": 1},
                        "not_supported": {"total": 2, "invalid": 1},
                    },
                    "false_positive_rejection": {"rate": 0.75},
                    "invalid_reason_counts": {"invalid_evidence_ref_type": 1},
                    "severity_exact_match_rate": 0.5,
                },
                predictions=[
                    _accepted_prediction("supported", "high", "high"),
                    _accepted_prediction("weakly_supported", "medium", "high"),
                    _accepted_prediction("not_supported", "unknown", "unknown"),
                    _invalid_prediction("not_supported", "low"),
                ],
            )

            result = run_detector_ablation_summary(
                output_root=root / "summary",
                variants=[
                    ("Rule-only", rule_root),
                    ("Detector-LoRA", model_root),
                ],
            )

            self.assertEqual(result.status, "passed", result.errors)
            summary = _read_json(result.summary_json_path)
            markdown = result.summary_markdown_path.read_text(encoding="utf-8")
            self.assertEqual([row["variant"] for row in summary["variants"]], ["Rule-only", "Detector-LoRA"])
            self.assertEqual(summary["variants"][0]["valid_output_rate"], 1.0)
            self.assertAlmostEqual(summary["variants"][1]["support_level_macro_f1"], 4 / 9)
            self.assertEqual(summary["variants"][1]["weakly_supported_recall"], 0.0)
            self.assertEqual(summary["variants"][1]["invalid_evidence_ref_rate"], 0.25)
            self.assertAlmostEqual(summary["variants"][1]["severity_macro_f1"], 5 / 12)
            self.assertIn("| Rule-only |", markdown)
            self.assertIn("| Detector-LoRA |", markdown)
            self.assertIn("Synthetic detector validation split", markdown)
            self.assertIn("Valid output rate", markdown)

    def test_public_command_writes_summary_from_variant_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            variant_root = _write_variant(
                root / "rule",
                manifest={"variant": "rule_only_detector_baseline", "split": "validation"},
                metrics={
                    "num_examples": 1,
                    "num_valid_predictions": 1,
                    "num_invalid_responses": 0,
                    "support_level_exact_match_rate": 1.0,
                    "support_level_macro_f1": 1.0,
                    "weakly_supported_detection": {
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "true_positive": 0,
                        "predicted_total": 0,
                        "gold_total": 0,
                    },
                    "false_positive_rejection": {"rate": None},
                    "invalid_reason_counts": {},
                    "severity_exact_match_rate": 1.0,
                    "severity_macro_f1": 1.0,
                },
            )
            output_root = root / "summary"

            completed = _run_summary_command(
                "--output-root",
                str(output_root),
                "--variant",
                f"Rule-only={variant_root}",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "passed")
            self.assertTrue((output_root / "summary.json").exists())
            self.assertTrue((output_root / "summary.md").exists())


def _write_variant(path: Path, *, manifest: dict, metrics: dict, predictions: list[dict] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if predictions is not None:
        (path / "predictions.jsonl").write_text(
            "".join(json.dumps(prediction, sort_keys=True) + "\n" for prediction in predictions),
            encoding="utf-8",
        )
    return path


def _accepted_prediction(support_level: str, gold_severity: str, predicted_severity: str) -> dict:
    return {
        "prediction_status": "accepted",
        "gold_support_level": support_level,
        "gold_severity": gold_severity,
        "predicted_support_level": support_level,
        "predicted_severity": predicted_severity,
    }


def _invalid_prediction(support_level: str, gold_severity: str) -> dict:
    return {
        "prediction_status": "invalid",
        "gold_support_level": support_level,
        "gold_severity": gold_severity,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_summary_command(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.detector_ablation_summary", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
