import json
import tempfile
import unittest
from pathlib import Path

from research.detector_weak_error_analysis import (
    run_detector_weak_error_analysis,
)


class Wave4DetectorWeakErrorAnalysisTest(unittest.TestCase):
    def test_generates_markdown_and_json_for_gold_weak_cases_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions = root / "predictions.jsonl"
            _write_jsonl(
                predictions,
                [
                    _prediction("weak-1", gold_support="weakly_supported", predicted_support="supported"),
                    _prediction("supported-1", gold_support="supported", predicted_support="supported"),
                ],
            )
            output_md = root / "analysis.md"
            output_json = root / "analysis.json"

            result = run_detector_weak_error_analysis(
                predictions_jsonl=predictions,
                output_markdown=output_md,
                output_json=output_json,
            )

            self.assertEqual(result.weak_case_count, 1)
            analysis = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(analysis["weak_case_count"], 1)
            self.assertEqual(analysis["predicted_support_counts"], {"supported": 1})
            self.assertEqual(analysis["cases"][0]["example_id"], "weak-1")
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("weak-1", markdown)
            self.assertNotIn("supported-1", markdown)


def _prediction(example_id: str, *, gold_support: str, predicted_support: str) -> dict:
    return {
        "example_id": example_id,
        "packet_id": f"PACKET_{example_id}",
        "risk_category": "earnings_cashflow_mismatch",
        "gold_support_level": gold_support,
        "predicted_support_level": predicted_support,
        "gold_severity": "medium",
        "predicted_severity": "high",
        "prediction_status": "accepted",
        "guard_actions": ["stripped_evidence_ref_aux_fields"],
        "model_visible_input": {
            "data": {
                "candidate_summary": {
                    "reason_for_candidate": "Fixture candidate reason.",
                }
            }
        },
        "prediction": {
            "rationale_short": "The model treats a partial positive signal as fully supported.",
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
