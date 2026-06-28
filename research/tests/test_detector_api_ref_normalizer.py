import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.detector_api_ref_normalizer import (
    run_detector_api_ref_normalizer,
)


class Wave4DetectorApiRefNormalizerTest(unittest.TestCase):
    def test_repairs_unique_suffix_ref_ids_and_rescores_with_sft_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            output_root = root / "normalized"
            packet = _packet("repair-1")
            _write_source_eval(
                source_root,
                predictions=[
                    _invalid_prediction(
                        "repair-1",
                        packet=packet,
                        gold_support_level="weakly_supported",
                        gold_severity="medium",
                    )
                ],
                raw_invalid=[
                    _raw_invalid_response(
                        "repair-1",
                        packet=packet,
                        raw_response_text=_assessment_json(
                            packet,
                            support_level="weakly_supported",
                            severity="medium",
                            ref_id="row-1",
                        ),
                    )
                ],
            )

            result = run_detector_api_ref_normalizer(
                source_eval_root=source_root,
                output_root=output_root,
                variant="deepseek_fixture_ref_normalized",
            )

            self.assertEqual(result.status, "passed", result.errors)
            predictions = _read_jsonl(result.predictions_path)
            metrics = _read_json(result.metrics_path)
            manifest = _read_json(result.manifest_path)
            repairs = _read_jsonl(result.repair_records_path)
            self.assertEqual(predictions[0]["prediction_status"], "accepted")
            self.assertEqual(predictions[0]["ref_normalizer_actions"], ["repaired_evidence_ref_id"])
            self.assertEqual(predictions[0]["ref_normalizer_repaired_ref_count"], 2)
            self.assertEqual(
                predictions[0]["prediction"]["cited_evidence_refs"],
                [{"evidence_ref_type": "table_row", "ref_id": "REPORT_1:row-1", "role": "supporting"}],
            )
            self.assertEqual(metrics["num_valid_predictions"], 1)
            self.assertEqual(metrics["num_invalid_responses"], 0)
            self.assertEqual(metrics["support_level_exact_match_rate"], 1.0)
            self.assertEqual(
                metrics["ref_normalizer_counts"],
                {"actions": {"repaired_evidence_ref_id": 1}, "repaired_records": 1, "repaired_ref_count": 2},
            )
            self.assertEqual(repairs[0]["changes"][0]["old_ref_id"], "row-1")
            self.assertEqual(repairs[0]["changes"][0]["new_ref_id"], "REPORT_1:row-1")
            self.assertFalse(repairs[0]["support_level_or_severity_overrides"])
            self.assertEqual(manifest["variant"], "deepseek_fixture_ref_normalized")
            self.assertFalse(manifest["ref_normalizer_policy"]["support_level_or_severity_overrides"])

    def test_unrepairable_refs_remain_invalid_and_keep_raw_debug_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            output_root = root / "normalized"
            packet = _packet("bad-1")
            _write_source_eval(
                source_root,
                predictions=[
                    _invalid_prediction(
                        "bad-1",
                        packet=packet,
                        gold_support_level="supported",
                        gold_severity="medium",
                    )
                ],
                raw_invalid=[
                    _raw_invalid_response(
                        "bad-1",
                        packet=packet,
                        raw_response_text=_assessment_json(
                            packet,
                            support_level="supported",
                            severity="medium",
                            ref_id="not-visible",
                        ),
                    )
                ],
            )

            result = run_detector_api_ref_normalizer(
                source_eval_root=source_root,
                output_root=output_root,
                variant="deepseek_fixture_ref_normalized",
            )

            self.assertEqual(result.status, "passed", result.errors)
            predictions = _read_jsonl(result.predictions_path)
            self.assertEqual(predictions[0]["prediction_status"], "invalid")
            self.assertEqual(predictions[0]["invalid_reason_codes"], ["invalid_evidence_ids"])
            self.assertEqual(_read_jsonl(result.repair_records_path), [])
            invalid_responses = _read_jsonl(result.invalid_responses_path)
            self.assertEqual(invalid_responses[0]["raw_response_debug_path"], "debug/raw_invalid_responses.jsonl")
            raw_invalid = _read_jsonl(output_root / "debug" / "raw_invalid_responses.jsonl")
            self.assertEqual(raw_invalid[0]["raw_response_text"].count("not-visible"), 2)

    def test_public_command_writes_normalized_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source"
            output_root = root / "normalized"
            packet = _packet("cmd-1")
            _write_source_eval(
                source_root,
                predictions=[
                    _invalid_prediction(
                        "cmd-1",
                        packet=packet,
                        gold_support_level="supported",
                        gold_severity="medium",
                    )
                ],
                raw_invalid=[
                    _raw_invalid_response(
                        "cmd-1",
                        packet=packet,
                        raw_response_text=_assessment_json(
                            packet,
                            support_level="supported",
                            severity="medium",
                            ref_id="row-1",
                        ),
                    )
                ],
            )

            completed = _run_command(
                "--source-eval-root",
                str(source_root),
                "--output-root",
                str(output_root),
                "--variant",
                "deepseek_fixture_ref_normalized",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "passed")
            self.assertTrue((output_root / "predictions.jsonl").exists())
            self.assertTrue((output_root / "metrics.json").exists())
            self.assertTrue((output_root / "manifest.json").exists())
            self.assertTrue((output_root / "repair_records.jsonl").exists())


def _write_source_eval(path: Path, *, predictions: list[dict], raw_invalid: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "debug").mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"status": "passed", "split": "validation", "predictions": "predictions.jsonl"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(json.dumps({"status": "source_fixture"}, sort_keys=True) + "\n", encoding="utf-8")
    (path / "predictions.jsonl").write_text(
        "".join(json.dumps(prediction, sort_keys=True) + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    (path / "invalid_responses.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "example_id": prediction["example_id"],
                    "packet_id": prediction["packet_id"],
                    "candidate_id": prediction["model_visible_input"]["data"]["candidate_id"],
                    "failure_stage": "response_validation",
                    "reason_codes": ["invalid_evidence_ids"],
                    "raw_response_debug_path": "debug/raw_invalid_responses.jsonl",
                },
                sort_keys=True,
            )
            + "\n"
            for prediction in predictions
        ),
        encoding="utf-8",
    )
    (path / "debug" / "raw_invalid_responses.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in raw_invalid),
        encoding="utf-8",
    )


def _invalid_prediction(
    example_id: str,
    *,
    packet: dict,
    gold_support_level: str,
    gold_severity: str,
) -> dict:
    return {
        "example_id": example_id,
        "split": "validation",
        "packet_id": packet["packet_id"],
        "risk_category": packet["task"]["risk_category"],
        "gold_support_level": gold_support_level,
        "gold_severity": gold_severity,
        "prediction_status": "invalid",
        "schema_valid": False,
        "evidence_valid": False,
        "invalid_reason_codes": ["invalid_evidence_ids"],
        "model_visible_input": {"type": "DetectorPacket", "data": packet},
    }


def _raw_invalid_response(example_id: str, *, packet: dict, raw_response_text: str) -> dict:
    return {
        "example_id": example_id,
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "raw_response_text": raw_response_text,
        "reason_codes": ["invalid_evidence_ids"],
        "non_canonical": True,
        "debug_only": True,
        "not_labels": True,
        "not_training_data": True,
        "not_detector_visible_data": True,
    }


def _packet(example_id: str) -> dict:
    return {
        "packet_id": f"PACKET_{example_id}",
        "candidate_id": f"CAND_{example_id}",
        "report_id": "REPORT_1",
        "task": {"risk_category": "earnings_cashflow_quality_risk"},
        "metadata": {},
        "candidate_summary": {
            "priority": "medium",
            "reason_for_candidate": "Fixture candidate.",
            "supporting_signal_ids": ["signal-1"],
            "evidence_refs": [{"evidence_ref_type": "table_row", "ref_id": "REPORT_1:row-1", "role": "supporting"}],
        },
        "relevant_table_rows": [{"report_id": "REPORT_1", "row_id": "row-1", "values": {}}],
        "relevant_notes": [],
        "relevant_variance_explanations": [],
        "tool_findings": [
            {
                "tool_result_id": "tool-1",
                "report_id": "REPORT_1",
                "tool_name": "fixture_tool",
                "risk_category": "earnings_cashflow_quality_risk",
                "signal_id": "signal-1",
                "flag": True,
                "finding_summary": "Fixture signal.",
                "evidence_refs": [{"evidence_ref_type": "table_row", "ref_id": "REPORT_1:row-1", "role": "supporting"}],
            }
        ],
        "rules": [{"rule_id": "rule-1", "related_signal_ids": ["signal-1"]}],
        "constraints": {},
    }


def _assessment_json(packet: dict, *, support_level: str, severity: str, ref_id: str) -> str:
    return json.dumps(
        {
            "assessment_id": f"ASSESS_{packet['packet_id']}",
            "packet_id": packet["packet_id"],
            "candidate_id": packet["candidate_id"],
            "report_id": packet["report_id"],
            "risk_category": packet["task"]["risk_category"],
            "support_level": support_level,
            "confidence": 0.8,
            "severity": severity,
            "validated_signals": [
                {
                    "signal_id": "signal-1",
                    "status": "partially_validated" if support_level == "weakly_supported" else "validated",
                    "support_level": support_level,
                    "tool_result_id": "tool-1",
                    "cited_evidence_refs": [{"evidence_ref_type": "table_row", "ref_id": ref_id, "role": "supporting"}],
                }
            ],
            "cited_evidence_refs": [{"evidence_ref_type": "table_row", "ref_id": ref_id, "role": "supporting"}],
            "rationale_short": "Fixture assessment.",
        },
        sort_keys=True,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_command(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.detector_api_ref_normalizer", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
