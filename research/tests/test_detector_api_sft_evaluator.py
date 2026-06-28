import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.api_llm_detector_baseline import (
    ApiLlmDetectorRequest,
    ApiLlmDetectorResponse,
)
from research.detector_api_sft_evaluator import (
    run_detector_api_sft_evaluation,
)


class Wave4DetectorApiSftEvaluatorTest(unittest.TestCase):
    def test_fake_mode_writes_sft_scorer_compatible_metrics_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            packet = _packet("api-1")
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "api-1",
                        split="validation",
                        packet=packet,
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                    _sft_row(
                        "train-1",
                        split="train",
                        packet=_packet("train-1"),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                ],
            )
            output_root = root / "eval"

            result = run_detector_api_sft_evaluation(
                sft_jsonl=sft_jsonl,
                output_root=output_root,
                mode="fake",
                model="deepseek-v4-flash",
                allow_noncanonical_sft_jsonl=True,
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.records_loaded, 1)
            self.assertEqual(result.predictions_written, 1)
            self.assertEqual(result.invalid_response_count, 0)

            manifest = _read_json(output_root / "manifest.json")
            self.assertEqual(manifest["provider"], "deepseek")
            self.assertEqual(manifest["model"], "deepseek-v4-flash")
            self.assertEqual(manifest["mode"], "fake")
            self.assertEqual(manifest["model_variant"], "api_deepseek_deepseek_v4_flash")
            self.assertFalse(manifest["artifact_policy"]["raw_valid_responses_canonical"])
            self.assertTrue(manifest["artifact_policy"]["predictions_are_evaluation_outputs_only"])

            predictions = _read_jsonl(output_root / "predictions.jsonl")
            self.assertEqual(predictions[0]["prediction_status"], "accepted")
            self.assertEqual(predictions[0]["provider_metadata"]["strategy"], "sft_gold_fake_client")
            self.assertNotIn("raw_completion", predictions[0])

            metrics = _read_json(output_root / "metrics.json")
            self.assertEqual(metrics["split"], "validation")
            self.assertEqual(metrics["num_examples"], 1)
            self.assertEqual(metrics["num_valid_predictions"], 1)
            self.assertEqual(metrics["support_level_exact_match_rate"], 1.0)
            self.assertEqual(metrics["severity_exact_match_rate"], 1.0)

    def test_invalid_api_response_is_counted_and_raw_text_is_debug_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "bad-1",
                        split="validation",
                        packet=_packet("bad-1"),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                    _sft_row(
                        "train-1",
                        split="train",
                        packet=_packet("train-1"),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                ],
            )
            output_root = root / "eval"

            result = run_detector_api_sft_evaluation(
                sft_jsonl=sft_jsonl,
                output_root=output_root,
                mode="fake",
                model="deepseek-v4-pro",
                client=InvalidJsonClient(),
                allow_noncanonical_sft_jsonl=True,
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.invalid_response_count, 1)
            predictions = _read_jsonl(output_root / "predictions.jsonl")
            self.assertEqual(predictions[0]["prediction_status"], "invalid")
            self.assertEqual(predictions[0]["invalid_reason_codes"], ["invalid_json"])
            self.assertNotIn("raw_response_text", predictions[0])
            invalid_responses = _read_jsonl(output_root / "invalid_responses.jsonl")
            self.assertEqual(invalid_responses[0]["raw_response_debug_path"], "debug/raw_invalid_responses.jsonl")
            raw_invalid = _read_jsonl(output_root / "debug" / "raw_invalid_responses.jsonl")
            self.assertEqual(raw_invalid[0]["raw_response_text"], "not-json")
            self.assertTrue(raw_invalid[0]["debug_only"])

    def test_public_command_writes_fake_eval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "cmd-1",
                        split="validation",
                        packet=_packet("cmd-1"),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                    _sft_row(
                        "train-1",
                        split="train",
                        packet=_packet("train-1"),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                ],
            )
            output_root = root / "eval"

            completed = _run_command(
                "--sft-jsonl",
                str(sft_jsonl),
                "--output-root",
                str(output_root),
                "--mode",
                "fake",
                "--model",
                "deepseek-v4-flash",
                "--allow-noncanonical-sft-jsonl",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            stdout = json.loads(completed.stdout)
            self.assertEqual(stdout["status"], "passed")
            self.assertEqual(stdout["records_loaded"], 1)
            self.assertTrue((output_root / "predictions.jsonl").exists())
            self.assertTrue((output_root / "metrics.json").exists())
            self.assertTrue((output_root / "manifest.json").exists())


class InvalidJsonClient:
    def complete(self, request: ApiLlmDetectorRequest) -> ApiLlmDetectorResponse:
        return ApiLlmDetectorResponse(content="not-json")


def _sft_row(
    example_id: str,
    *,
    split: str,
    packet: dict,
    gold_support_level: str,
    gold_severity: str,
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "locked system prompt"},
            {"role": "user", "content": json.dumps(packet, sort_keys=True)},
            {"role": "assistant", "content": _assessment_json(packet, support_level=gold_support_level, severity=gold_severity)},
        ],
        "metadata": {
            "example_id": example_id,
            "source_type": "synthetic_injected_filtered",
            "split": split,
        },
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


def _assessment_json(packet: dict, *, support_level: str, severity: str) -> str:
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
                    "status": "validated",
                    "support_level": support_level,
                    "tool_result_id": "tool-1",
                    "cited_evidence_refs": [{"evidence_ref_type": "table_row", "ref_id": "REPORT_1:row-1", "role": "supporting"}],
                }
            ],
            "cited_evidence_refs": [{"evidence_ref_type": "table_row", "ref_id": "REPORT_1:row-1", "role": "supporting"}],
            "rationale_short": "Fixture assessment.",
        },
        sort_keys=True,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_command(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.detector_api_sft_evaluator", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
