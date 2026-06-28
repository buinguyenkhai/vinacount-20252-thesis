import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.detector_sft_evaluator import (
    _read_detector_record_eval_rows,
    _score_predictions,
    run_detector_sft_evaluation,
)


class Wave4DetectorSftEvaluatorTest(unittest.TestCase):
    def test_base_model_only_evaluation_writes_base_manifest_without_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            packet = _packet("base-1")
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "base-1",
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

            with (
                patch(
                    "research.detector_sft_evaluator._load_model_bundle",
                    return_value={"device": "cuda"},
                ) as load_model_bundle,
                patch(
                    "research.detector_sft_evaluator._generate_detector_assessment",
                    return_value=_assessment_json(packet, support_level="supported", severity="medium"),
                ),
            ):
                result = run_detector_sft_evaluation(
                    sft_jsonl=sft_jsonl,
                    output_root=output_root,
                    split="validation",
                    base_model_only=True,
                    allow_noncanonical_sft_jsonl=True,
                )

            self.assertEqual(result.status, "passed", result.errors)
            load_model_bundle.assert_called_once()
            self.assertIsNone(load_model_bundle.call_args.kwargs["adapter_dir"])
            self.assertTrue(load_model_bundle.call_args.kwargs["base_model_only"])
            manifest = _read_json(output_root / "manifest.json")
            metrics = _read_json(output_root / "metrics.json")
            self.assertIsNone(manifest["adapter_dir"])
            self.assertEqual(manifest["model_variant"], "qwen3_5_4b_base")
            self.assertEqual(metrics["support_level_exact_match_rate"], 1.0)

    def test_detector_record_evaluation_accepts_validation_only_real_manual_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector_jsonl = root / "real_manual_validation.jsonl"
            packet = _packet("real-1")
            _write_jsonl(
                detector_jsonl,
                [
                    _detector_record(
                        "real-1",
                        split="validation",
                        packet=packet,
                        gold_support_level="weakly_supported",
                        gold_severity="medium",
                    )
                ],
            )
            output_root = root / "eval"

            with (
                patch(
                    "research.detector_sft_evaluator._load_model_bundle",
                    return_value={"device": "cuda"},
                ),
                patch(
                    "research.detector_sft_evaluator._generate_detector_assessment",
                    return_value=_assessment_json(packet, support_level="weakly_supported", severity="medium"),
                ),
            ):
                result = run_detector_sft_evaluation(
                    sft_jsonl=detector_jsonl,
                    output_root=output_root,
                    split="validation",
                    source_format="detector_record",
                )

            self.assertEqual(result.status, "passed", result.errors)
            manifest = _read_json(output_root / "manifest.json")
            metrics = _read_json(output_root / "metrics.json")
            predictions = [
                json.loads(line)
                for line in (output_root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest["source_format"], "detector_record")
            self.assertEqual(manifest["source_split_counts"], {"validation": 1})
            self.assertEqual(manifest["source_type_counts"], {"human_gold_real_report": 1})
            self.assertTrue(manifest["packet_evidence_role_enrichment"]["enabled_for_detector_record_source"])
            self.assertFalse(manifest["packet_evidence_role_enrichment"]["uses_gold_labels"])
            self.assertEqual(manifest["evaluation_policy"]["gold_source"], "output.data in detector record JSONL")
            self.assertTrue(manifest["evaluation_policy"]["not_training_data"])
            visible_packet = predictions[0]["model_visible_input"]["data"]
            self.assertEqual(visible_packet["tool_findings"][0]["evidence_role"], "primary_trigger")
            self.assertFalse(visible_packet["tool_findings"][0]["independent_corroboration_present"])
            self.assertEqual(
                visible_packet["constraints"]["evidence_bundle_semantics"]["tool_finding_strength_scope"],
                "trigger_magnitude_only",
            )
            self.assertEqual(metrics["num_examples"], 1)
            self.assertEqual(metrics["support_level_exact_match_rate"], 1.0)

    def test_detector_record_loader_can_use_evidence_bundle_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detector_jsonl = Path(temp_dir) / "real_manual_validation.jsonl"
            packet = _packet("real-1")
            _write_jsonl(
                detector_jsonl,
                [
                    _detector_record(
                        "real-1",
                        split="validation",
                        packet=packet,
                        gold_support_level="weakly_supported",
                        gold_severity="medium",
                    )
                ],
            )

            rows = _read_detector_record_eval_rows(
                detector_jsonl,
                system_prompt_version="v2_evidence_bundle",
            )

            self.assertIn("complete visible evidence bundle", rows[0].messages[0]["content"])
            self.assertIn("strong isolated signal", rows[0].messages[0]["content"])
            packet = json.loads(rows[0].messages[1]["content"])
            self.assertEqual(packet["tool_findings"][0]["evidence_role"], "primary_trigger")
            self.assertFalse(packet["tool_findings"][0]["independent_corroboration_present"])

    def test_detector_record_loader_can_disable_packet_evidence_role_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            detector_jsonl = Path(temp_dir) / "real_manual_validation.jsonl"
            packet = _packet("real-1")
            _write_jsonl(
                detector_jsonl,
                [
                    _detector_record(
                        "real-1",
                        split="validation",
                        packet=packet,
                        gold_support_level="weakly_supported",
                        gold_severity="medium",
                    )
                ],
            )

            rows = _read_detector_record_eval_rows(
                detector_jsonl,
                system_prompt_version="v2_evidence_bundle",
                packet_evidence_role_enrichment=False,
            )

            packet = json.loads(rows[0].messages[1]["content"])
            self.assertNotIn("evidence_role", json.dumps(packet))
            self.assertNotIn("evidence_bundle_semantics", packet["constraints"])

    def test_evaluation_batches_generation_when_batch_size_exceeds_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            packet_1 = _packet("batch-1")
            packet_2 = _packet("batch-2")
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "batch-1",
                        split="validation",
                        packet=packet_1,
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                    _sft_row(
                        "batch-2",
                        split="validation",
                        packet=packet_2,
                        gold_support_level="weakly_supported",
                        gold_severity="low",
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

            with (
                patch(
                    "research.detector_sft_evaluator._load_model_bundle",
                    return_value={"device": "cuda"},
                ),
                patch(
                    "research.detector_sft_evaluator._generate_detector_assessments",
                    return_value=[
                        _assessment_json(packet_1, support_level="supported", severity="medium"),
                        _assessment_json(packet_2, support_level="weakly_supported", severity="low"),
                    ],
                ) as generate_batch,
            ):
                result = run_detector_sft_evaluation(
                    sft_jsonl=sft_jsonl,
                    output_root=output_root,
                    split="validation",
                    batch_size=2,
                    allow_noncanonical_sft_jsonl=True,
                )

            self.assertEqual(result.status, "passed", result.errors)
            generate_batch.assert_called_once()
            self.assertEqual(len(generate_batch.call_args.args[1]), 2)
            manifest = _read_json(output_root / "manifest.json")
            metrics = _read_json(output_root / "metrics.json")
            predictions = [
                json.loads(line)
                for line in (output_root / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(manifest["batch_size"], 2)
            self.assertEqual(metrics["num_examples"], 2)
            self.assertEqual(metrics["support_level_exact_match_rate"], 1.0)
            self.assertNotIn("_gold_assessment", predictions[0])

    def test_scores_support_severity_contract_and_conservative_false_positive_metrics(self) -> None:
        predictions = [
            _accepted_prediction(
                "supported-1",
                risk_category="earnings_cashflow_quality_risk",
                gold_support_level="supported",
                predicted_support_level="supported",
                gold_severity="high",
                predicted_severity="high",
            ),
            _accepted_prediction(
                "not-supported-1",
                risk_category="disclosure_inconsistency_or_obfuscation",
                gold_support_level="not_supported",
                predicted_support_level="weakly_supported",
                gold_severity="low",
                predicted_severity="medium",
            ),
            _accepted_prediction(
                "insufficient-1",
                risk_category="asset_quality_valuation_risk",
                gold_support_level="insufficient_evidence",
                predicted_support_level="insufficient_evidence",
                gold_severity="unknown",
                predicted_severity="unknown",
            ),
            {
                "example_id": "invalid-1",
                "risk_category": "asset_quality_valuation_risk",
                "gold_support_level": "insufficient_evidence",
                "gold_severity": "unknown",
                "prediction_status": "invalid",
                "schema_valid": False,
                "evidence_valid": False,
                "invalid_reason_codes": ["invalid_json"],
            },
        ]

        metrics = _score_predictions(predictions, split="validation")

        self.assertEqual(metrics["num_examples"], 4)
        self.assertEqual(metrics["num_valid_predictions"], 3)
        self.assertEqual(metrics["num_invalid_responses"], 1)
        self.assertEqual(metrics["schema_valid_count"], 3)
        self.assertEqual(metrics["evidence_id_valid_count"], 3)
        self.assertEqual(metrics["support_level_exact_match_count"], 2)
        self.assertEqual(metrics["support_level_exact_match_rate"], 0.5)
        category_metrics = metrics["support_metrics_by_risk_category"]["earnings_cashflow_quality_risk"]
        self.assertEqual(category_metrics["num_examples"], 1)
        self.assertEqual(category_metrics["support_level_exact_match_rate"], 1.0)
        self.assertEqual(
            category_metrics["support_level_class_metrics"]["supported"]["recall"],
            1.0,
        )
        self.assertEqual(metrics["support_level_exact_match_rate_on_valid"], 2 / 3)
        self.assertEqual(metrics["severity_exact_match_count"], 2)
        self.assertEqual(metrics["severity_exact_match_rate"], 0.5)
        self.assertEqual(
            metrics["false_positive_rejection"],
            {"correct_or_conservative": 2, "total": 3, "rate": 2 / 3},
        )
        self.assertEqual(
            metrics["insufficient_evidence_detection"],
            {"true_positive": 1, "predicted_total": 1, "gold_total": 2, "recall": 0.5, "precision": 1.0},
        )
        self.assertEqual(
            metrics["support_level_confusion_matrix_counts"],
            {
                "insufficient_evidence": {"insufficient_evidence": 1},
                "not_supported": {"weakly_supported": 1},
                "supported": {"supported": 1},
            },
        )
        self.assertEqual(metrics["invalid_reason_counts"], {"invalid_json": 1})
        self.assertEqual(metrics["by_gold_support_level"]["supported"]["support_level_exact_match_rate"], 1.0)
        self.assertEqual(metrics["by_risk_category"]["asset_quality_valuation_risk"]["invalid"], 1)
        self.assertAlmostEqual(metrics["support_level_macro_f1"], 5 / 12)
        self.assertEqual(
            metrics["support_level_class_metrics"]["weakly_supported"],
            {"true_positive": 0, "predicted_total": 1, "gold_total": 0, "precision": 0.0, "recall": None, "f1": 0.0},
        )
        self.assertEqual(metrics["weakly_supported_detection"], metrics["support_level_class_metrics"]["weakly_supported"])
        self.assertAlmostEqual(metrics["severity_macro_f1"], 5 / 12)


def _accepted_prediction(
    example_id: str,
    *,
    risk_category: str,
    gold_support_level: str,
    predicted_support_level: str,
    gold_severity: str,
    predicted_severity: str,
) -> dict:
    return {
        "example_id": example_id,
        "risk_category": risk_category,
        "gold_support_level": gold_support_level,
        "gold_severity": gold_severity,
        "prediction_status": "accepted",
        "schema_valid": True,
        "evidence_valid": True,
        "predicted_support_level": predicted_support_level,
        "predicted_severity": predicted_severity,
        "support_level_exact_match": predicted_support_level == gold_support_level,
        "severity_exact_match": predicted_severity == gold_severity,
        "assessment_exact_match": False,
    }


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


def _detector_record(
    example_id: str,
    *,
    split: str,
    packet: dict,
    gold_support_level: str,
    gold_severity: str,
) -> dict:
    return {
        "dataset_version": "tdf_v1.0.0",
        "example_id": example_id,
        "input": {"type": "DetectorPacket", "data": packet},
        "metadata": {
            "risk_category": packet["task"]["risk_category"],
            "severity": gold_severity,
            "split_metadata": {"split": split, "usable_for_training": False},
            "support_level": gold_support_level,
        },
        "output": {
            "type": "DetectorAssessment",
            "data": json.loads(_assessment_json(packet, support_level=gold_support_level, severity=gold_severity)),
        },
        "source_type": "human_gold_real_report",
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
