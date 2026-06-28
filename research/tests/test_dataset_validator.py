import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from research.dataset_traceability import (
    apply_report_artifact_traceability,
    build_source_report_manifest,
)


def _valid_packet() -> dict:
    return {
        "packet_id": "PACKET_FPT_2024_Q3_001",
        "candidate_id": "CAND_FPT_2024_Q3_001",
        "report_id": "FPT_2024_Q3",
        "task": {
            "risk_category": "revenue_income_recognition_risk",
            "question": "Does the provided evidence support the candidate risk signal?",
            "expected_output": "Return a structured DetectorAssessment.",
        },
        "metadata": {
            "company_name": "FPT Corporation",
            "ticker": "FPT",
            "period": "2024-Q3",
            "report_profile": "standard_corporate",
            "currency": "VND",
            "unit": "million_vnd",
            "language": "vi",
        },
        "candidate_summary": {
            "risk_category": "revenue_income_recognition_risk",
            "reason_for_candidate": "Revenue increased while receivables increased faster than revenue.",
            "priority": "high",
        },
        "relevant_table_rows": [
            {
                "row_id": "IS_2024_Q3:revenue",
                "report_id": "FPT_2024_Q3",
                "account_standard": "revenue",
                "values": {
                    "current": {"period": "2024_Q3", "value": 125000, "cell_id": "CELL_REVENUE_CURRENT"},
                    "prior": {"period": "2023_Q3", "value": 100000, "cell_id": "CELL_REVENUE_PRIOR"},
                },
            }
        ],
        "relevant_notes": [
            {"note_id": "NOTE_REV_1", "report_id": "FPT_2024_Q3", "text": "Revenue note excerpt."}
        ],
        "relevant_variance_explanations": [
            {"span_id": "SPAN_VAR_1", "report_id": "FPT_2024_Q3", "text": "Revenue variance explanation."}
        ],
        "tool_findings": [
            {
                "tool_result_id": "TOOL_REV_GROWTH_1",
                "tool_name": "revenue_growth_tool",
                "risk_category": "revenue_income_recognition_risk",
                "signal_id": "revenue_growth_high",
                "flag": True,
                "strength": "moderate",
                "finding_summary": "Revenue increased above the configured threshold.",
                "evidence_refs": [
                    {"evidence_ref_type": "table_cell", "ref_id": "FPT_2024_Q3:CELL_REVENUE_CURRENT"}
                ],
            }
        ],
        "rules": [
            {
                "rule_id": "RULE_REV_GROWTH",
                "description": "Revenue growth above threshold may support a revenue quality risk signal.",
            }
        ],
        "constraints": {
            "allowed_decisions": [
                "supported",
                "weakly_supported",
                "not_supported",
                "insufficient_evidence",
            ],
            "evidence_must_reference_provided_ids": True,
            "do_not_claim_fraud": True,
            "max_rationale_sentences": 3,
        },
    }


def _valid_assessment() -> dict:
    return {
        "assessment_id": "ASSESS_PACKET_FPT_2024_Q3_001",
        "packet_id": "PACKET_FPT_2024_Q3_001",
        "candidate_id": "CAND_FPT_2024_Q3_001",
        "report_id": "FPT_2024_Q3",
        "risk_category": "revenue_income_recognition_risk",
        "support_level": "supported",
        "confidence": 0.82,
        "severity": "medium",
        "validated_signals": [
            {
                "signal_id": "revenue_growth_high",
                "tool_result_id": "TOOL_REV_GROWTH_1",
                "status": "validated",
                "support_level": "supported",
                "cited_evidence_refs": [
                    {
                        "evidence_ref_type": "table_cell",
                        "ref_id": "FPT_2024_Q3:CELL_REVENUE_CURRENT",
                        "role": "supporting",
                    }
                ],
            }
        ],
        "cited_evidence_refs": [
            {"evidence_ref_type": "tool_result", "ref_id": "TOOL_REV_GROWTH_1", "role": "supporting"},
            {
                "evidence_ref_type": "table_cell",
                "ref_id": "FPT_2024_Q3:CELL_REVENUE_CURRENT",
                "role": "supporting",
            },
            {"evidence_ref_type": "note", "ref_id": "FPT_2024_Q3:NOTE_REV_1", "role": "context"},
            {
                "evidence_ref_type": "variance_explanation_span",
                "ref_id": "FPT_2024_Q3:SPAN_VAR_1",
                "role": "context",
            },
            {"evidence_ref_type": "rule", "ref_id": "RULE_REV_GROWTH", "role": "supporting"},
        ],
        "rationale_short": "The cited packet evidence supports a revenue quality risk signal.",
    }


def _valid_record() -> dict:
    record = {
        "example_id": "SFT_SYN_000001",
        "dataset_version": "tdf_v1.0.0",
        "source_type": "synthetic_injected_filtered",
        "input": {"type": "DetectorPacket", "data": _valid_packet()},
        "output": {"type": "DetectorAssessment", "data": _valid_assessment()},
        "metadata": {
            "risk_category": "revenue_income_recognition_risk",
            "support_level": "supported",
            "severity": "medium",
            "report_profile": "standard_corporate",
            "report_period_type": "quarterly",
            "language": "vi",
            "evidence_profile": {},
            "generation_metadata": {},
            "validation_metadata": {},
            "split_metadata": {"split": "train"},
            "audit_metadata": {},
        },
    }
    return apply_report_artifact_traceability(record, _artifact_traceability())


def _artifact_traceability() -> dict:
    return {
        "company_key": "FPT",
        "period_key": "2024_Q3",
        "group_key": "FPT_2024_Q3",
        "source_file_sha256": "sha256-source-pdf",
        "normalized_text_hash": "sha256-normalized-text",
        "table_content_hash": "sha256-table-content",
        "derived_from_report_artifact_id": "RPTMEM_FPT_2024_Q3_v1",
        "derived_from_source_document_id": "DOC_FPT_2024_Q3_FS_001",
    }


def _write_dataset(root: Path, records: list[dict]) -> None:
    (root / "train.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "dataset_version": "tdf_v1.0.0",
        "created_at": "2026-01-15T10:30:00Z",
        "created_by": "tests",
        "num_examples_total": len(records),
        "num_examples_by_split": {"train": len(records)},
        "num_examples_by_source_type": {"synthetic_injected_filtered": len(records)},
        "num_examples_by_support_level": {"supported": len(records)},
        "num_examples_by_report_profile": {"standard_corporate": len(records)},
        "num_examples_by_risk_category": {"revenue_income_recognition_risk": len(records)},
        "schema_versions": {
            "DetectorPacket": "v1.0.0",
            "DetectorAssessment": "v1.0.0",
            "CandidateRisk": "v1.0.0",
            "ToolFinding": "v1.0.0",
        },
        "validation_pipeline_version": "validator_v0.1.0",
        "generation_pipeline_version": "synthetic_generator_v0.1.0",
        "source_report_traceability": build_source_report_manifest(records),
        "known_limitations": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _run_validator(dataset_dir: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.dataset_validator", str(dataset_dir)],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": pythonpath,
        },
        check=False,
    )


class Wave4DatasetValidatorPublicCommandTest(unittest.TestCase):
    def test_public_api_copies_artifact_traceability_without_changing_detector_packet(self):
        record = {
            **_valid_record(),
            "metadata": {
                **_valid_record()["metadata"],
                "split_metadata": {"split": "train"},
                "audit_metadata": {},
            },
        }
        original_packet = deepcopy(record["input"]["data"])

        copied = apply_report_artifact_traceability(record, _artifact_traceability())

        self.assertEqual(copied["input"]["data"], original_packet)
        self.assertEqual(
            copied["metadata"]["split_metadata"],
            {
                "split": "train",
                "company_key": "FPT",
                "period_key": "2024_Q3",
                "group_key": "FPT_2024_Q3",
                "source_file_sha256": "sha256-source-pdf",
                "normalized_text_hash": "sha256-normalized-text",
                "table_content_hash": "sha256-table-content",
                "derived_from_report_artifact_id": "RPTMEM_FPT_2024_Q3_v1",
                "derived_from_source_document_id": "DOC_FPT_2024_Q3_FS_001",
            },
        )
        self.assertEqual(
            copied["metadata"]["audit_metadata"]["dataset_artifact_traceability"],
            {
                "derived_from_report_artifact_id": "RPTMEM_FPT_2024_Q3_v1",
                "derived_from_source_document_id": "DOC_FPT_2024_Q3_FS_001",
            },
        )

    def test_public_command_rejects_missing_required_traceability_before_release(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            del record["metadata"]["split_metadata"]["source_file_sha256"]
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("split_metadata is missing required traceability field source_file_sha256", completed.stdout)

    def test_manifest_traceability_summary_groups_records_without_artifact_bodies(self):
        first = _valid_record()
        second = deepcopy(_valid_record())
        second["example_id"] = "SFT_SYN_000002"
        second["metadata"]["split_metadata"]["split"] = "validation"
        second["metadata"]["split_metadata"]["raw_ocr_text"] = "must not appear"

        summary = build_source_report_manifest([first, second])

        self.assertEqual(
            summary,
            {
                "report_artifacts": [
                    {
                        "derived_from_report_artifact_id": "RPTMEM_FPT_2024_Q3_v1",
                        "derived_from_source_document_id": "DOC_FPT_2024_Q3_FS_001",
                        "company_key": "FPT",
                        "period_key": "2024_Q3",
                        "group_key": "FPT_2024_Q3",
                        "source_file_sha256": "sha256-source-pdf",
                        "normalized_text_hash": "sha256-normalized-text",
                        "table_content_hash": "sha256-table-content",
                        "example_count": 2,
                        "splits": ["train", "validation"],
                    }
                ]
            },
        )
        self.assertNotIn("raw_ocr_text", json.dumps(summary))

    def test_public_command_rejects_missing_or_stale_manifest_traceability_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_dataset(dataset_dir, [_valid_record()])
            manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
            del manifest["source_report_traceability"]
            (dataset_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("manifest source_report_traceability is required", completed.stdout)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_dataset(dataset_dir, [_valid_record()])
            manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["source_report_traceability"]["report_artifacts"][0]["example_count"] = 3
            (dataset_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("manifest source_report_traceability does not match actual records", completed.stdout)

    def test_public_command_accepts_minimal_valid_detector_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_dataset(dataset_dir, [_valid_record()])

            completed = _run_validator(dataset_dir)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('"status": "passed"', completed.stdout)

    def test_public_command_accepts_packet_visible_table_row_evidence_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            record["output"]["data"]["cited_evidence_refs"] = [
                {
                    "evidence_ref_type": "table_row",
                    "ref_id": "FPT_2024_Q3:IS_2024_Q3:revenue",
                    "role": "supporting",
                }
            ]
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_public_command_accepts_decimal_values_inside_concise_rationale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            record["output"]["data"]["rationale_short"] = (
                "Profit grew 40% while operating cash flow declined 54.1%, yielding a "
                "cash-to-profit ratio of 0.35. This supports an earnings quality risk signal."
            )
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_public_command_rejects_malformed_jsonl_and_wrong_top_level_wrapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_dataset(dataset_dir, [_valid_record()])
            (dataset_dir / "train.jsonl").write_text("{not json}\n", encoding="utf-8")

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("malformed JSONL", completed.stdout)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            record["input"]["type"] = "ReportMemory"
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("input wrapper type must be DetectorPacket", completed.stdout)

    def test_public_command_rejects_invalid_packet_and_assessment_schema_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            del record["input"]["data"]["candidate_id"]
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("DetectorPacket is missing fields", completed.stdout)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            del record["output"]["data"]["validated_signals"]
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("DetectorAssessment is missing fields", completed.stdout)

    def test_public_command_rejects_id_mismatch_risk_mismatch_and_detector_value_constraints(self):
        cases = [
            ("packet_id", "OTHER_PACKET", "packet_id must match"),
            ("risk_category", "asset_quality_valuation_risk", "risk_category must match"),
            ("support_level", "confirmed", "support_level is not allowed"),
            ("severity", "critical", "severity is not allowed"),
            ("confidence", 1.1, "confidence must be between 0 and 1"),
            (
                "rationale_short",
                "Sentence one. Sentence two. Sentence three. Sentence four.",
                "rationale_short must be 1-3 concise sentences",
            ),
        ]
        for field, value, expected_error in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                dataset_dir = Path(temp_dir)
                record = _valid_record()
                record["output"]["data"][field] = value
                _write_dataset(dataset_dir, [record])

                completed = _run_validator(dataset_dir)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stdout)

    def test_public_command_rejects_assessment_evidence_refs_not_present_in_packet(self):
        ref_cases = [
            ("table_cell", "FPT_2024_Q3:MISSING_CELL"),
            ("note", "FPT_2024_Q3:MISSING_NOTE"),
            ("variance_explanation_span", "FPT_2024_Q3:MISSING_SPAN"),
            ("tool_result", "MISSING_TOOL"),
            ("rule", "MISSING_RULE"),
        ]
        for ref_type, ref_id in ref_cases:
            with self.subTest(ref_type=ref_type), tempfile.TemporaryDirectory() as temp_dir:
                dataset_dir = Path(temp_dir)
                record = _valid_record()
                record["output"]["data"]["cited_evidence_refs"] = [
                    {"evidence_ref_type": ref_type, "ref_id": ref_id, "role": "supporting"}
                ]
                _write_dataset(dataset_dir, [record])

                completed = _run_validator(dataset_dir)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("cited evidence must be visible", completed.stdout)

    def test_public_command_rejects_prohibited_language_but_allows_conservative_risk_signal_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            record["output"]["data"]["rationale_short"] = "The evidence proves intentional fraud."
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("prohibited legal or misconduct wording", completed.stdout)

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            record = _valid_record()
            record["output"]["data"]["rationale_short"] = (
                "The cited evidence supports a conservative revenue quality risk signal."
            )
            _write_dataset(dataset_dir, [record])

            completed = _run_validator(dataset_dir)

            self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_public_command_rejects_detector_visible_hidden_metadata_and_raw_context_leakage(self):
        cases = [
            ("hidden_injection_details", {"modified_cells": []}),
            ("cache_record_id", "CACHE_1"),
            ("source_file_sha256", "sha256-source-pdf"),
            ("normalized_text_hash", "sha256-normalized-text"),
            ("table_content_hash", "sha256-table-content"),
            ("raw_ocr_text", "raw OCR text"),
            ("raw_pdf_coordinates", {"x0": 1}),
            ("sampling_pool", "ordinary_filing_evaluation_pool"),
            ("correction_amendment_provenance_type", "directed_correction_package"),
            ("ordinary_provenance", "ordinary-control sampling metadata"),
            ("zip_container_sha256", "sha256-container"),
            ("selected_member_sha256", "sha256-member"),
            ("omitted_evidence_ids", ["NOTE_2"]),
            ("omitted_evidence_summaries", ["extra note existed"]),
            ("omitted_evidence_count", 3),
            ("external_context", "outside context"),
            ("long_reasoning", "private reasoning"),
            ("chain_of_thought", "hidden chain"),
        ]
        for key, value in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                dataset_dir = Path(temp_dir)
                record = _valid_record()
                record["input"]["data"]["metadata"][key] = value
                _write_dataset(dataset_dir, [record])

                completed = _run_validator(dataset_dir)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("detector-visible data contains prohibited hidden or raw metadata", completed.stdout)

    def test_public_command_validates_manifest_counts_against_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            records = [_valid_record(), deepcopy(_valid_record())]
            records[1]["example_id"] = "SFT_SYN_000002"
            records[1]["output"]["data"]["assessment_id"] = "ASSESS_PACKET_FPT_2024_Q3_002"
            records[1]["source_type"] = "manual_edge_case"
            records[1]["metadata"]["support_level"] = "weakly_supported"
            records[1]["metadata"]["report_profile"] = "securities"
            records[1]["metadata"]["risk_category"] = "asset_quality_valuation_risk"
            _write_dataset(dataset_dir, records)

            completed = _run_validator(dataset_dir)

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("num_examples_by_source_type", completed.stdout)
            self.assertIn("num_examples_by_support_level", completed.stdout)
            self.assertIn("num_examples_by_report_profile", completed.stdout)
            self.assertIn("num_examples_by_risk_category", completed.stdout)


if __name__ == "__main__":
    unittest.main()
