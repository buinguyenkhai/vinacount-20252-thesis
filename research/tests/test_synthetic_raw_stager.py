import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from research.tests.test_synthetic_detector_assessment_gate import _raw_record


class Wave4SyntheticRawStagerPublicCommandTest(unittest.TestCase):
    def test_public_command_writes_canonical_synthetic_raw_gate_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staged_jsonl = root / "staged_packets.jsonl"
            output_dir = root / "raw"
            record = _raw_record()
            _add_traceability(record)
            _write_jsonl(staged_jsonl, [record])

            completed = _run_stager(
                "--input-jsonl",
                str(staged_jsonl),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "raw-smoke",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            raw_path = output_dir / "synthetic_injected_raw.jsonl"
            manifest_path = output_dir / "manifest.json"
            metrics_path = output_dir / "metrics.json"
            self.assertTrue(raw_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(metrics_path.exists())
            raw_records = _read_jsonl(raw_path)
            self.assertEqual(len(raw_records), 1)
            self.assertEqual(raw_records[0]["source_type"], "synthetic_injected_raw")
            self.assertNotIn("output", raw_records[0])
            self.assertEqual(raw_records[0]["input"], record["input"])
            self.assertEqual(raw_records[0]["metadata"]["generation_metadata"]["base_report_id"], "FPT_2024_Q3_CLEAN")
            self.assertEqual(
                raw_records[0]["metadata"]["split_metadata"]["source_file_sha256"],
                "sha256-source-FPT_2024_Q3_SYN_REV_001",
            )
            self.assertNotIn("source_file_sha256", json.dumps(raw_records[0]["input"], sort_keys=True))

            manifest = _read_json(manifest_path)
            self.assertEqual(manifest["run_id"], "raw-smoke")
            self.assertEqual(manifest["artifact_contract_version"], "synthetic_injected_raw_staging_v1")
            self.assertEqual(manifest["output_jsonl"], str(raw_path))

            metrics = _read_json(metrics_path)
            self.assertEqual(metrics["records_written"], 1)
            self.assertEqual(metrics["source_types"], {"synthetic_injected_raw": 1})

            gate_check = _run_gate_dry_run(raw_path, root / "gate")
            self.assertEqual(gate_check.returncode, 0, gate_check.stderr)

    def test_public_command_rejects_records_with_outputs_or_real_manual_source_type(self) -> None:
        cases = [
            ("already_labeled", lambda record: record.update({"output": {"type": "DetectorAssessment", "data": {}}}), "must not contain output"),
            ("detector_assessment", lambda record: record.update({"DetectorAssessment": {}}), "label-like target"),
            ("assessment", lambda record: record.update({"assessment": {"support_level": "supported"}}), "label-like target"),
            ("label", lambda record: record.update({"label": "supported"}), "label-like target"),
            ("real_manual", lambda record: record.update({"source_type": "human_gold_real_report"}), "synthetic_injected_raw"),
        ]
        for _, mutate, expected_error in cases:
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                staged_jsonl = root / "staged_packets.jsonl"
                record = _raw_record()
                _add_traceability(record)
                mutate(record)
                _write_jsonl(staged_jsonl, [record])

                completed = _run_stager(
                    "--input-jsonl",
                    str(staged_jsonl),
                    "--output-dir",
                    str(root / "raw"),
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stdout)
                self.assertFalse((root / "raw").exists())

    def test_public_command_rejects_missing_lineage_or_traceability(self) -> None:
        cases = [
            ("generation_metadata", "generation_method"),
            ("generation_metadata", "base_report_id"),
            ("generation_metadata", "synthetic_report_id"),
            ("generation_metadata", "injection_scenario_id"),
            ("generation_metadata", "target_risk_category"),
            ("generation_metadata", "target_support_level"),
            ("split_metadata", "company_key"),
            ("split_metadata", "period_key"),
            ("split_metadata", "group_key"),
            ("split_metadata", "derived_from_group_key"),
            ("split_metadata", "source_file_sha256"),
            ("split_metadata", "normalized_text_hash"),
            ("split_metadata", "table_content_hash"),
            ("split_metadata", "derived_from_report_artifact_id"),
            ("split_metadata", "derived_from_source_document_id"),
        ]
        for metadata_section, field in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                staged_jsonl = root / "staged_packets.jsonl"
                record = _raw_record()
                _add_traceability(record)
                del record["metadata"][metadata_section][field]
                _write_jsonl(staged_jsonl, [record])

                completed = _run_stager(
                    "--input-jsonl",
                    str(staged_jsonl),
                    "--output-dir",
                    str(root / "raw"),
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(field, completed.stdout)
                self.assertFalse((root / "raw").exists())

    def test_public_command_rejects_empty_lineage_or_traceability(self) -> None:
        cases = [
            ("generation_metadata", "generation_method"),
            ("split_metadata", "normalized_text_hash"),
        ]
        for metadata_section, field in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                staged_jsonl = root / "staged_packets.jsonl"
                record = _raw_record()
                _add_traceability(record)
                record["metadata"][metadata_section][field] = ""
                _write_jsonl(staged_jsonl, [record])

                completed = _run_stager(
                    "--input-jsonl",
                    str(staged_jsonl),
                    "--output-dir",
                    str(root / "raw"),
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(field, completed.stdout)
                self.assertFalse((root / "raw").exists())

    def test_public_command_rejects_hidden_metadata_inside_detector_packet(self) -> None:
        cases = [
            ("hidden_injection", "hidden_injection_details", {"synthetic_value": 100}),
            ("omitted_evidence", "omitted_evidence_ids", ["NOTE_001"]),
            ("raw_ocr", "raw_ocr_text", "raw page text"),
            ("raw_pdf", "raw_pdf_path", "report.pdf"),
            ("cache", "cache_key", "cache-001"),
            ("source_hash", "source_file_sha256", "sha256-source"),
            ("traceability", "derived_from_report_artifact_id", "RPT_BASE"),
            ("final_report", "final_report_text", "final report"),
        ]
        for _, field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                staged_jsonl = root / "staged_packets.jsonl"
                record = _raw_record()
                _add_traceability(record)
                record["input"]["data"]["metadata"][field] = value
                _write_jsonl(staged_jsonl, [record])

                completed = _run_stager(
                    "--input-jsonl",
                    str(staged_jsonl),
                    "--output-dir",
                    str(root / "raw"),
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("DetectorPacket", completed.stdout)
                self.assertFalse((root / "raw").exists())

    def test_public_command_rejects_risk_category_inconsistencies(self) -> None:
        cases = [
            (
                "generation_metadata",
                lambda record: record["metadata"]["generation_metadata"].update({"target_risk_category": "asset_quality_risk"}),
            ),
            ("record_metadata", lambda record: record["metadata"].update({"risk_category": "asset_quality_risk"})),
            (
                "tool_finding",
                lambda record: record["input"]["data"]["tool_findings"][0].update({"risk_category": "asset_quality_risk"}),
            ),
            ("rule", lambda record: record["input"]["data"]["rules"][0].update({"risk_category": "asset_quality_risk"})),
        ]
        for boundary, mutate in cases:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                staged_jsonl = root / "staged_packets.jsonl"
                record = _raw_record()
                _add_traceability(record)
                mutate(record)
                _write_jsonl(staged_jsonl, [record])

                completed = _run_stager(
                    "--input-jsonl",
                    str(staged_jsonl),
                    "--output-dir",
                    str(root / "raw"),
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("risk_category", completed.stdout)
                self.assertFalse((root / "raw").exists())


def _run_stager(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.synthetic_raw_stager", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )


def _run_gate_dry_run(input_jsonl: Path, output_root: Path) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "research.synthetic_detector_assessment_gate",
            "--input-jsonl",
            str(input_jsonl),
            "--output-root",
            str(output_root),
            "--mode",
            "dry_run",
            "--run-id",
            "dry-run-check",
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )


def _add_traceability(record: dict) -> None:
    split_metadata = record["metadata"]["split_metadata"]
    split_metadata.update(
        {
            "company_key": "FPT",
            "period_key": "2024_Q3",
            "source_file_sha256": "sha256-source-FPT_2024_Q3_SYN_REV_001",
            "normalized_text_hash": "sha256-normalized-FPT_2024_Q3_SYN_REV_001",
            "table_content_hash": "sha256-table-FPT_2024_Q3_SYN_REV_001",
            "derived_from_report_artifact_id": "RPT_FPT_2024_Q3_SYN_REV_001",
            "derived_from_source_document_id": "DOC_FPT_2024_Q3_SYN_REV_001",
        }
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
