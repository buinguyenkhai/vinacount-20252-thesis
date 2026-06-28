import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from research.tests.test_detector_split_builder import _synthetic_record


LOCKED_SYSTEM_PROMPT = (
    "You are a specialized accounting irregularity risk-signal detector for Vietnamese financial reports.\n\n"
    "Your task is to assess whether the provided DetectorPacket supports the given CandidateRisk category.\n\n"
    "Return only a valid DetectorAssessment JSON object.\n\n"
    "Use only evidence provided in the packet.\n"
    "Reference only provided evidence IDs.\n"
    "Do not use outside knowledge.\n"
    "Do not use hidden injection metadata.\n"
    "Do not claim fraud, manipulation, intent, concealment, or legal misstatement.\n"
    "Use risk-signal language only.\n"
    "Keep rationale_short to 1–3 sentences."
)


class Wave4SftExporterPublicCommandTest(unittest.TestCase):
    def test_public_command_exports_valid_synthetic_train_record_as_chat_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_dir = root / "split-release"
            release_dir.mkdir()
            record = _valid_export_record(split="train")
            _write_split_release(release_dir, {"train": [record]})
            output_jsonl = root / "sft.jsonl"

            completed = _run_exporter(str(release_dir), "--output-jsonl", str(output_jsonl))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            exported = _read_jsonl(output_jsonl)
            self.assertEqual(len(exported), 1)
            chat_record = exported[0]
            self.assertEqual(
                [message["role"] for message in chat_record["messages"]],
                ["system", "user", "assistant"],
            )
            self.assertEqual(chat_record["messages"][0]["content"], LOCKED_SYSTEM_PROMPT)
            self.assertEqual(json.loads(chat_record["messages"][1]["content"]), record["input"]["data"])
            self.assertEqual(json.loads(chat_record["messages"][2]["content"]), record["output"]["data"])
            self.assertEqual(
                chat_record["metadata"],
                {
                    "example_id": "SFT_SYN_FPT_2024_Q3_001",
                    "dataset_version": "tdf_v1.0.0",
                    "source_type": "synthetic_injected_filtered",
                    "split": "train",
                    "risk_category": "revenue_income_recognition_risk",
                    "support_level": "supported",
                    "report_profile": "standard_corporate",
                    "report_id": "FPT_2024_Q3",
                    "company_key": "FPT",
                    "period_key": "2024_Q3",
                    "group_key": "FPT_2024_Q3",
                    "source_file_sha256": "sha256-source-FPT_2024_Q3",
                    "normalized_text_hash": "sha256-normalized-FPT_2024_Q3",
                    "table_content_hash": "sha256-table-FPT_2024_Q3",
                    "derived_from_report_artifact_id": "RPT_FPT_2024_Q3",
                    "derived_from_source_document_id": "DOC_FPT_2024_Q3",
                    "decontamination_group_id": "GROUP_FPT_2024_Q3",
                    "packet_content_hash": "packet-hash",
                    "assessment_rationale_hash": "rationale-hash",
                },
            )

    def test_public_command_skips_default_excluded_real_manual_validation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_dir = root / "split-release"
            release_dir.mkdir()
            synthetic_record = _valid_export_record(split="train")
            validation_record = _valid_export_record(
                split="validation",
                source_type="human_gold_real_report",
            )
            validation_record["metadata"]["split_metadata"]["usable_for_training"] = False
            _write_split_release(
                release_dir,
                {
                    "train": [synthetic_record],
                    "validation": [validation_record],
                },
            )
            output_jsonl = root / "sft.jsonl"

            completed = _run_exporter(str(release_dir), "--output-jsonl", str(output_jsonl))

            self.assertEqual(completed.returncode, 0, completed.stdout)
            exported = _read_jsonl(output_jsonl)
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]["metadata"]["source_type"], "synthetic_injected_filtered")

    def test_public_command_refuses_non_flattened_detector_assessment_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_dir = root / "split-release"
            release_dir.mkdir()
            record = _valid_export_record()
            record["output"]["data"] = {
                "type": "DetectorAssessment",
                "data": deepcopy(record["output"]["data"]),
            }
            _write_split_release(release_dir, {"train": [record]})

            completed = _run_exporter(str(release_dir), "--output-jsonl", str(root / "sft.jsonl"))

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("DetectorAssessment", completed.stdout)
            self.assertFalse((root / "sft.jsonl").exists())

    def test_public_command_keeps_traceability_metadata_outside_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_dir = root / "split-release"
            release_dir.mkdir()
            record = _valid_export_record()
            _write_split_release(release_dir, {"train": [record]})
            output_jsonl = root / "sft.jsonl"

            completed = _run_exporter(str(release_dir), "--output-jsonl", str(output_jsonl))

            self.assertEqual(completed.returncode, 0, completed.stderr)
            chat_record = _read_jsonl(output_jsonl)[0]
            message_text = json.dumps(chat_record["messages"], sort_keys=True)
            self.assertIn("source_file_sha256", chat_record["metadata"])
            self.assertIn("derived_from_report_artifact_id", chat_record["metadata"])
            self.assertNotIn("sha256-source-FPT_2024_Q3", message_text)
            self.assertNotIn("RPT_FPT_2024_Q3", message_text)
            self.assertEqual(json.loads(chat_record["messages"][1]["content"]), record["input"]["data"])

    def test_public_command_rejects_detector_visible_payload_leakage_in_messages(self) -> None:
        cases = [
            ("source_file_sha256", "sha256-source"),
            ("raw_ocr_text", "raw OCR page text"),
            ("hidden_injection_details", {"scenario": "hidden injection"}),
            ("omitted_evidence_ids", ["NOTE_99"]),
            ("final_report_text", "Final report narrative must stay outside messages."),
        ]
        for key, value in cases:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                release_dir = root / "split-release"
                release_dir.mkdir()
                record = _valid_export_record()
                record["input"]["data"]["metadata"][key] = value
                _write_split_release(release_dir, {"train": [record]})

                completed = _run_exporter(str(release_dir), "--output-jsonl", str(root / "sft.jsonl"))

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("DetectorPacket", completed.stdout)

    def test_public_command_refuses_default_excluded_real_manual_and_disallowed_rows(self) -> None:
        cases = [
            ("holdout", "synthetic_injected_filtered", "unsupported default SFT split"),
            ("train", "human_gold_real_report", "human_gold_real_report cannot be exported by default"),
            ("train", "manual_edge_case", "not allowed for default SFT export"),
        ]
        for split, source_type, expected_error in cases:
            with self.subTest(split=split, source_type=source_type), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                release_dir = root / "split-release"
                release_dir.mkdir()
                record = _valid_export_record(split=split, source_type=source_type)
                _write_split_release(release_dir, {split: [record]})
                if split == "holdout":
                    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
                    manifest["sft_export_allowed_splits"] = ["holdout"]
                    (release_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

                completed = _run_exporter(str(release_dir), "--output-jsonl", str(root / "sft.jsonl"))

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(expected_error, completed.stdout)

    def test_public_command_fails_clearly_when_manifest_is_not_sft_export_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_dir = root / "split-release"
            release_dir.mkdir()
            _write_split_release(release_dir, {"train": [_valid_export_record()]})
            manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["sft_export_ready"] = False
            (release_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            completed = _run_exporter(str(release_dir), "--output-jsonl", str(root / "sft.jsonl"))

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not SFT-export-ready", completed.stdout)

    def test_public_command_refuses_release_that_did_not_pass_split_validator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_dir = root / "split-release"
            release_dir.mkdir()
            _write_split_release(release_dir, {"train": [_valid_export_record()]})
            manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["integrity_checks"]["validator_status"] = "failed"
            (release_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            completed = _run_exporter(str(release_dir), "--output-jsonl", str(root / "sft.jsonl"))

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("validator_status must be passed", completed.stdout)
            self.assertFalse((root / "sft.jsonl").exists())

    def test_public_command_blocks_real_manual_rows_even_if_manifest_allows_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_dir = root / "split-release"
            release_dir.mkdir()
            record = _valid_export_record(source_type="human_gold_real_report")
            _write_split_release(release_dir, {"train": [record]})
            manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["sft_export_default_source_types"] = ["human_gold_real_report"]
            manifest["sft_export_excluded_source_types_by_default"] = []
            (release_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            completed = _run_exporter(str(release_dir), "--output-jsonl", str(root / "sft.jsonl"))

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("human_gold_real_report cannot be exported by default", completed.stdout)
            self.assertFalse((root / "sft.jsonl").exists())


def _run_exporter(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.sft_exporter", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )


def _valid_export_record(*, split: str = "train", source_type: str = "synthetic_injected_filtered") -> dict:
    record = _synthetic_record("FPT", "2024_Q3")
    record["source_type"] = source_type
    record["metadata"]["split_metadata"]["split"] = split
    record["metadata"]["split_metadata"]["usable_for_training"] = True
    record["metadata"]["split_metadata"]["exclusion_reason"] = None
    record["metadata"]["split_metadata"]["decontamination_group_id"] = "GROUP_FPT_2024_Q3"
    record["metadata"]["split_metadata"]["packet_content_hash"] = "packet-hash"
    record["metadata"]["split_metadata"]["assessment_rationale_hash"] = "rationale-hash"
    signal = record["output"]["data"]["validated_signals"][0]
    signal["status"] = "validated"
    signal["cited_evidence_refs"] = [
        {
            "evidence_ref_type": "table_cell",
            "ref_id": "FPT_2024_Q3:CELL_REVENUE_CURRENT",
            "role": "supporting",
        }
    ]
    return record


def _write_split_release(release_dir: Path, records_by_split: dict[str, list[dict]]) -> None:
    splits = ["train", "validation", "test", "holdout", "excluded"]
    records = [record for split in splits for record in records_by_split.get(split, [])]
    for split in splits:
        _write_jsonl(release_dir / f"{split}.jsonl", records_by_split.get(split, []))
    (release_dir / "leakage_report.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "dataset_version": "tdf_v1.0.0",
        "split_run_id": "test-release",
        "num_examples_total": len(records),
        "num_examples_by_split": {split: len(records_by_split.get(split, [])) for split in splits},
        "sft_export_ready": True,
        "sft_export_allowed_splits": ["train", "validation", "test"],
        "sft_export_default_source_types": [
            "synthetic_injected_filtered",
            "synthetic_injected_human_reviewed",
        ],
        "sft_export_excluded_source_types_by_default": ["human_gold_real_report"],
        "integrity_checks": {"validator_status": "passed"},
    }
    (release_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (release_dir / "metrics.json").write_text(json.dumps({"validator_status": "passed"}, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
