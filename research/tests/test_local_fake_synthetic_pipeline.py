import json
import tempfile
import unittest
from pathlib import Path

from research.tests.test_grounded_synthetic_packet_generator import (
    _clean_structured_report,
    _run_module,
    _write_json,
)


class Wave4LocalFakeSyntheticPipelinePublicCommandTest(unittest.TestCase):
    def test_public_commands_export_trainable_synthetic_chat_without_detector_visible_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            generated_dir = root / "generated"
            raw_dir = root / "raw"
            gate_root = root / "gate"
            split_root = root / "splits"
            export_path = root / "detector_sft_chat.jsonl"
            _write_json(clean_artifact, _clean_structured_report())

            generated = _run_module(
                "research.grounded_synthetic_packet_generator",
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(generated_dir),
                "--run-id",
                "local-fake-generator",
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)

            staged = _run_module(
                "research.synthetic_raw_stager",
                "--input-jsonl",
                str(generated_dir / "staged_synthetic_packets.jsonl"),
                "--output-dir",
                str(raw_dir),
                "--run-id",
                "local-fake-raw",
            )
            self.assertEqual(staged.returncode, 0, staged.stdout + staged.stderr)

            gate = _run_module(
                "research.synthetic_detector_assessment_gate",
                "--input-jsonl",
                str(raw_dir / "synthetic_injected_raw.jsonl"),
                "--output-root",
                str(gate_root),
                "--run-id",
                "local-fake-gate",
                "--mode",
                "fake",
                "--teacher-model",
                "fake-teacher",
                "--judge-model",
                "fake-judge",
            )
            self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)

            gate_run_dir = gate_root / "local-fake-gate"
            split = _run_module(
                "research.detector_split_builder",
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(split_root),
                "--run-id",
                "local-fake-split",
                "--seed",
                "42",
                "--synthetic-train-ratio",
                "1",
                "--synthetic-validation-ratio",
                "0",
                "--synthetic-test-ratio",
                "0",
                "--synthetic-holdout-ratio",
                "0",
                "--allow-fake-smoke",
            )
            self.assertEqual(split.returncode, 0, split.stdout + split.stderr)

            split_release_dir = split_root / "local-fake-split"
            exported = _run_module(
                "research.sft_exporter",
                str(split_release_dir),
                "--output-jsonl",
                str(export_path),
            )
            self.assertEqual(exported.returncode, 0, exported.stdout + exported.stderr)

            split_manifest = _read_json(split_release_dir / "manifest.json")
            self.assertEqual(split_manifest["integrity_checks"]["validator_status"], "passed")
            train_rows = _read_jsonl(split_release_dir / "train.jsonl")
            self.assertGreater(len(train_rows), 0)
            self.assertTrue(all(row["source_type"] == "synthetic_injected_filtered" for row in train_rows))
            self.assertFalse(
                any(
                    row["source_type"] in {"human_gold_real_report", "manual_edge_case"}
                    for row in train_rows
                )
            )

            chat_rows = _read_jsonl(export_path)
            self.assertEqual(len(chat_rows), len(train_rows))
            chat_row = chat_rows[0]
            self.assertEqual(
                [message["role"] for message in chat_row["messages"]],
                ["system", "user", "assistant"],
            )
            self.assertIn("metadata", chat_row)

            message_text = json.dumps(chat_row["messages"], ensure_ascii=False, sort_keys=True)
            for prohibited_text in [
                "generation_metadata",
                "split_metadata",
                "target_support_level",
                "source_file_sha256",
                "normalized_text_hash",
                "table_content_hash",
                "raw_ocr",
                "raw_pdf",
                "cache",
                "omitted_evidence",
                "final_report",
                "audit_traceability",
                "sha256-source-FPT_2024_Q3",
                "sha256-normalized-FPT_2024_Q3",
                "sha256-table-FPT_2024_Q3",
            ]:
                self.assertNotIn(prohibited_text, message_text)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
