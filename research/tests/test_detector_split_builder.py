import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from research.tests.test_dataset_validator import _valid_record


class Wave4DetectorSplitBuilderPublicCommandTest(unittest.TestCase):
    def test_public_command_rejects_temperature_probe_gate_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "probe"
            gate_run_dir.mkdir(parents=True)
            _write_jsonl(gate_run_dir / "filtered.jsonl", [_synthetic_record("FPT", "2024_Q3")])
            _write_json(
                gate_run_dir / "manifest.json",
                {
                    "run_id": "probe",
                    "mode": "live",
                    "run_purpose": "temperature_probe",
                    "downstream_split_builder_allowed": False,
                    "trainable_labels_approved": False,
                },
            )
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 1})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "must-reject-probe",
                "--seed",
                "42",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not approved for downstream split building", completed.stderr)

    def test_public_command_writes_canonical_split_release_from_synthetic_gate_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            records = [_synthetic_record("FPT", "2024_Q3"), _synthetic_record("VHC", "2024_Q1"), _synthetic_record("SSI", "2025_Q3")]
            _write_jsonl(gate_run_dir / "filtered.jsonl", records)
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": len(records)})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "split-smoke",
                "--seed",
                "42",
                "--synthetic-train-ratio",
                "0.34",
                "--synthetic-validation-ratio",
                "0.33",
                "--synthetic-test-ratio",
                "0.33",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "split-smoke"
            self.assertEqual(
                sorted(path.name for path in release_dir.iterdir()),
                [
                    "excluded.jsonl",
                    "holdout.jsonl",
                    "leakage_report.jsonl",
                    "manifest.json",
                    "metrics.json",
                    "test.jsonl",
                    "train.jsonl",
                    "validation.jsonl",
                ],
            )
            manifest = _read_json(release_dir / "manifest.json")
            self.assertEqual(manifest["split_run_id"], "split-smoke")
            self.assertEqual(manifest["split_seed"], 42)
            self.assertEqual(manifest["integrity_checks"]["validator_status"], "passed")
            self.assertEqual(manifest["num_examples_total"], 3)
            self.assertEqual(set(manifest["num_examples_by_split"]), {"train", "validation", "test", "holdout", "excluded"})

            released_records = []
            for split in ["train", "validation", "test"]:
                for record in _read_jsonl(release_dir / f"{split}.jsonl"):
                    released_records.append(record)
                    split_metadata = record["metadata"]["split_metadata"]
                    self.assertEqual(split_metadata["split"], split)
                    self.assertEqual(split_metadata["split_strategy"], "decontamination_grouped_v1")
                    self.assertEqual(split_metadata["split_seed"], 42)
                    self.assertEqual(split_metadata["split_run_id"], "split-smoke")
                    self.assertTrue(split_metadata["decontamination_group_id"])
                    self.assertTrue(split_metadata["packet_content_hash"])
                    self.assertTrue(split_metadata["assessment_rationale_hash"])
                    self.assertTrue(split_metadata["usable_for_training"])
                    self.assertIsNone(split_metadata["exclusion_reason"])
            self.assertEqual(len(released_records), 3)

    def test_public_command_accepts_issue151_standard_legacy_gate_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "issue151-standard"
            gate_run_dir.mkdir(parents=True)
            records = [_synthetic_record("FPT", "2024_Q3")]
            _write_jsonl(gate_run_dir / "filtered.jsonl", records)
            _write_json(gate_run_dir / "manifest.json", _issue151_standard_legacy_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": len(records)})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "issue151-standard-split",
                "--seed",
                "151",
                "--synthetic-validation-ratio",
                "0",
                "--synthetic-test-ratio",
                "0",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "issue151-standard-split"
            self.assertEqual(len(_read_jsonl(release_dir / "train.jsonl")), 1)

    def test_public_command_rejects_issue151_standard_legacy_manifest_without_promotions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "issue151-empty-standard"
            gate_run_dir.mkdir(parents=True)
            _write_jsonl(gate_run_dir / "filtered.jsonl", [])
            manifest = _issue151_standard_legacy_gate_manifest()
            manifest["promoted_records"] = 0
            _write_json(gate_run_dir / "manifest.json", manifest)
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 0})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "must-reject-empty-issue151-standard",
                "--seed",
                "151",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("not approved for downstream split building", completed.stderr)

    def test_public_command_accepts_multiple_approved_synthetic_gate_dirs_without_merged_filtered_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_gate_dir = root / "gate" / "live-approved" / "live-gate-batch-001-v1"
            second_gate_dir = root / "gate" / "live-approved" / "live-gate-batch-002-v1"
            first_gate_dir.mkdir(parents=True)
            second_gate_dir.mkdir(parents=True)
            first_records = [_synthetic_record("FPT", "2024_Q3"), _synthetic_record("VHC", "2024_Q1")]
            second_records = [_synthetic_record("SSI", "2025_Q3"), _synthetic_record("VNM", "2024_Q4")]
            _write_jsonl(first_gate_dir / "filtered.jsonl", first_records)
            _write_jsonl(second_gate_dir / "filtered.jsonl", second_records)
            _write_json(first_gate_dir / "manifest.json", {**_approved_gate_manifest(), "run_id": "live-gate-batch-001-v1"})
            _write_json(second_gate_dir / "manifest.json", {**_approved_gate_manifest(), "run_id": "live-gate-batch-002-v1"})
            _write_json(first_gate_dir / "metrics.json", {"promoted_records": len(first_records)})
            _write_json(second_gate_dir / "metrics.json", {"promoted_records": len(second_records)})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(first_gate_dir),
                "--synthetic-gate-run-dir",
                str(second_gate_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "cumulative-split",
                "--seed",
                "151",
                "--synthetic-train-ratio",
                "0.5",
                "--synthetic-validation-ratio",
                "0.25",
                "--synthetic-test-ratio",
                "0.25",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "cumulative-split"
            manifest = _read_json(release_dir / "manifest.json")
            self.assertEqual(
                manifest["input_artifacts"]["synthetic_gate_run_dirs"],
                [str(first_gate_dir), str(second_gate_dir)],
            )
            self.assertEqual(
                [artifact["run_id"] for artifact in manifest["input_artifacts"]["synthetic_gate_runs"]],
                ["live-gate-batch-001-v1", "live-gate-batch-002-v1"],
            )
            for artifact in manifest["input_artifacts"]["synthetic_gate_runs"]:
                self.assertTrue(artifact["manifest_sha256"])
                self.assertTrue(artifact["metrics_sha256"])
                self.assertTrue(artifact["filtered_jsonl_sha256"])
            self.assertEqual(manifest["num_examples_total"], 4)
            self.assertEqual(manifest["num_examples_by_source_type"], {"synthetic_injected_filtered": 4})
            self.assertEqual(manifest["integrity_checks"]["validator_status"], "passed")
            self.assertFalse((root / "gate" / "merged" / "filtered.jsonl").exists())

            released_example_ids = {
                record["example_id"]
                for split in ["train", "validation", "test", "holdout", "excluded"]
                for record in _read_jsonl(release_dir / f"{split}.jsonl")
            }
            self.assertEqual(
                released_example_ids,
                {record["example_id"] for record in first_records + second_records},
            )

    def test_public_command_keeps_same_decontamination_group_in_one_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            first = _synthetic_record("FPT", "2024_Q3", sequence="001")
            second = _synthetic_record("FPT", "2024_Q3", sequence="002")
            second["example_id"] = "SFT_SYN_FPT_2024_Q3_002"
            second["input"]["data"]["packet_id"] = "PACKET_FPT_2024_Q3_002"
            second["output"]["data"]["packet_id"] = "PACKET_FPT_2024_Q3_002"
            second["output"]["data"]["assessment_id"] = "ASSESS_PACKET_FPT_2024_Q3_002"
            _write_jsonl(gate_run_dir / "filtered.jsonl", [first, second])
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 2})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "grouped",
                "--seed",
                "7",
                "--synthetic-train-ratio",
                "0.5",
                "--synthetic-validation-ratio",
                "0.5",
                "--synthetic-test-ratio",
                "0",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "grouped"
            records = [
                record
                for split in ["train", "validation", "test", "holdout", "excluded"]
                for record in _read_jsonl(release_dir / f"{split}.jsonl")
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                len({record["metadata"]["split_metadata"]["split"] for record in records}),
                1,
            )
            self.assertEqual(
                len({record["metadata"]["split_metadata"]["decontamination_group_id"] for record in records}),
                1,
            )

    def test_public_command_rejects_synthetic_group_that_collides_with_reserved_real_manual_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            synthetic = _synthetic_record("FPT", "2024_Q3")
            _write_jsonl(gate_run_dir / "filtered.jsonl", [synthetic])
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 1})

            real_release_dir = root / "real"
            real_release_dir.mkdir()
            real = _real_manual_record("FPT", "2024_Q3")
            _write_jsonl(real_release_dir / "validation.jsonl", [real])
            _write_json(real_release_dir / "manifest.json", {"num_examples_by_split": {"validation": 1}})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--real-manual-release-dir",
                str(real_release_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "reserved-collision",
                "--seed",
                "42",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exclude reserved real/manual anchors before synthetic generation", completed.stderr)

    def test_public_command_can_inspect_legacy_reserved_collision_artifacts_with_explicit_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            synthetic = _synthetic_record("FPT", "2024_Q3")
            _write_jsonl(gate_run_dir / "filtered.jsonl", [synthetic])
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 1})

            real_release_dir = root / "real"
            real_release_dir.mkdir()
            real = _real_manual_record("FPT", "2024_Q3")
            _write_jsonl(real_release_dir / "validation.jsonl", [real])
            _write_json(real_release_dir / "manifest.json", {"num_examples_by_split": {"validation": 1}})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--real-manual-release-dir",
                str(real_release_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "reserved-collision",
                "--seed",
                "42",
                "--allow-legacy-reserved-collision-exclusion",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "reserved-collision"
            validation_records = _read_jsonl(release_dir / "validation.jsonl")
            excluded_records = _read_jsonl(release_dir / "excluded.jsonl")
            self.assertEqual([record["source_type"] for record in validation_records], ["human_gold_real_report"])
            self.assertFalse(validation_records[0]["metadata"]["split_metadata"]["usable_for_training"])
            self.assertEqual([record["source_type"] for record in excluded_records], ["synthetic_injected_filtered"])
            self.assertFalse(excluded_records[0]["metadata"]["split_metadata"]["usable_for_training"])
            self.assertEqual(
                excluded_records[0]["metadata"]["split_metadata"]["exclusion_reason"],
                "collides_with_reserved_real_manual",
            )
            leakage_rows = _read_jsonl(release_dir / "leakage_report.jsonl")
            self.assertEqual(leakage_rows[0]["action"], "exclude_synthetic_group")

    def test_public_command_groups_exact_packet_and_rationale_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            first = _synthetic_record("FPT", "2024_Q3")
            duplicate = deepcopy(first)
            duplicate["example_id"] = "SFT_SYN_DUPLICATE_PACKET_001"
            duplicate["metadata"]["split_metadata"].update(
                {
                    "company_key": "VHC",
                    "period_key": "2024_Q1",
                    "group_key": "VHC_2024_Q1",
                    "source_file_sha256": "sha256-source-duplicate",
                    "normalized_text_hash": "sha256-normalized-duplicate",
                    "table_content_hash": "sha256-table-duplicate",
                    "derived_from_report_artifact_id": "RPT_DUPLICATE",
                    "derived_from_source_document_id": "DOC_DUPLICATE",
                    "derived_from_group_key": "VHC_2024_Q1_CLEAN",
                }
            )
            duplicate["metadata"]["generation_metadata"].update(
                {
                    "base_report_id": "VHC_2024_Q1_CLEAN",
                    "synthetic_report_id": "VHC_2024_Q1_SYN_001",
                    "injection_scenario_id": "INJ_VHC_2024_Q1_001",
                }
            )
            duplicate["metadata"]["audit_metadata"]["dataset_artifact_traceability"].update(
                {
                    "derived_from_report_artifact_id": "RPT_DUPLICATE",
                    "derived_from_source_document_id": "DOC_DUPLICATE",
                }
            )
            _write_jsonl(gate_run_dir / "filtered.jsonl", [first, duplicate])
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 2})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "duplicate-hashes",
                "--seed",
                "42",
                "--synthetic-train-ratio",
                "0.5",
                "--synthetic-validation-ratio",
                "0.5",
                "--synthetic-test-ratio",
                "0",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "duplicate-hashes"
            records = [
                record
                for split in ["train", "validation", "test", "holdout", "excluded"]
                for record in _read_jsonl(release_dir / f"{split}.jsonl")
            ]
            self.assertEqual(len({record["metadata"]["split_metadata"]["split"] for record in records}), 1)
            self.assertEqual(len({record["metadata"]["split_metadata"]["decontamination_group_id"] for record in records}), 1)
            self.assertEqual(len({record["metadata"]["split_metadata"]["packet_content_hash"] for record in records}), 1)
            self.assertEqual(len({record["metadata"]["split_metadata"]["assessment_rationale_hash"] for record in records}), 1)

    def test_public_command_groups_cross_filing_prior_year_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            target = _synthetic_record("FPT", "2024_Q3")
            target["input"]["data"]["relevant_table_rows"].append(
                {
                    "row_id": "ROW_PRIOR_REVENUE",
                    "report_id": "FPT_2023_Q3",
                    "values": {
                        "prior": {"period": "2023_Q3", "value": 100000, "cell_id": "CELL_PRIOR_REVENUE"}
                    },
                }
            )
            target["input"]["data"]["tool_findings"][0]["evidence_refs"].append(
                {
                    "evidence_ref_type": "table_cell",
                    "ref_id": "FPT_2023_Q3:CELL_PRIOR_REVENUE",
                    "report_id": "FPT_2023_Q3",
                    "role": "input",
                }
            )
            prior = _synthetic_record("FPT", "2023_Q3")
            prior["output"]["data"]["rationale_short"] = "Prior-year packet evidence supports a separate revenue risk signal."
            _write_jsonl(gate_run_dir / "filtered.jsonl", [target, prior])
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 2})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "cross-filing",
                "--seed",
                "99",
                "--synthetic-train-ratio",
                "0.5",
                "--synthetic-validation-ratio",
                "0.5",
                "--synthetic-test-ratio",
                "0",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "cross-filing"
            records = [
                record
                for split in ["train", "validation", "test", "holdout", "excluded"]
                for record in _read_jsonl(release_dir / f"{split}.jsonl")
            ]
            self.assertEqual(len({record["metadata"]["split_metadata"]["split"] for record in records}), 1)
            self.assertEqual(len({record["metadata"]["split_metadata"]["decontamination_group_id"] for record in records}), 1)

    def test_public_command_does_not_group_unrelated_reports_by_shared_scenario_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            first = _synthetic_record("FPT", "2024_Q3")
            second = _synthetic_record("VHC", "2024_Q1")
            first["metadata"]["generation_metadata"]["injection_scenario_id"] = "shared_revenue_scenario"
            second["metadata"]["generation_metadata"]["injection_scenario_id"] = "shared_revenue_scenario"
            first["output"]["data"]["rationale_short"] = "FPT packet evidence supports the scoped revenue risk signal."
            second["output"]["data"]["rationale_short"] = "VHC packet evidence supports the scoped revenue risk signal."
            _write_jsonl(gate_run_dir / "filtered.jsonl", [first, second])
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 2})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "shared-scenario",
                "--seed",
                "151",
                "--synthetic-train-ratio",
                "0.5",
                "--synthetic-validation-ratio",
                "0.5",
                "--synthetic-test-ratio",
                "0",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            release_dir = root / "splits" / "shared-scenario"
            records = [
                record
                for split in ["train", "validation", "test", "holdout", "excluded"]
                for record in _read_jsonl(release_dir / f"{split}.jsonl")
            ]
            self.assertEqual(len({record["metadata"]["split_metadata"]["decontamination_group_id"] for record in records}), 2)
            self.assertEqual({record["metadata"]["split_metadata"]["split"] for record in records}, {"train", "validation"})

    def test_public_command_hard_fails_synthetic_records_missing_lineage_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gate_run_dir = root / "gate" / "run-001"
            gate_run_dir.mkdir(parents=True)
            record = _synthetic_record("FPT", "2024_Q3")
            del record["metadata"]["generation_metadata"]["base_report_id"]
            _write_jsonl(gate_run_dir / "filtered.jsonl", [record])
            _write_json(gate_run_dir / "manifest.json", _approved_gate_manifest())
            _write_json(gate_run_dir / "metrics.json", {"promoted_records": 1})

            completed = _run_split_builder(
                "--synthetic-gate-run-dir",
                str(gate_run_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "missing-lineage",
                "--seed",
                "42",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing required synthetic lineage metadata", completed.stderr)

    def test_public_command_hard_fails_real_manual_records_assigned_to_train(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            real_release_dir = root / "real"
            real_release_dir.mkdir()
            real = _real_manual_record("FPT", "2024_Q3")
            real["metadata"]["split_metadata"]["split"] = "train"
            _write_jsonl(real_release_dir / "train.jsonl", [real])
            _write_json(real_release_dir / "manifest.json", {"num_examples_by_split": {"train": 1}})

            completed = _run_split_builder(
                "--real-manual-release-dir",
                str(real_release_dir),
                "--output-root",
                str(root / "splits"),
                "--run-id",
                "real-train",
                "--seed",
                "42",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("human_gold_real_report records cannot be assigned to train", completed.stderr)


def _run_split_builder(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.detector_split_builder", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )


def _synthetic_record(company_key: str, period_key: str, *, sequence: str = "001") -> dict:
    record = deepcopy(_valid_record())
    report_id = f"{company_key}_{period_key}"
    old_report_id = "FPT_2024_Q3"
    serialized = json.dumps(record, ensure_ascii=False)
    record = json.loads(serialized.replace(old_report_id, report_id).replace("FPT", company_key))
    record["example_id"] = f"SFT_SYN_{report_id}_{sequence}"
    record["metadata"]["generation_metadata"] = {
        "generation_method": "clean_report_risk_injection_best_of_n",
        "base_report_id": f"{report_id}_CLEAN",
        "synthetic_report_id": f"{report_id}_SYN_001",
        "injection_scenario_id": f"INJ_{report_id}_001",
    }
    split_metadata = record["metadata"]["split_metadata"]
    split_metadata.update(
        {
            "company_key": company_key,
            "period_key": period_key,
            "group_key": report_id,
            "source_file_sha256": f"sha256-source-{report_id}",
            "normalized_text_hash": f"sha256-normalized-{report_id}",
            "table_content_hash": f"sha256-table-{report_id}",
            "derived_from_report_artifact_id": f"RPT_{report_id}",
            "derived_from_source_document_id": f"DOC_{report_id}",
            "derived_from_group_key": f"{report_id}_CLEAN",
        }
    )
    record["metadata"]["audit_metadata"]["dataset_artifact_traceability"].update(
        {
            "derived_from_report_artifact_id": f"RPT_{report_id}",
            "derived_from_source_document_id": f"DOC_{report_id}",
        }
    )
    return record


def _real_manual_record(company_key: str, period_key: str) -> dict:
    record = _synthetic_record(company_key, period_key)
    record["source_type"] = "human_gold_real_report"
    record["example_id"] = f"GOLD_REAL_{company_key}_{period_key}_001"
    record["metadata"]["generation_metadata"] = {
        "builder_version": "test_real_manual_release",
        "generation_method": "human_gold_real_report_materialization",
    }
    record["metadata"]["split_metadata"]["split"] = "validation"
    record["metadata"]["split_metadata"]["usable_for_training"] = False
    record["metadata"]["split_metadata"]["exclusion_reason"] = None
    return record


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def _approved_gate_manifest() -> dict:
    return {
        "run_id": "run-001",
        "artifact_contract_version": "synthetic_detector_assessment_gate_v1",
        "mode": "live",
        "run_purpose": "approved_live",
        "downstream_split_builder_allowed": True,
        "trainable_labels_approved": True,
    }


def _issue151_standard_legacy_gate_manifest() -> dict:
    return {
        "run_id": "issue151-standard",
        "artifact_contract_version": "synthetic_detector_assessment_gate_v1",
        "mode": "live",
        "run_purpose": "approved_issue151_standard",
        "downstream_split_builder_allowed": False,
        "trainable_labels_approved": False,
        "input_records": 80,
        "approved_live_input_records_required": 80,
        "promoted_records": 1,
        "judge_replay_candidates_jsonl_path": None,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
