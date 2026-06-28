from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.fixture_spine import validate_detector_assessment, validate_detector_packet
from research.dataset_traceability import TRACEABILITY_FIELDS, build_source_report_manifest


@dataclass(frozen=True)
class Wave4DatasetValidationResult:
    status: str
    records_validated: int
    errors: list[str]


def validate_detector_dataset_release(dataset_dir: Path | str) -> Wave4DatasetValidationResult:
    root = Path(dataset_dir)
    errors: list[str] = []
    records: list[dict[str, Any]] = []

    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Wave4DatasetValidationResult("failed", 0, ["manifest.json is required"])
    except json.JSONDecodeError as error:
        return Wave4DatasetValidationResult("failed", 0, [f"manifest.json is malformed JSON: {error.msg}"])

    for split, expected_count in manifest.get("num_examples_by_split", {}).items():
        split_path = root / f"{split}.jsonl"
        if not split_path.exists():
            errors.append(f"{split}.jsonl is required by manifest")
            continue
        for line_number, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{split}.jsonl:{line_number} is malformed JSONL: {error.msg}")
                continue
            try:
                _validate_record(record, expected_split=split)
            except ValueError as error:
                errors.append(f"{split}.jsonl:{line_number}: {error}")
                continue
            records.append(record)

        actual_split_count = sum(
            1 for record in records if record["metadata"]["split_metadata"]["split"] == split
        )
        if actual_split_count != expected_count:
            errors.append(
                f"manifest num_examples_by_split[{split}] expected {expected_count} but found {actual_split_count}"
            )

    errors.extend(_manifest_count_errors(manifest, records))
    errors.extend(_manifest_traceability_errors(manifest, records))
    errors.extend(_split_decontamination_errors(records))
    return Wave4DatasetValidationResult(
        status="failed" if errors else "passed",
        records_validated=len(records),
        errors=errors,
    )


def _validate_record(record: dict[str, Any], *, expected_split: str) -> None:
    required_fields = {"example_id", "dataset_version", "source_type", "input", "output", "metadata"}
    missing = required_fields - record.keys()
    if missing:
        raise ValueError(f"dataset record is missing fields: {sorted(missing)}")
    if record["input"].get("type") != "DetectorPacket":
        raise ValueError("input wrapper type must be DetectorPacket")
    if record["output"].get("type") != "DetectorAssessment":
        raise ValueError("output wrapper type must be DetectorAssessment")
    packet = record["input"].get("data")
    assessment = record["output"].get("data")
    if not isinstance(packet, dict):
        raise ValueError("input.data must be a DetectorPacket object")
    if not isinstance(assessment, dict):
        raise ValueError("output.data must be a DetectorAssessment object")
    if _contains_detector_visible_leakage(packet) or _contains_detector_visible_leakage(assessment):
        raise ValueError("detector-visible data contains prohibited hidden or raw metadata")

    validate_detector_packet(packet)
    validate_detector_assessment(assessment, packet)
    _validate_assessment_constraints(assessment)

    metadata = record["metadata"]
    for field in [
        "risk_category",
        "support_level",
        "severity",
        "report_profile",
        "report_period_type",
        "language",
        "evidence_profile",
        "generation_metadata",
        "validation_metadata",
        "split_metadata",
        "audit_metadata",
    ]:
        if field not in metadata:
            raise ValueError(f"metadata is missing {field}")
    if metadata["split_metadata"].get("split") != expected_split:
        raise ValueError("metadata split must match the JSONL split file")
    if metadata["risk_category"] != assessment["risk_category"]:
        raise ValueError("metadata risk_category must match DetectorAssessment")
    if metadata["support_level"] != assessment["support_level"]:
        raise ValueError("metadata support_level must match DetectorAssessment")
    if metadata["severity"] != assessment["severity"]:
        raise ValueError("metadata severity must match DetectorAssessment")
    _validate_traceability_metadata(metadata)


def _validate_traceability_metadata(metadata: dict[str, Any]) -> None:
    split_metadata = metadata["split_metadata"]
    for field in TRACEABILITY_FIELDS:
        if not split_metadata.get(field):
            raise ValueError(f"split_metadata is missing required traceability field {field}")
    audit_traceability = metadata["audit_metadata"].get("dataset_artifact_traceability")
    if not isinstance(audit_traceability, dict):
        raise ValueError("audit_metadata is missing dataset_artifact_traceability")
    for field in ["derived_from_report_artifact_id", "derived_from_source_document_id"]:
        if audit_traceability.get(field) != split_metadata.get(field):
            raise ValueError(f"audit_metadata dataset_artifact_traceability must match split_metadata {field}")


def _validate_assessment_constraints(assessment: dict[str, Any]) -> None:
    confidence = assessment["confidence"]
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("DetectorAssessment confidence must be a number between 0 and 1")
    if confidence < 0 or confidence > 1:
        raise ValueError("DetectorAssessment confidence must be between 0 and 1")


def _contains_detector_visible_leakage(value: Any) -> bool:
    prohibited_keys = {
        "hidden_injection_details",
        "injection_scenario",
        "injection_script_version",
        "sampling_pool",
        "correction_provenance",
        "correction_amendment_provenance_type",
        "ordinary_provenance",
        "ordinary_filing_provenance",
        "ordinary_provenance_rationale",
        "zip_container_sha256",
        "selected_member_sha256",
        "selected_member_path",
        "target_selected_member_sha256",
        "prior_year_selected_member_sha256",
        "target_selected_member_path",
        "prior_year_selected_member_path",
        "cache_record_id",
        "cache_key",
        "report_artifact_cache",
        "source_file_hash",
        "source_file_sha256",
        "normalized_text_hash",
        "table_content_hash",
        "raw_ocr_text",
        "full_raw_ocr_text",
        "raw_tables",
        "raw_pdf_coordinates",
        "raw_coordinates",
        "coordinates",
        "bbox",
        "bounding_box",
        "external_context",
        "outside_context",
        "outside_knowledge",
        "omitted_evidence_ids",
        "omitted_evidence_summaries",
        "omitted_evidence_count",
        "long_reasoning",
        "hidden_reasoning",
        "chain_of_thought",
        "hidden_chain_of_thought",
    }
    if isinstance(value, dict):
        return any(
            key in prohibited_keys or _contains_detector_visible_leakage(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_detector_visible_leakage(item) for item in value)
    return False


def _manifest_count_errors(manifest: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    checks = [
        ("num_examples_total", None, Counter({"total": len(records)}), "total"),
        ("num_examples_by_source_type", "source_type", Counter(record["source_type"] for record in records), None),
        (
            "num_examples_by_support_level",
            "metadata.support_level",
            Counter(record["metadata"]["support_level"] for record in records),
            None,
        ),
        (
            "num_examples_by_report_profile",
            "metadata.report_profile",
            Counter(record["metadata"]["report_profile"] for record in records),
            None,
        ),
        (
            "num_examples_by_risk_category",
            "metadata.risk_category",
            Counter(record["metadata"]["risk_category"] for record in records),
            None,
        ),
    ]
    errors: list[str] = []
    for manifest_field, _, actual, single_key in checks:
        expected = manifest.get(manifest_field)
        if single_key:
            if expected != actual[single_key]:
                errors.append(f"manifest {manifest_field} expected {expected} but found {actual[single_key]}")
            continue
        if expected != dict(actual):
            errors.append(f"manifest {manifest_field} does not match actual records")
    return errors


def _manifest_traceability_errors(manifest: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    expected = build_source_report_manifest(records)
    actual = manifest.get("source_report_traceability")
    if actual is None:
        return ["manifest source_report_traceability is required"]
    if actual != expected:
        return ["manifest source_report_traceability does not match actual records"]
    return []


def _split_decontamination_errors(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    fields = [
        "group_key",
        "source_file_sha256",
        "normalized_text_hash",
        "table_content_hash",
        "derived_from_report_artifact_id",
        "derived_from_source_document_id",
    ]
    for field in fields:
        splits_by_value: dict[str, set[str]] = {}
        for record in records:
            split_metadata = record["metadata"]["split_metadata"]
            value = split_metadata.get(field)
            split = split_metadata.get("split")
            if split == "excluded":
                continue
            if not value or not split:
                continue
            splits_by_value.setdefault(value, set()).add(split)
        for value, splits in sorted(splits_by_value.items()):
            if len(splits) > 1:
                errors.append(
                    f"split decontamination violation: {field} {value} appears in splits {sorted(splits)}"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Wave 4 detector dataset release directory.")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()

    result = validate_detector_dataset_release(args.dataset_dir)
    print(
        json.dumps(
            {
                "status": result.status,
                "records_validated": result.records_validated,
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result.status == "passed" else 1)


if __name__ == "__main__":
    main()

