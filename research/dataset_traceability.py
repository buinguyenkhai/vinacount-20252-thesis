from __future__ import annotations

from copy import deepcopy
from typing import Any


TRACEABILITY_FIELDS = (
    "company_key",
    "period_key",
    "group_key",
    "source_file_sha256",
    "normalized_text_hash",
    "table_content_hash",
    "derived_from_report_artifact_id",
    "derived_from_source_document_id",
)

SPLIT_TRACEABILITY_FIELDS = TRACEABILITY_FIELDS

AUDIT_TRACEABILITY_FIELDS = (
    "derived_from_report_artifact_id",
    "derived_from_source_document_id",
)


def apply_report_artifact_traceability(
    dataset_record: dict[str, Any],
    report_artifact_metadata: dict[str, Any],
) -> dict[str, Any]:
    record = deepcopy(dataset_record)
    metadata = record.setdefault("metadata", {})
    split_metadata = metadata.setdefault("split_metadata", {})
    audit_metadata = metadata.setdefault("audit_metadata", {})

    for field in SPLIT_TRACEABILITY_FIELDS:
        value = report_artifact_metadata.get(field)
        if value is not None:
            split_metadata[field] = value

    traceability = audit_metadata.setdefault("dataset_artifact_traceability", {})
    for field in AUDIT_TRACEABILITY_FIELDS:
        value = report_artifact_metadata.get(field)
        if value is not None:
            traceability[field] = value

    return record


def build_source_report_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        split_metadata = record.get("metadata", {}).get("split_metadata", {})
        report_artifact_id = split_metadata.get("derived_from_report_artifact_id")
        source_document_id = split_metadata.get("derived_from_source_document_id")
        if not report_artifact_id or not source_document_id:
            continue
        key = (report_artifact_id, source_document_id)
        if key not in grouped:
            grouped[key] = {
                "derived_from_report_artifact_id": report_artifact_id,
                "derived_from_source_document_id": source_document_id,
                "company_key": split_metadata.get("company_key"),
                "period_key": split_metadata.get("period_key"),
                "group_key": split_metadata.get("group_key"),
                "source_file_sha256": split_metadata.get("source_file_sha256"),
                "normalized_text_hash": split_metadata.get("normalized_text_hash"),
                "table_content_hash": split_metadata.get("table_content_hash"),
                "example_count": 0,
                "splits": [],
            }
        grouped[key]["example_count"] += 1
        split = split_metadata.get("split")
        if split and split not in grouped[key]["splits"]:
            grouped[key]["splits"].append(split)

    summaries = sorted(
        grouped.values(),
        key=lambda item: (
            item["derived_from_report_artifact_id"],
            item["derived_from_source_document_id"],
        ),
    )
    for summary in summaries:
        summary["splits"] = sorted(summary["splits"])
    return {"report_artifacts": summaries}
