from __future__ import annotations

from typing import Any

from research.detector_contract_validation import (
    contains_prohibited_detector_visible_payload,
    validate_detector_packet,
)


REQUIRED_GENERATION_FIELDS = (
    "generation_method",
    "base_report_id",
    "synthetic_report_id",
    "injection_scenario_id",
    "target_risk_category",
    "target_support_level",
)
REQUIRED_SPLIT_FIELDS = (
    "company_key",
    "period_key",
    "group_key",
    "derived_from_group_key",
    "source_file_sha256",
    "normalized_text_hash",
    "table_content_hash",
    "derived_from_report_artifact_id",
    "derived_from_source_document_id",
)
PROHIBITED_RAW_CANDIDATE_FIELDS = {
    "DetectorAssessment",
    "assessment",
    "detector_assessment",
    "label",
    "labels",
    "target",
}


def validate_synthetic_raw_candidate(record: Any) -> None:
    if not isinstance(record, dict):
        raise ValueError("staged record must be a JSON object")
    if record.get("source_type") != "synthetic_injected_raw":
        raise ValueError("staged records must have source_type synthetic_injected_raw")
    if "output" in record:
        raise ValueError("synthetic_injected_raw staged records must not contain output")
    label_like_fields = sorted(_find_prohibited_label_like_fields(record))
    if label_like_fields:
        raise ValueError(f"synthetic_injected_raw staged records must not contain label-like target fields: {label_like_fields}")
    input_wrapper = record.get("input")
    if not isinstance(input_wrapper, dict) or input_wrapper.get("type") != "DetectorPacket":
        raise ValueError("input wrapper type must be DetectorPacket")
    packet = input_wrapper.get("data")
    if not isinstance(packet, dict):
        raise ValueError("input.data must be a DetectorPacket object")
    if contains_prohibited_detector_visible_payload(packet):
        raise ValueError("DetectorPacket contains prohibited hidden, raw, traceability, or final-report metadata")
    try:
        validate_detector_packet(packet)
    except ValueError as error:
        raise ValueError(f"DetectorPacket is invalid: {error}") from error

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata is required")
    generation_metadata = metadata.get("generation_metadata")
    if not isinstance(generation_metadata, dict):
        raise ValueError("metadata.generation_metadata is required")
    split_metadata = metadata.get("split_metadata")
    if not isinstance(split_metadata, dict):
        raise ValueError("metadata.split_metadata is required")

    _require_fields(generation_metadata, REQUIRED_GENERATION_FIELDS, "metadata.generation_metadata")
    _require_fields(split_metadata, REQUIRED_SPLIT_FIELDS, "metadata.split_metadata")
    risk_category = packet["task"]["risk_category"]
    if generation_metadata["target_risk_category"] != risk_category:
        raise ValueError("generation_metadata target_risk_category must match DetectorPacket task risk_category")
    if metadata.get("risk_category") != risk_category:
        raise ValueError("metadata risk_category must match DetectorPacket task risk_category")
    if not isinstance(packet["tool_findings"], list) or not all(isinstance(finding, dict) for finding in packet["tool_findings"]):
        raise ValueError("DetectorPacket tool_findings must contain ToolFinding objects")
    if not isinstance(packet["rules"], list) or not all(isinstance(rule, dict) for rule in packet["rules"]):
        raise ValueError("DetectorPacket rules must contain rule context objects")
    if any(finding.get("risk_category") != risk_category for finding in packet["tool_findings"]):
        raise ValueError("DetectorPacket ToolFinding risk_category must match DetectorPacket task risk_category")
    if any(rule.get("risk_category") != risk_category for rule in packet["rules"]):
        raise ValueError("DetectorPacket rule risk_category must match DetectorPacket task risk_category")
    if split_metadata["group_key"] == split_metadata["derived_from_group_key"]:
        raise ValueError("split_metadata derived_from_group_key must identify the clean/base group")


def _require_fields(data: dict[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if not data.get(field)]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _find_prohibited_label_like_fields(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = PROHIBITED_RAW_CANDIDATE_FIELDS & value.keys()
        for child in value.values():
            found |= _find_prohibited_label_like_fields(child)
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for item in value:
            found |= _find_prohibited_label_like_fields(item)
        return found
    return set()
