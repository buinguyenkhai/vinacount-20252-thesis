from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research.fixture_spine import validate_detector_packet
from vinacount.report_model import CompanyReportSet, validate_real_extraction_quality_gate


@dataclass(frozen=True)
class Wave4DatasetEligibilityResult:
    status: str
    records: list[dict[str, Any]]


def validate_real_extracted_packet_dataset_eligibility(
    report_set: CompanyReportSet,
    detector_packets: list[dict[str, Any]],
) -> Wave4DatasetEligibilityResult:
    records: list[dict[str, Any]] = []
    source_reason_codes = _source_and_report_set_reason_codes(report_set)

    for packet in detector_packets:
        packet_reason_codes = [*source_reason_codes, *_packet_reason_codes(report_set, packet)]
        if packet_reason_codes:
            for reason_code, details in packet_reason_codes:
                records.append(_record(report_set, packet, "dataset_ineligible", reason_code, details))
        else:
            records.append(
                _record(
                    report_set,
                    packet,
                    "dataset_eligible",
                    "dataset_eligible",
                    "Real extracted DetectorPacket passed Wave 4 dataset eligibility gates.",
                )
            )

    return Wave4DatasetEligibilityResult(
        status="failed" if any(record["record_type"] == "dataset_ineligible" for record in records) else "passed",
        records=records,
    )


def _source_and_report_set_reason_codes(report_set: CompanyReportSet) -> list[tuple[str, str]]:
    reason_codes: list[tuple[str, str]] = []
    gate_result = validate_real_extraction_quality_gate(report_set)
    for record in gate_result.records:
        reason_codes.append((record["reason_code"], record["reason"]))

    for role, report_memory in (("target", report_set.target), ("prior_year", report_set.prior_year)):
        metadata = report_memory.metadata
        canonical_id = metadata.get("canonical_source_document_id")
        if not canonical_id:
            reason_codes.append(("canonical_source_not_selected", f"{role} canonical source is not selected."))

        confirmation_status = metadata.get("source_confirmation_status", "confirmed")
        certainty = metadata.get("canonical_source_certainty", "resolved")
        if confirmation_status not in {"confirmed", "source_confirmed", "selected"}:
            reason_codes.append(("source_confirmation_required", f"{role} source confirmation status is {confirmation_status}."))
        if certainty != "resolved":
            reason_codes.append(("canonical_source_certainty_unresolved", f"{role} canonical source certainty is {certainty}."))

        source_quality_status = metadata.get("source_quality_status", "passed")
        if source_quality_status == "recoverable_source_quality_error":
            reason_codes.append(("recoverable_source_quality_error", f"{role} has recoverable source-quality errors."))
        if source_quality_status in {"hitl_needed", "source_quality_hitl_needed"}:
            reason_codes.append(("source_confirmation_required", f"{role} source quality requires HITL."))

        language = metadata.get("language", "vi")
        source_details = metadata.get("source_quality_details", {})
        if language == "en" and source_details.get("vietnamese_candidate_available"):
            reason_codes.append(("english_duplicate_excluded", f"{role} is an English duplicate while a Vietnamese candidate exists."))
        if source_details.get("source_role") == "english_duplicate" and source_details.get("vietnamese_candidate_available"):
            reason_codes.append(("english_duplicate_excluded", f"{role} is an English duplicate while a Vietnamese candidate exists."))
        if source_details.get("source_role") == "searchable_filing_version" and source_details.get("identity_check_status") != "confirmed":
            reason_codes.append(("unresolved_tra_cuu_identity", f"{role} searchable filing identity is not confirmed."))
        if source_details.get("reviewed_supersession_status") == "unresolved":
            reason_codes.append(("unresolved_reviewed_supersession", f"{role} reviewed/original supersession is unresolved."))
        if source_details.get("full_report_amendment_supersession_status") == "unresolved":
            reason_codes.append(("unresolved_full_report_amendment_supersession", f"{role} full-report amendment supersession is unresolved."))

    return _dedupe_reason_codes(reason_codes)


def _packet_reason_codes(report_set: CompanyReportSet, packet: dict[str, Any]) -> list[tuple[str, str]]:
    reason_codes: list[tuple[str, str]] = []
    try:
        validate_detector_packet(packet)
    except ValueError as error:
        if "raw extraction payload" in str(error):
            reason_codes.append(("prohibited_metadata_leakage", str(error)))
        else:
            reason_codes.append(("invalid_detector_packet", str(error)))

    if _contains_raw_or_operational_leakage(packet):
        reason_codes.append(("prohibited_metadata_leakage", "DetectorPacket contains prohibited source, sampling, or operational metadata."))

    invalid_evidence = _invalid_packet_evidence_ids(report_set, packet)
    if invalid_evidence:
        reason_codes.append(("invalid_evidence_ids", invalid_evidence))

    return _dedupe_reason_codes(reason_codes)


def _invalid_packet_evidence_ids(report_set: CompanyReportSet, packet: dict[str, Any]) -> str | None:
    cell_ids = set(report_set.target.raw.get("cell_index", {})) | set(report_set.prior_year.raw.get("cell_index", {}))
    note_ids = {note.get("note_id") for note in report_set.target.raw.get("notes", [])}
    note_ids |= {note.get("note_id") for note in report_set.prior_year.raw.get("notes", [])}
    variance_ids = {span.get("span_id") for span in report_set.target.raw.get("variance_explanations", [])}
    variance_ids |= {span.get("span_id") for span in report_set.prior_year.raw.get("variance_explanations", [])}

    missing: list[str] = []
    for row in packet.get("relevant_table_rows", []):
        for value in row.get("values", {}).values():
            cell_id = value.get("cell_id")
            if cell_id and cell_id not in cell_ids:
                missing.append(cell_id)
    for note in packet.get("relevant_notes", []):
        note_id = note.get("note_id")
        if note_id and note_id not in note_ids:
            missing.append(note_id)
    for span in packet.get("relevant_variance_explanations", []):
        span_id = span.get("span_id")
        if span_id and span_id not in variance_ids:
            missing.append(span_id)
    for finding in packet.get("tool_findings", []):
        for cell_id in finding.get("evidence_cell_ids", []):
            if cell_id not in cell_ids:
                missing.append(cell_id)
        for note_id in finding.get("evidence_note_ids", []):
            if note_id not in note_ids:
                missing.append(note_id)

    if missing:
        return f"DetectorPacket references evidence IDs not present in target/prior registries: {', '.join(sorted(set(missing)))}."
    return None


def _contains_raw_or_operational_leakage(value: Any) -> bool:
    prohibited_keys = {
        "sampling_pool",
        "correction_provenance",
        "correction_amendment_provenance_type",
        "ordinary_provenance",
        "ordinary_filing_provenance",
        "ordinary_provenance_rationale",
        "zip_container_sha256",
        "selected_member_sha256",
        "target_selected_member_sha256",
        "prior_year_selected_member_sha256",
        "selected_member_path",
        "target_selected_member_path",
        "prior_year_selected_member_path",
        "raw_coordinates",
        "coordinates",
        "bbox",
        "bounding_box",
        "raw_ocr_text",
        "full_raw_ocr_text",
        "source_acquisition_notes",
        "local_path",
        "source_file_hash",
        "source_file_sha256",
        "zip_member_path",
        "fetch_details",
        "provider_run_id",
        "external_context",
        "hidden_reasoning",
        "cache_record_id",
        "cache_key",
        "omitted_evidence_hints",
        "superseded_original_values",
    }
    if isinstance(value, dict):
        return any(key in prohibited_keys or _contains_raw_or_operational_leakage(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_raw_or_operational_leakage(item) for item in value)
    return False


def _dedupe_reason_codes(reason_codes: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    deduped = []
    for reason_code, details in reason_codes:
        key = (reason_code, details)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((reason_code, details))
    return deduped


def _record(
    report_set: CompanyReportSet,
    packet: dict[str, Any],
    record_type: str,
    reason_code: str,
    details: str,
) -> dict[str, Any]:
    return {
        "case_id": report_set.case_id,
        "packet_id": packet.get("packet_id"),
        "record_type": record_type,
        "reason_code": reason_code,
        "reason": details,
        "dataset_flow": "real_extracted_detector_packet",
        "target_report_id": report_set.target.report_id,
        "prior_year_report_id": report_set.prior_year.report_id,
        "target_canonical_source_document_id": report_set.target.metadata.get("canonical_source_document_id"),
        "prior_year_canonical_source_document_id": report_set.prior_year.metadata.get("canonical_source_document_id"),
        "source_quality_details": {
            "target": report_set.target.metadata.get("source_quality_details", {}),
            "prior_year": report_set.prior_year.metadata.get("source_quality_details", {}),
        },
    }

