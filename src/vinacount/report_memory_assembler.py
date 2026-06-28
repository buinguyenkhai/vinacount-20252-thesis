from __future__ import annotations

import re
import unicodedata
from typing import Any

from vinacount.filing_package import FilingPackage, select_canonical_source
from vinacount.raw_extraction_artifact import RawExtractionArtifact
from vinacount.report_model import ReportMemory, validate_report_memory


ALLOWED_CURRENCIES = {"VND"}
ALLOWED_UNITS = {"vnd", "thousand_vnd", "million_vnd", "billion_vnd"}


def assemble_report_memory(
    *,
    filing_package: FilingPackage,
    raw_extraction_artifacts: list[RawExtractionArtifact],
) -> ReportMemory:
    selection = select_canonical_source(filing_package)
    corrected_value_resolutions = _validated_corrected_value_resolutions(filing_package, raw_extraction_artifacts)
    corrected_note_resolutions = _validated_corrected_note_resolutions(filing_package, raw_extraction_artifacts)
    canonical_artifact = _canonical_artifact(
        raw_extraction_artifacts,
        selection.canonical_source_document_id,
    )
    candidates = canonical_artifact.raw.get("extraction_candidates")
    if not isinstance(candidates, dict):
        raise ValueError("ReportMemory assembler requires extraction_candidates")

    metadata_candidates = candidates.get("metadata", {})
    if not isinstance(metadata_candidates, dict):
        raise ValueError("ReportMemory extraction_candidates.metadata must be an object")

    canonical_document = _source_document(
        filing_package,
        selection.canonical_source_document_id,
    )
    unit = metadata_candidates.get("unit")
    currency = metadata_candidates.get("currency")
    _validate_currency_and_unit(currency=currency, unit=unit)
    report_id = _report_id(filing_package)
    sections = [_section(section) for section in candidates.get("sections", [])]
    tables = [
        _table(table, unit=unit)
        for table in candidates.get("statement_tables", [])
    ]
    _apply_corrected_value_resolutions(tables, corrected_value_resolutions)
    tool_facing_artifacts = _tool_facing_text_artifacts(
        filing_package,
        raw_extraction_artifacts,
        canonical_source_document_id=selection.canonical_source_document_id,
    )
    notes = _resolve_note_references(_notes(tool_facing_artifacts), tables, canonical_artifact)
    _apply_corrected_note_resolutions(notes, corrected_note_resolutions)
    cell_index = _cell_index(tables)

    raw_report_memory = {
        "report_id": report_id,
        "metadata": {
            "company_name": filing_package.company_name,
            "ticker": filing_package.raw["company"].get("ticker"),
            "period": filing_package.period,
            "report_period_type": metadata_candidates.get("report_period_type", "quarterly"),
            "report_profile": metadata_candidates.get("report_profile"),
            "report_basis": filing_package.report_basis,
            "business_context_tags": metadata_candidates.get("business_context_tags", []),
            "insurance_subprofile": metadata_candidates.get("insurance_subprofile"),
            "industry": metadata_candidates.get("industry"),
            "report_assurance_type": metadata_candidates.get("report_assurance_type", "unknown"),
            "currency": currency,
            "unit": unit,
            "filing_status": selection.filing_status,
            "canonical_source_document_id": selection.canonical_source_document_id,
            "source_document_fingerprint_sha256": canonical_document.raw["fingerprint"]["hash_value"],
            "source_file": canonical_document.raw["local_artifact"]["path"],
            "language": metadata_candidates.get("language"),
            "extraction_method": canonical_artifact.raw["extraction_method"],
            "extraction_limitations": metadata_candidates.get("extraction_limitations", []),
        },
        "sections": sections,
        "tables": tables,
        "notes": notes,
        "variance_explanations": _variance_explanations(tool_facing_artifacts),
        "evidence_surface_status": _evidence_surface_status(tool_facing_artifacts),
        "cell_index": cell_index,
    }
    return validate_report_memory(raw_report_memory)


def _tool_facing_text_artifacts(
    filing_package: FilingPackage,
    artifacts: list[RawExtractionArtifact],
    *,
    canonical_source_document_id: str,
) -> list[RawExtractionArtifact]:
    allowed_document_ids = {canonical_source_document_id}
    allowed_document_ids.update(
        document.document_id
        for document in filing_package.supporting_documents
        if document.document_type in {"variance_explanation", "variance_explanation_attachment"}
    )
    return [
        artifact
        for artifact in artifacts
        if artifact.source_document_id in allowed_document_ids
    ]


def _validated_corrected_value_resolutions(
    filing_package: FilingPackage,
    artifacts: list[RawExtractionArtifact],
) -> list[dict[str, Any]]:
    amendment_context_ids = {
        document.document_id
        for document in filing_package.supporting_documents
        if document.document_type == "amendment_context_attachment"
    }
    resolutions = []
    for artifact in artifacts:
        if artifact.source_document_id not in amendment_context_ids:
            continue
        candidates = artifact.raw.get("extraction_candidates")
        if not isinstance(candidates, dict):
            continue
        artifact_resolutions = candidates.get("corrected_value_resolutions")
        if artifact_resolutions:
            for resolution in artifact_resolutions:
                _validate_corrected_value_resolution(resolution, artifact.source_document_id)
                resolutions.append(resolution)
            continue
        if (
            candidates.get("statement_tables")
            or candidates.get("corrected_value_candidates")
            or artifact_resolutions is not None
        ):
            raise ValueError(
                "Recoverable data/source-quality error: corrected-value resolution required before ReportMemory, tools, or detector packets"
            )
    return resolutions


def _validated_corrected_note_resolutions(
    filing_package: FilingPackage,
    artifacts: list[RawExtractionArtifact],
) -> list[dict[str, Any]]:
    amendment_context_ids = {
        document.document_id
        for document in filing_package.supporting_documents
        if document.document_type == "amendment_context_attachment"
    }
    resolutions = []
    for artifact in artifacts:
        if artifact.source_document_id not in amendment_context_ids:
            continue
        candidates = artifact.raw.get("extraction_candidates")
        if not isinstance(candidates, dict):
            continue
        artifact_resolutions = candidates.get("corrected_note_resolutions")
        if artifact_resolutions:
            for resolution in artifact_resolutions:
                _validate_corrected_note_resolution(resolution, artifact.source_document_id)
                resolutions.append(resolution)
            continue
        if candidates.get("corrected_note_candidates") or artifact_resolutions is not None:
            raise ValueError(
                "Recoverable data/source-quality error: corrected-note resolution required before ReportMemory, tools, or detector packets"
            )
    return resolutions


def _validate_corrected_value_resolution(resolution: Any, amendment_context_id: str) -> None:
    if not isinstance(resolution, dict):
        raise ValueError("Recoverable data/source-quality error: corrected-value resolution must be structured")
    if resolution.get("resolution_status") != "resolved" or resolution.get("basis") != "deterministic_one_to_one":
        raise ValueError(
            "Recoverable data/source-quality error: corrected-value resolution required before ReportMemory, tools, or detector packets"
        )
    target = resolution.get("target")
    corrected = resolution.get("corrected")
    if not isinstance(target, dict) or not isinstance(corrected, dict):
        raise ValueError("Recoverable data/source-quality error: corrected-value resolution must include target and corrected values")
    required_target = {"source_document_id", "standard_account", "period_basis", "period", "original_value"}
    required_corrected = {"source_document_id", "value"}
    if required_target - target.keys() or required_corrected - corrected.keys():
        raise ValueError("Recoverable data/source-quality error: corrected-value resolution is incomplete")
    if corrected["source_document_id"] != amendment_context_id:
        raise ValueError("Recoverable data/source-quality error: corrected value must trace to the amendment context attachment")


def _validate_corrected_note_resolution(resolution: Any, amendment_context_id: str) -> None:
    if not isinstance(resolution, dict):
        raise ValueError("Recoverable data/source-quality error: corrected-note resolution must be structured")
    if resolution.get("resolution_status") != "resolved" or resolution.get("basis") != "deterministic_note_span_replacement":
        raise ValueError(
            "Recoverable data/source-quality error: corrected-note resolution required before ReportMemory, tools, or detector packets"
        )
    target = resolution.get("target")
    corrected = resolution.get("corrected")
    if not isinstance(target, dict) or not isinstance(corrected, dict):
        raise ValueError("Recoverable data/source-quality error: corrected-note resolution must include target and corrected text")
    required_target = {"source_document_id", "original_text"}
    required_corrected = {"source_document_id", "text"}
    if required_target - target.keys() or required_corrected - corrected.keys():
        raise ValueError("Recoverable data/source-quality error: corrected-note resolution is incomplete")
    if not target.get("note_id") and not target.get("note_number"):
        raise ValueError("Recoverable data/source-quality error: corrected-note resolution target note is incomplete")
    if corrected["source_document_id"] != amendment_context_id:
        raise ValueError("Recoverable data/source-quality error: corrected note must trace to the amendment context attachment")


def _apply_corrected_value_resolutions(
    tables: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
) -> None:
    seen_targets = set()
    for resolution in resolutions:
        target = resolution["target"]
        target_key = (
            target["source_document_id"],
            target["standard_account"],
            target["period_basis"],
            target["period"],
        )
        if target_key in seen_targets:
            raise ValueError("Recoverable data/source-quality error: ambiguous corrected-value resolution")
        seen_targets.add(target_key)
        matches = [
            (table, row, cell)
            for table in tables
            if table.get("source_document_id") == target["source_document_id"]
            and table.get("period_basis") == target["period_basis"]
            for row in table.get("rows", [])
            if row.get("standard_account") == target["standard_account"]
            for cell in row.get("cells", [])
            if cell.get("period") == target["period"]
        ]
        if len(matches) != 1:
            raise ValueError("Recoverable data/source-quality error: corrected-value resolution did not map to one original value")
        table, row, cell = matches[0]
        if cell.get("value") != target["original_value"]:
            raise ValueError("Recoverable data/source-quality error: corrected-value resolution original value mismatch")
        corrected = resolution["corrected"]
        cell["value"] = corrected["value"]
        cell["source_document_id"] = corrected["source_document_id"]
        cell["amendment_resolution"] = {
            "basis": resolution["basis"],
            "amendment_context_attachment_id": corrected["source_document_id"],
            "superseded_source_document_id": target["source_document_id"],
            "superseded_value": target["original_value"],
        }
        row["source_document_id"] = corrected["source_document_id"]
        table["source_document_id"] = corrected["source_document_id"]


def _apply_corrected_note_resolutions(
    notes: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
) -> None:
    seen_targets = set()
    for resolution in resolutions:
        target = resolution["target"]
        target_key = (
            target["source_document_id"],
            target.get("note_id"),
            target.get("note_number"),
        )
        if target_key in seen_targets:
            raise ValueError("Recoverable data/source-quality error: ambiguous corrected-note resolution")
        seen_targets.add(target_key)
        matches = [
            note
            for note in notes
            if note.get("source_document_id") == target["source_document_id"]
            and (
                (target.get("note_id") and note.get("note_id") == target.get("note_id"))
                or (target.get("note_number") and note.get("note_number") == target.get("note_number"))
            )
        ]
        if len(matches) != 1:
            raise ValueError("Recoverable data/source-quality error: corrected-note resolution did not map to one original note")
        note = matches[0]
        if note.get("text") != target["original_text"]:
            raise ValueError("Recoverable data/source-quality error: corrected-note resolution original text mismatch")
        corrected = resolution["corrected"]
        note["text"] = corrected["text"]
        note["amendment_resolution"] = {
            "basis": resolution["basis"],
            "amendment_context_attachment_id": corrected["source_document_id"],
            "superseded_source_document_id": target["source_document_id"],
        }


def _validate_currency_and_unit(*, currency: Any, unit: Any) -> None:
    if currency not in ALLOWED_CURRENCIES:
        raise ValueError("ReportMemory currency must be VND")
    if unit not in ALLOWED_UNITS:
        raise ValueError("ReportMemory unit must be vnd, thousand_vnd, million_vnd, or billion_vnd")


def _canonical_artifact(
    artifacts: list[RawExtractionArtifact],
    canonical_source_document_id: str,
) -> RawExtractionArtifact:
    matching = [
        artifact
        for artifact in artifacts
        if artifact.source_document_id == canonical_source_document_id
    ]
    if len(matching) != 1:
        raise ValueError("ReportMemory assembler requires one canonical raw extraction artifact")
    return matching[0]


def _source_document(filing_package: FilingPackage, source_document_id: str):
    for source_document in filing_package.source_documents:
        if source_document.document_id == source_document_id:
            return source_document
    raise ValueError("ReportMemory canonical source_document_id must exist in FilingPackage")


def _variance_explanations(artifacts: list[RawExtractionArtifact]) -> list[dict[str, Any]]:
    explanations = []
    for artifact in artifacts:
        candidates = artifact.raw.get("extraction_candidates")
        if not isinstance(candidates, dict):
            continue
        for explanation in candidates.get("variance_explanations", []):
            if not isinstance(explanation, dict):
                continue
            explanations.append(_tool_facing_variance_explanation(explanation, artifact))
    return explanations


def _evidence_surface_status(artifacts: list[RawExtractionArtifact]) -> list[dict[str, Any]]:
    statuses = []
    for artifact in artifacts:
        candidates = artifact.raw.get("extraction_candidates")
        if not isinstance(candidates, dict):
            continue
        for status in candidates.get("evidence_surface_status", []):
            if not isinstance(status, dict):
                continue
            cleaned = _tool_facing_evidence_surface_status(status, artifact)
            if cleaned:
                statuses.append(cleaned)
    return statuses


def _tool_facing_evidence_surface_status(
    status: dict[str, Any],
    artifact: RawExtractionArtifact,
) -> dict[str, Any] | None:
    allowed_surfaces = {
        "notes",
        "variance_explanations",
        "related_party_evidence",
        "accounting_policy_evidence",
        "extraction_quality",
    }
    allowed_states = {
        "present",
        "absent_in_source",
        "not_applicable",
        "unsupported_by_extraction_path",
        "not_extracted_yet",
        "ambiguous_failed_closed",
    }
    surface = status.get("surface")
    state = status.get("state")
    source_document_id = status.get("source_document_id") or artifact.source_document_id
    evidence_ref = status.get("evidence_ref")
    if surface not in allowed_surfaces or state not in allowed_states:
        return None
    if source_document_id != artifact.source_document_id:
        return None
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        return None
    cleaned = {
        "surface": surface,
        "state": state,
        "source_document_id": source_document_id,
        "evidence_ref": evidence_ref,
    }
    for key in ("producer", "producer_version", "reason_code", "message"):
        value = status.get(key)
        if isinstance(value, str) and value.strip():
            cleaned[key] = value
    return cleaned


def _notes(artifacts: list[RawExtractionArtifact]) -> list[dict[str, Any]]:
    notes = []
    for artifact in artifacts:
        candidates = artifact.raw.get("extraction_candidates")
        if not isinstance(candidates, dict):
            continue
        for note in candidates.get("notes", []):
            if not isinstance(note, dict):
                continue
            notes.append(_tool_facing_note(note, artifact))
    return notes


def _tool_facing_note(note: dict[str, Any], artifact: RawExtractionArtifact) -> dict[str, Any]:
    allowed_fields = {
        "note_id",
        "section_id",
        "note_type",
        "note_number",
        "title",
        "source_document_id",
        "text",
        "linked_table_ids",
        "linked_row_ids",
        "linked_cell_ids",
        "periods",
        "status",
        "evidence_provenance",
    }
    cleaned = {
        key: value
        for key, value in note.items()
        if key in allowed_fields
    }
    cleaned.setdefault("source_document_id", artifact.source_document_id)
    _copy_sanitized_evidence_provenance(cleaned, note, artifact)
    return cleaned


def _tool_facing_variance_explanation(
    explanation: dict[str, Any],
    artifact: RawExtractionArtifact,
) -> dict[str, Any]:
    allowed_fields = {
        "span_id",
        "section_id",
        "title",
        "source_heading",
        "source_document_id",
        "text",
        "related_metric",
        "period_basis",
        "report_basis",
        "related_table_ids",
        "related_row_ids",
        "periods",
        "status",
        "evidence_provenance",
    }
    cleaned = {
        key: value
        for key, value in explanation.items()
        if key in allowed_fields
    }
    cleaned.setdefault("source_document_id", artifact.source_document_id)
    _copy_sanitized_evidence_provenance(cleaned, explanation, artifact)
    return cleaned


def _copy_sanitized_evidence_provenance(
    cleaned: dict[str, Any],
    source: dict[str, Any],
    artifact: RawExtractionArtifact,
) -> None:
    provenance = source.get("evidence_provenance")
    if not isinstance(provenance, dict):
        cleaned.pop("evidence_provenance", None)
        return
    source_document_id = provenance.get("source_document_id")
    if source_document_id != cleaned.get("source_document_id") or source_document_id != artifact.source_document_id:
        cleaned.pop("evidence_provenance", None)
        return
    sanitized = {"source_document_id": source_document_id}
    page = provenance.get("page")
    if isinstance(page, int):
        sanitized["page"] = page
    extraction_method = provenance.get("extraction_method")
    if isinstance(extraction_method, str) and extraction_method.strip():
        sanitized["extraction_method"] = extraction_method
    cleaned["evidence_provenance"] = sanitized


def _report_id(filing_package: FilingPackage) -> str:
    ticker = filing_package.raw["company"].get("ticker")
    company_slug = filing_package.company_name.upper().replace(" ", "_")
    period_slug = filing_package.period.replace("-", "_")
    return f"{ticker or company_slug}_{period_slug}"


def _section(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "section_id": section["section_id"],
        "section_type": section["section_type"],
        "title": section["title"],
        "status": section.get("status", "included"),
        "page_range": section.get("page_range"),
        "ignore_reason": section.get("ignore_reason"),
    }


def _table(table: dict[str, Any], *, unit: str) -> dict[str, Any]:
    rows = [_row(row, table=table, unit=unit) for row in table.get("rows", [])]
    return {
        "table_id": table["table_id"],
        "section_id": table["section_id"],
        "table_type": table["table_type"],
        "title": table["title"],
        "period_basis": table.get("period_basis"),
        "source_document_id": table["source_document_id"],
        "periods": _periods(rows),
        "rows": rows,
    }


def _row(row: dict[str, Any], *, table: dict[str, Any], unit: str) -> dict[str, Any]:
    period_slug = _period_slug(row["cells"][0]["period"])
    standard_account = row["standard_account"]
    row_id = row.get("row_id", f"ROW_{standard_account.upper()}_{period_slug}")
    cells = []
    for cell in row.get("cells", []):
        assembled_cell = {
            "cell_id": cell.get("cell_id", f"CELL_{standard_account.upper()}_{_period_slug(cell['period'])}"),
            "period": cell["period"],
            "value": cell["value"],
            "unit": cell.get("unit", unit),
            **({"source_document_id": cell["source_document_id"]} if cell.get("source_document_id") else {}),
            **({"amendment_resolution": cell["amendment_resolution"]} if cell.get("amendment_resolution") else {}),
        }
        page_number = cell.get("source_page_number", row.get("source_page_number"))
        if isinstance(page_number, int) and page_number > 0:
            assembled_cell["source_page_number"] = page_number
        source_excerpt = cell.get("source_excerpt") or row.get("source_excerpt")
        if isinstance(source_excerpt, str) and source_excerpt.strip():
            assembled_cell["source_excerpt"] = source_excerpt.strip()
        cells.append(assembled_cell)
    assembled_row = {
        "row_id": row_id,
        "account_code": row.get("account_code"),
        "standard_account": standard_account,
        "account_group": row["account_group"],
        "label": row["label"],
        "original_label": row.get("original_label"),
        "source_document_id": table["source_document_id"],
        "cells": cells,
    }
    if row.get("note_ref"):
        assembled_row["_note_ref"] = row["note_ref"]
        assembled_row["_note_title"] = row.get("note_title")
        assembled_row["_note_link_required"] = row.get("note_link_required", True)
    return assembled_row


def _periods(rows: list[dict[str, Any]]) -> list[str]:
    seen = set()
    ordered = []
    for row in rows:
        for cell in row["cells"]:
            period = cell["period"]
            if period in seen:
                continue
            seen.add(period)
            ordered.append(period)
    return ordered


def _cell_index(tables: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index = {}
    for table in tables:
        for row in table["rows"]:
            for cell in row["cells"]:
                cell_id = cell["cell_id"]
                if cell_id in index:
                    raise ValueError("ReportMemory cell_index contains duplicate cell evidence IDs")
                index[cell_id] = {
                    "table_id": table["table_id"],
                    "row_id": row["row_id"],
                    "period": cell["period"],
                    "value": cell["value"],
                    "unit": cell["unit"],
                    "source_document_id": cell.get("source_document_id", table["source_document_id"]),
                }
                if cell.get("amendment_resolution"):
                    index[cell_id]["amendment_resolution"] = cell["amendment_resolution"]
                if cell.get("source_page_number"):
                    index[cell_id]["source_page_number"] = cell["source_page_number"]
                if cell.get("source_excerpt"):
                    index[cell_id]["source_excerpt"] = cell["source_excerpt"]
    return index


def _resolve_note_references(
    note_candidates: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    artifact: RawExtractionArtifact,
) -> list[dict[str, Any]]:
    notes = [dict(note) for note in note_candidates]
    for table in tables:
        for row in table["rows"]:
            note_ref = row.pop("_note_ref", None)
            note_title = row.pop("_note_title", None)
            note_link_required = row.pop("_note_link_required", True)
            if not note_ref:
                continue
            if not note_title and not note_link_required:
                _record_unresolved_note_reference_warning(artifact, row["row_id"])
                continue
            matched_note = _strict_note_match(notes, note_ref=note_ref, note_title=note_title, row=row)
            row["linked_note_ids"] = _append_unique(row.get("linked_note_ids", []), matched_note["note_id"])
            matched_note["linked_table_ids"] = _append_unique(matched_note.get("linked_table_ids", []), table["table_id"])
            matched_note["linked_row_ids"] = _append_unique(matched_note.get("linked_row_ids", []), row["row_id"])
            for cell in row["cells"]:
                matched_note["linked_cell_ids"] = _append_unique(
                    matched_note.get("linked_cell_ids", []),
                    cell["cell_id"],
                )
    return notes


def _record_unresolved_note_reference_warning(artifact: RawExtractionArtifact, row_id: str) -> None:
    artifact.raw.setdefault("parser_warnings", []).append(
        {
            "warning_id": f"WARN_{row_id}_NOTE_REFERENCE_UNRESOLVED",
            "message": f"Irrelevant note reference hint was not resolved for row {row_id}.",
        }
    )


def _strict_note_match(
    notes: list[dict[str, Any]],
    *,
    note_ref: str,
    note_title: str | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    if not note_title:
        raise ValueError("Recoverable setup/data error: Note Reference Match requires note title")
    normalized_ref = _normalize_note_ref(note_ref)
    normalized_titles = {
        _normalize_text(note_title),
        _normalize_text(row.get("label")),
        _normalize_text(row.get("original_label")),
    }
    normalized_titles.discard("")
    matches = [
        note
        for note in notes
        if _normalize_note_ref(note.get("note_number")) == normalized_ref
        and _normalize_text(note.get("title")) in normalized_titles
    ]
    if len(matches) != 1:
        raise ValueError("Recoverable setup/data error: Note Reference Match is ambiguous or missing")
    return matches[0]


def _append_unique(values: list[str], value: str) -> list[str]:
    if value in values:
        return values
    return values + [value]


def _normalize_note_ref(value: Any) -> str:
    text = _normalize_text(value)
    text = re.sub(r"[^a-z0-9.]+", "", text)
    parts = [part for part in text.split(".") if part]
    normalized_parts = []
    for part in parts:
        if part.isdigit():
            normalized_parts.append(str(int(part)))
        else:
            normalized_parts.append(str(_roman_to_int(part) or part))
    return ".".join(normalized_parts)


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFD", value.strip().casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"[^\w\s.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def _roman_to_int(value: str) -> int | None:
    roman_values = {"i": 1, "v": 5, "x": 10, "l": 50}
    if not value or any(char not in roman_values for char in value):
        return None
    total = 0
    previous = 0
    for char in reversed(value):
        current = roman_values[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def _period_slug(period: str) -> str:
    return period.replace("-", "_")
