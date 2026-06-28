from __future__ import annotations

from typing import Any, Iterable

from vinacount.canonical_source_selection import CanonicalSourceSelectionResult
from vinacount.filing_package import FilingPackage, SourceDocument
from vinacount.source_document_classifier import SourceDocumentClassification


HITL_SCOPE = "source_identity_confirmation_before_extraction_or_cached_reuse"


def build_source_confirmation_payload(
    source_packages: Iterable[
        tuple[
            str,
            FilingPackage,
            CanonicalSourceSelectionResult,
            list[SourceDocumentClassification],
        ]
    ],
) -> dict[str, Any]:
    packages = [
        _source_package_payload(role, package, selection_result, classifications)
        for role, package, selection_result, classifications in source_packages
    ]
    status = _payload_status(packages)
    return {
        "checkpoint": "source_confirmation",
        "hitl_boundary": {
            "before": "extraction_or_cached_artifact_reuse",
            "allowed_decisions": ["confirm_source_identity", "reject_source_identity", "resolve_source_identity_ambiguity"],
            "disallowed_decisions": [
                "extraction_repair",
                "manual_value_editing",
                "silent_basis_switching",
                "english_source_fallback_for_normal_path",
                "original_amended_value_mixing",
            ],
        },
        "status": status,
        "hitl_needed": any(item["hitl"]["needed"] for item in packages),
        "source_packages": packages,
    }


def _source_package_payload(
    role: str,
    package: FilingPackage,
    selection_result: CanonicalSourceSelectionResult,
    classifications: list[SourceDocumentClassification],
) -> dict[str, Any]:
    classifications_by_id = {item.source_document_id: item for item in classifications}
    canonical_id = selection_result.selection.canonical_source_document_id if selection_result.selection else None
    canonical_document = _document_by_id(package, canonical_id)
    canonical_classification = classifications_by_id.get(canonical_id or "")
    hitl_needed = selection_result.status != "selected"
    return {
        "role": role,
        "ticker": (package.raw.get("company") or {}).get("ticker"),
        "company_name": package.company_name,
        "year": _year(package.period),
        "quarter": package.raw.get("quarter"),
        "period": package.period,
        "requested_basis": package.report_basis,
        "selected_basis_family": _selected_basis_family(selection_result, canonical_classification),
        "visible_filing_label": _visible_label(selection_result, canonical_id, canonical_classification),
        "source_reference": _source_reference(canonical_document),
        "canonical_source_document_id": canonical_id,
        "language": canonical_classification.language if canonical_classification else None,
        "document_role": canonical_classification.document_role if canonical_classification else None,
        "searchable_status": _searchable_status(canonical_document, canonical_classification),
        "assurance": _assurance(selection_result, canonical_classification),
        "supersession": _supersession(selection_result, classifications_by_id),
        "supporting_documents": _supporting_documents(selection_result, package, classifications_by_id),
        "candidate_documents": _candidate_documents(
            package,
            selection_result,
            classifications,
        ),
        "records": [_record_payload(record) for record in selection_result.records],
        "warnings": [_record_payload(record) for record in selection_result.records if record.severity in {"warning", "hitl_needed"}],
        "errors": [
            _record_payload(record)
            for record in selection_result.records
            if record.severity in {"recoverable_data_error", "error"}
        ],
        "hitl": _hitl_payload(hitl_needed, selection_result.code),
        "audit_references": _audit_references(package, selection_result),
        "basis_profile_consistency": {
            "requested_basis": package.report_basis,
            "selected_basis_family": _selected_basis_family(selection_result, canonical_classification),
            "basis_switch_approved": False,
            "normal_language": "vi",
            "english_fallback_approved": False,
        },
    }


def _payload_status(packages: list[dict[str, Any]]) -> str:
    if any(item["hitl"]["needed"] for item in packages):
        return "hitl_needed"
    if any(item["errors"] for item in packages):
        return "ready_for_confirmation_with_warnings"
    return "ready_for_confirmation"


def _hitl_payload(needed: bool, reason: str | None) -> dict[str, Any]:
    payload = {"needed": needed, "scope": HITL_SCOPE}
    if needed:
        payload["reason"] = reason
    return payload


def _document_by_id(package: FilingPackage, document_id: str | None) -> SourceDocument | None:
    for document in package.source_documents:
        if document.document_id == document_id:
            return document
    return None


def _year(period: str) -> int | None:
    try:
        return int(str(period).split("-")[0])
    except (TypeError, ValueError):
        return None


def _selected_basis_family(
    selection_result: CanonicalSourceSelectionResult,
    classification: SourceDocumentClassification | None,
) -> str | None:
    if selection_result.selection and selection_result.selection.selected_basis_family:
        return selection_result.selection.selected_basis_family
    return classification.basis_family if classification else None


def _visible_label(
    selection_result: CanonicalSourceSelectionResult,
    canonical_id: str | None,
    classification: SourceDocumentClassification | None,
) -> str | None:
    labels = selection_result.selection.visible_filing_label_by_document_id if selection_result.selection else None
    if canonical_id and labels and labels.get(canonical_id):
        return labels[canonical_id]
    return classification.visible_filing_label if classification else None


def _source_reference(document: SourceDocument | None) -> dict[str, str | None]:
    if document is None:
        return {"source_name": None, "source_url": None, "display_reference": None}
    provenance = document.provenance
    source_url = provenance.get("source_url")
    return {
        "source_name": provenance.get("source_name"),
        "source_url": source_url,
        "display_reference": source_url or provenance.get("source_name"),
    }


def _searchable_status(
    document: SourceDocument | None,
    classification: SourceDocumentClassification | None,
) -> dict[str, Any]:
    text = " ".join(
        value
        for value in [
            (classification.visible_filing_label if classification else None),
            *list(classification.visible_evidence_snippets if classification else []),
            (_source_reference(document).get("display_reference") if document else None),
        ]
        if isinstance(value, str)
    ).lower()
    is_searchable = "tra cứu" in text or "tra cuu" in text or "searchable" in text
    return {
        "is_searchable": is_searchable,
        "reason": "visible_label_or_source_reference_contains_tra_cuu" if is_searchable else None,
    }


def _assurance(
    selection_result: CanonicalSourceSelectionResult,
    classification: SourceDocumentClassification | None,
) -> dict[str, Any]:
    assurance_type = selection_result.selection.assurance_type if selection_result.selection else None
    return {
        "type": assurance_type,
        "visible_hints": list(classification.assurance_hints if classification else []),
    }


def _supersession(
    selection_result: CanonicalSourceSelectionResult,
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> dict[str, Any]:
    if selection_result.selection is None:
        return {"superseded_source_document_ids": [], "provenance": []}
    superseded_ids = list(selection_result.selection.superseded_source_document_ids or [])
    return {
        "superseded_source_document_ids": superseded_ids,
        "provenance": [
            {
                "source_document_id": source_document_id,
                "visible_filing_label": classifications_by_id.get(source_document_id).visible_filing_label
                if classifications_by_id.get(source_document_id)
                else None,
            }
            for source_document_id in superseded_ids
        ],
    }


def _supporting_documents(
    selection_result: CanonicalSourceSelectionResult,
    package: FilingPackage,
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> list[dict[str, Any]]:
    if selection_result.selection is None:
        return []
    links = selection_result.selection.supporting_document_links or []
    summaries = []
    for link in links:
        source_document_id = str(link["source_document_id"])
        classification = classifications_by_id.get(source_document_id)
        document = _document_by_id(package, source_document_id)
        summaries.append(
            {
                "source_document_id": source_document_id,
                "document_role": link.get("document_role"),
                "visible_filing_label": link.get("visible_filing_label") or (classification.visible_filing_label if classification else None),
                "language": classification.language if classification else None,
                "source_reference": _source_reference(document),
                "linked_canonical_source_document_id": link.get("linked_canonical_source_document_id"),
                "requires_corrected_value_resolution": bool(link.get("requires_corrected_value_resolution")),
                "corrected_value_resolution_status": link.get("corrected_value_resolution_status"),
            }
        )
    return summaries


def _candidate_documents(
    package: FilingPackage,
    selection_result: CanonicalSourceSelectionResult,
    classifications: list[SourceDocumentClassification],
) -> list[dict[str, Any]]:
    if selection_result.status != "hitl_needed" or selection_result.selection is not None:
        return []
    documents_by_id = {document.document_id: document for document in package.source_documents}
    candidates = []
    for classification in classifications:
        document = documents_by_id.get(classification.source_document_id)
        if document is None:
            continue
        if classification.state != "classified":
            continue
        if classification.document_role != "financial_statement" or not classification.is_full_financial_statement:
            continue
        if classification.language not in {"vi", "mixed"}:
            continue
        if classification.basis_family != package.report_basis:
            continue
        candidates.append(
            {
                "source_document_id": document.document_id,
                "company_name_vi": _first_or_none(classification.company_evidence) or package.company_name,
                "ticker": (package.raw.get("company") or {}).get("ticker"),
                "period_label": package.period,
                "quarter": _quarter_number(package.raw.get("quarter")),
                "fiscal_year": _year(package.period),
                "report_basis": package.report_basis,
                "filing_status": (package.raw.get("filing_event") or {}).get("filing_status"),
                "document_type": document.document_type,
                "language": classification.language,
                "source_origin": document.provenance.get("source_kind"),
                "source_name": document.provenance.get("source_name"),
                "source_url": document.provenance.get("source_url"),
                "is_searchable_version": _searchable_status(document, classification)["is_searchable"],
                "file_size_bytes": (document.raw.get("local_artifact") or {}).get("file_size_bytes"),
                "page_count": document.raw.get("page_count"),
                "visible_filing_label": classification.visible_filing_label,
                "first_page_identity": {
                    "visible_company_name": _first_or_none(classification.company_evidence),
                    "visible_period": _first_or_none(classification.period_evidence),
                    "visible_basis_clue": classification.basis_family,
                },
                "classification_evidence": list(classification.visible_evidence_snippets),
                "audit_references": {
                    "package_id": package.package_id,
                    "event_id": (package.raw.get("filing_event") or {}).get("event_id"),
                    "canonical_source_document_id": document.document_id,
                    "source_document_fingerprint_sha256": (document.raw.get("fingerprint") or {}).get("hash_value"),
                },
            }
        )
    return candidates


def _first_or_none(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _quarter_number(value: Any) -> int | None:
    text = str(value or "").upper()
    if text.startswith("Q"):
        text = text[1:]
    try:
        quarter = int(text)
    except ValueError:
        return None
    return quarter if quarter in {1, 2, 3, 4} else None


def _record_payload(record: Any) -> dict[str, str | None]:
    return {
        "severity": record.severity,
        "code": record.code,
        "message": record.message,
        "source_document_id": record.source_document_id,
    }


def _audit_references(
    package: FilingPackage,
    selection_result: CanonicalSourceSelectionResult,
) -> dict[str, Any]:
    return {
        "package_id": package.package_id,
        "event_id": (package.raw.get("filing_event") or {}).get("event_id"),
        "canonical_source_document_id": selection_result.selection.canonical_source_document_id
        if selection_result.selection
        else None,
        "supporting_source_document_ids": list(selection_result.selection.supporting_document_ids or [])
        if selection_result.selection
        else [],
    }
