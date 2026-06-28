from __future__ import annotations

import unicodedata
from dataclasses import dataclass
import re

from vinacount.filing_package import CanonicalSourceSelection, FilingPackage
from vinacount.source_document_classifier import SourceDocumentClassification


NON_CONSOLIDATED_BASIS_FAMILIES = {"parent", "separate", "non_consolidated"}
SUPPORTING_DOCUMENT_ROLES = {
    "amendment_context_attachment",
    "cover_disclosure_letter",
    "cover_or_disclosure",
    "variance_explanation",
}


@dataclass(frozen=True)
class CanonicalSourceSelectionRecord:
    severity: str
    code: str
    message: str
    source_document_id: str | None = None


@dataclass(frozen=True)
class CanonicalSourceSelectionResult:
    status: str
    code: str | None
    selection: CanonicalSourceSelection | None
    records: list[CanonicalSourceSelectionRecord]
    ignored_document_ids: list[str]


def select_canonical_source_from_classifications(
    package: FilingPackage,
    classifications: list[SourceDocumentClassification],
) -> CanonicalSourceSelectionResult:
    classifications_by_id = {item.source_document_id: item for item in classifications}
    candidate_ids = []
    supporting_ids = []
    ignored_ids = []
    records: list[CanonicalSourceSelectionRecord] = []

    for document in package.source_documents:
        classification = classifications_by_id.get(document.document_id)
        if classification is None:
            ignored_ids.append(document.document_id)
            records.append(
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="missing_classifier_record",
                    message="Source document has no classifier record.",
                    source_document_id=document.document_id,
                )
            )
            continue
        if classification.state != "classified":
            ignored_ids.append(document.document_id)
            records.append(
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="inconclusive_classifier_record",
                    message="Inconclusive classifier record cannot become canonical.",
                    source_document_id=document.document_id,
                )
            )
            continue
        if classification.is_supporting_document or classification.document_role in SUPPORTING_DOCUMENT_ROLES:
            if _is_english_supporting_duplicate(package, classification, classifications):
                ignored_ids.append(document.document_id)
                records.append(
                    CanonicalSourceSelectionRecord(
                        severity="info",
                        code="english_supporting_duplicate_ignored",
                        message="English supporting duplicate is outside normal source selection because a matching Vietnamese supporting document exists.",
                        source_document_id=document.document_id,
                    )
                )
                continue
            supporting_ids.append(document.document_id)
            continue
        if _is_requested_vietnamese_main_fs(package, classification):
            candidate_ids.append(document.document_id)
        else:
            ignored_ids.append(document.document_id)
            if _is_english_requested_basis_main_fs(package, classification):
                records.append(
                    CanonicalSourceSelectionRecord(
                        severity="info",
                        code="english_financial_statement_duplicate_ignored",
                        message="English financial statement is outside normal Vietnamese canonical source selection.",
                        source_document_id=document.document_id,
                    )
                )

    if (
        len(candidate_ids) == 1
        and _is_searchable_version(package, candidate_ids[0])
        and not _identity_confirms_requested_filing(package, classifications_by_id[candidate_ids[0]])
    ):
        return CanonicalSourceSelectionResult(
            status="recoverable_data_error",
            code="searchable_identity_not_confirmed",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="searchable_identity_not_confirmed",
                    message="Only in-scope Vietnamese financial statement is searchable, but identity checks did not confirm the requested filing.",
                    source_document_id=candidate_ids[0],
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    inconclusive_reviewed_ids = _inconclusive_reviewed_full_fs_identity_ids(package, classifications, candidate_ids)
    if inconclusive_reviewed_ids:
        return CanonicalSourceSelectionResult(
            status="recoverable_data_error",
            code="inconclusive_reviewed_full_fs_identity",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="recoverable_data_error",
                    code="inconclusive_reviewed_full_fs_identity",
                    message="A possible reviewed full financial statement has inconclusive identity and may supersede the unreviewed candidate.",
                    source_document_id=inconclusive_reviewed_ids[0],
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    preferred = _select_preferred_candidate(package, candidate_ids, classifications_by_id)
    if preferred is not None:
        canonical_id, reason_code, reason_message, superseded_ids = preferred
        filing_status = _filing_status(classifications_by_id[canonical_id])
        amendment_context_attachment_ids = _linked_amendment_context_attachment_ids(
            package,
            canonical_id,
            supporting_ids,
            classifications_by_id,
        )
        supporting_document_links = _supporting_document_links(
            package,
            canonical_id,
            supporting_ids,
            classifications_by_id,
        )
        selection = CanonicalSourceSelection(
            canonical_source_document_id=canonical_id,
            filing_status=filing_status,
            supporting_document_ids=supporting_ids,
            requested_report_basis=package.report_basis,
            selected_basis_family=classifications_by_id[canonical_id].basis_family,
            visible_filing_label_by_document_id={
                item.source_document_id: item.visible_filing_label
                for item in classifications
                if item.visible_filing_label
            },
            assurance_type=_assurance_type(classifications_by_id[canonical_id]),
            superseded_source_document_ids=superseded_ids,
            amendment_context_attachment_ids=amendment_context_attachment_ids,
            supporting_document_links=supporting_document_links,
        )
        correction_records = [
            CanonicalSourceSelectionRecord(
                severity="recoverable_data_error",
                code="corrected_value_resolution_required",
                message="Amendment context attachment may affect operative values; corrected-value resolution is required before tools and detector packets.",
                source_document_id=attachment_id,
            )
            for attachment_id in amendment_context_attachment_ids
        ]
        return CanonicalSourceSelectionResult(
            status="selected",
            code=None,
            selection=selection,
            records=[
                *records,
                *correction_records,
                CanonicalSourceSelectionRecord(
                    severity="info",
                    code=reason_code,
                    message=reason_message,
                    source_document_id=canonical_id,
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    reviewed_order_blocker = _reviewed_supersession_order_blocker(package, candidate_ids, classifications_by_id)
    if reviewed_order_blocker is not None:
        return CanonicalSourceSelectionResult(
            status="hitl_needed",
            code="ambiguous_reviewed_supersession_order",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="hitl_needed",
                    code="ambiguous_reviewed_supersession_order",
                    message="Reviewed and unreviewed same-identity full financial statements cannot be ordered by source provenance.",
                    source_document_id=reviewed_order_blocker,
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    if len(candidate_ids) > 1:
        return CanonicalSourceSelectionResult(
            status="hitl_needed",
            code="ambiguous_financial_statement_candidates",
            selection=None,
            records=[
                *records,
                CanonicalSourceSelectionRecord(
                    severity="hitl_needed",
                    code="ambiguous_financial_statement_candidates",
                    message="Multiple requested-basis Vietnamese financial statements cannot be ordered by locked policy.",
                ),
            ],
            ignored_document_ids=ignored_ids,
        )

    return CanonicalSourceSelectionResult(
        status="recoverable_data_error",
        code="missing_requested_basis_vietnamese_financial_statement",
        selection=None,
        records=[
            *records,
            CanonicalSourceSelectionRecord(
                severity="recoverable_data_error",
                code="missing_requested_basis_vietnamese_financial_statement",
                message="No classified requested-basis Vietnamese main financial statement is selectable.",
            ),
        ],
        ignored_document_ids=ignored_ids,
    )


def _linked_amendment_context_attachment_ids(
    package: FilingPackage,
    canonical_id: str,
    supporting_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> list[str]:
    canonical = classifications_by_id[canonical_id]
    linked_ids = []
    for supporting_id in supporting_ids:
        supporting = classifications_by_id[supporting_id]
        if (
            supporting.document_role == "amendment_context_attachment"
            and _same_supporting_identity(package, canonical, supporting)
        ):
            linked_ids.append(supporting_id)
    return linked_ids


def _supporting_document_links(
    package: FilingPackage,
    canonical_id: str,
    supporting_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> list[dict[str, str | bool | None]]:
    links = []
    canonical = classifications_by_id[canonical_id]
    for supporting_id in supporting_ids:
        attachment = classifications_by_id[supporting_id]
        if not _same_supporting_identity(package, canonical, attachment):
            continue
        requires_corrected_value_resolution = attachment.document_role == "amendment_context_attachment"
        links.append(
            {
                "source_document_id": supporting_id,
                "document_role": attachment.document_role,
                "visible_filing_label": attachment.visible_filing_label,
                "source_provenance": _bounded_source_provenance(package, supporting_id),
                "linked_canonical_source_document_id": canonical_id,
                "link_basis": "clear_company_period_quarter_basis_language_identity",
                "requires_corrected_value_resolution": requires_corrected_value_resolution,
                "corrected_value_resolution_status": "not_implemented" if requires_corrected_value_resolution else None,
            }
        )
    return links


def _bounded_source_provenance(package: FilingPackage, source_document_id: str) -> dict[str, str | None]:
    for document in package.source_documents:
        if document.document_id == source_document_id:
            provenance = document.provenance
            return {
                "source_kind": provenance.get("source_kind"),
                "source_name": provenance.get("source_name"),
            }
    return {"source_kind": None, "source_name": None}


def _select_preferred_candidate(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> tuple[str, str, str, list[str]] | None:
    if len(candidate_ids) == 1:
        return (
            candidate_ids[0],
            "canonical_source_selected",
            "Selected one requested-basis Vietnamese financial statement.",
            [],
        )

    amended_supersession = _select_amended_supersession(package, candidate_ids, classifications_by_id)
    if amended_supersession is not None:
        return amended_supersession

    reviewed_supersession = _select_reviewed_supersession(package, candidate_ids, classifications_by_id)
    if reviewed_supersession is not None:
        return reviewed_supersession

    searchable_ids = [candidate_id for candidate_id in candidate_ids if _is_searchable_version(package, candidate_id)]
    default_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in searchable_ids]
    confirmed_searchable_ids = [
        candidate_id
        for candidate_id in searchable_ids
        if _identity_confirms_requested_filing(package, classifications_by_id[candidate_id])
    ]
    if len(confirmed_searchable_ids) == 1 and len(default_ids) <= 1:
        return (
            confirmed_searchable_ids[0],
            "searchable_version_selected",
            "Selected confirmed searchable filing version over default official source.",
            [],
        )
    if len(searchable_ids) == 1 and len(default_ids) == 1:
        return (
            default_ids[0],
            "searchable_identity_not_confirmed_fallback",
            "Searchable filing version identity was not confirmed; selected default official source.",
            [],
        )
    return None


def _select_amended_supersession(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> tuple[str, str, str, list[str]] | None:
    amended_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _filing_status(classifications_by_id[candidate_id]) == "amended_or_replacement"
    ]
    original_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _filing_status(classifications_by_id[candidate_id]) == "original"
    ]
    if len(amended_ids) != 1 or not original_ids:
        return None
    amended_id = amended_ids[0]
    amended = classifications_by_id[amended_id]
    if not all(_same_requested_identity(package, amended, classifications_by_id[other_id]) for other_id in original_ids):
        return None
    return (
        amended_id,
        "amended_full_fs_supersedes_original",
        "Selected amended/replacement full financial statement over original same-identity statement.",
        original_ids,
    )


def _select_reviewed_supersession(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> tuple[str, str, str, list[str]] | None:
    reviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) == "reviewed"
    ]
    unreviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) != "reviewed"
    ]
    if len(reviewed_ids) != 1 or not unreviewed_ids:
        return None
    reviewed_id = reviewed_ids[0]
    reviewed = classifications_by_id[reviewed_id]
    if not all(_same_requested_identity(package, reviewed, classifications_by_id[other_id]) for other_id in unreviewed_ids):
        return None
    if not all(_document_fetched_after(package, reviewed_id, other_id) for other_id in unreviewed_ids):
        return None
    return (
        reviewed_id,
        "reviewed_full_fs_supersedes_unreviewed",
        "Selected later reviewed full financial statement over earlier unreviewed same-identity statement.",
        unreviewed_ids,
    )


def _reviewed_supersession_order_blocker(
    package: FilingPackage,
    candidate_ids: list[str],
    classifications_by_id: dict[str, SourceDocumentClassification],
) -> str | None:
    reviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) == "reviewed"
    ]
    unreviewed_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if _assurance_type(classifications_by_id[candidate_id]) != "reviewed"
    ]
    if len(reviewed_ids) != 1 or not unreviewed_ids:
        return None
    reviewed_id = reviewed_ids[0]
    reviewed = classifications_by_id[reviewed_id]
    if not all(_same_requested_identity(package, reviewed, classifications_by_id[other_id]) for other_id in unreviewed_ids):
        return None
    if any(not _document_fetched_after(package, reviewed_id, other_id) for other_id in unreviewed_ids):
        return reviewed_id
    return None


def _inconclusive_reviewed_full_fs_identity_ids(
    package: FilingPackage,
    classifications: list[SourceDocumentClassification],
    candidate_ids: list[str],
) -> list[str]:
    if not candidate_ids:
        return []
    return [
        item.source_document_id
        for item in classifications
        if item.source_document_id not in candidate_ids
        and item.state != "classified"
        and item.document_role == "financial_statement"
        and item.language == "vi"
        and _basis_matches(package.report_basis, item.basis_family)
        and _assurance_type(item) == "reviewed"
    ]


def _is_requested_vietnamese_main_fs(
    package: FilingPackage,
    classification: SourceDocumentClassification,
) -> bool:
    return (
        classification.document_role == "financial_statement"
        and classification.is_full_financial_statement
        and classification.language in {"vi", "mixed"}
        and _basis_matches(package.report_basis, classification.basis_family)
    )


def _is_english_requested_basis_main_fs(
    package: FilingPackage,
    classification: SourceDocumentClassification,
) -> bool:
    return (
        classification.document_role == "financial_statement"
        and classification.is_full_financial_statement
        and classification.language == "en"
        and _basis_matches(package.report_basis, classification.basis_family)
    )


def _is_english_supporting_duplicate(
    package: FilingPackage,
    classification: SourceDocumentClassification,
    classifications: list[SourceDocumentClassification],
) -> bool:
    return (
        classification.language == "en"
        and (
            any(_is_requested_vietnamese_main_fs(package, other) for other in classifications)
            or any(
                other.source_document_id != classification.source_document_id
                and other.state == "classified"
                and other.is_supporting_document
                and other.document_role == classification.document_role
                and other.language == "vi"
                and _basis_matches(classification.basis_family, other.basis_family)
                for other in classifications
            )
        )
    )


def _basis_matches(requested_basis: str, classified_basis: str) -> bool:
    if requested_basis == classified_basis:
        return True
    if requested_basis in NON_CONSOLIDATED_BASIS_FAMILIES:
        return classified_basis in NON_CONSOLIDATED_BASIS_FAMILIES
    return False


def _is_searchable_version(package: FilingPackage, source_document_id: str) -> bool:
    for document in package.source_documents:
        if document.document_id != source_document_id:
            continue
        filename = _normalize_search_text((document.raw.get("vietstock_member") or {}).get("filename") or "")
        return "tra cuu" in filename or "searchable" in filename
    return False


def _identity_confirms_requested_filing(
    package: FilingPackage,
    classification: SourceDocumentClassification,
) -> bool:
    company_tokens = [
        (package.raw.get("company") or {}).get("ticker"),
        package.company_name,
    ]
    return (
        _any_evidence_matches(classification.company_evidence, company_tokens)
        and _any_evidence_matches(classification.period_evidence, [package.period])
        and _any_evidence_matches(classification.quarter_evidence, [package.raw.get("quarter")])
        and _basis_matches(package.report_basis, classification.basis_family)
        and classification.document_role == "financial_statement"
        and classification.is_full_financial_statement
        and _filing_status(classification) == (package.raw.get("filing_event") or {}).get("filing_status", "original")
    )


def _same_requested_identity(
    package: FilingPackage,
    left: SourceDocumentClassification,
    right: SourceDocumentClassification,
) -> bool:
    company_tokens = [(package.raw.get("company") or {}).get("ticker"), package.company_name]
    return (
        _any_evidence_matches(left.company_evidence, company_tokens)
        and _any_evidence_matches(right.company_evidence, company_tokens)
        and _any_evidence_matches(left.period_evidence, [package.period])
        and _any_evidence_matches(right.period_evidence, [package.period])
        and _any_evidence_matches(left.quarter_evidence, [package.raw.get("quarter")])
        and _any_evidence_matches(right.quarter_evidence, [package.raw.get("quarter")])
        and left.language == right.language == "vi"
        and _basis_matches(left.basis_family, right.basis_family)
        and _basis_matches(package.report_basis, left.basis_family)
        and left.document_role == right.document_role == "financial_statement"
        and left.is_full_financial_statement
        and right.is_full_financial_statement
    )


def _same_supporting_identity(
    package: FilingPackage,
    canonical: SourceDocumentClassification,
    supporting: SourceDocumentClassification,
) -> bool:
    company_tokens = [(package.raw.get("company") or {}).get("ticker"), package.company_name]
    return (
        _any_evidence_matches(canonical.company_evidence, company_tokens)
        and _any_evidence_matches(supporting.company_evidence, company_tokens)
        and _any_evidence_matches(canonical.period_evidence, [package.period])
        and _any_evidence_matches(supporting.period_evidence, [package.period])
        and _any_evidence_matches(canonical.quarter_evidence, [package.raw.get("quarter")])
        and _any_evidence_matches(supporting.quarter_evidence, [package.raw.get("quarter")])
        and canonical.language == supporting.language == "vi"
        and _basis_matches(package.report_basis, canonical.basis_family)
        and _basis_matches(package.report_basis, supporting.basis_family)
    )


def _document_fetched_after(package: FilingPackage, later_id: str, earlier_id: str) -> bool:
    fetched_at_by_id = {
        document.document_id: (document.provenance.get("fetched_at") or "")
        for document in package.source_documents
    }
    return fetched_at_by_id.get(later_id, "") > fetched_at_by_id.get(earlier_id, "")


def _any_evidence_matches(evidence_values: list[str], expected_values: list[str | None]) -> bool:
    evidence = " ".join(_normalize_search_text(value) for value in evidence_values)
    return any(
        expected and _normalize_search_text(expected) in evidence
        for expected in expected_values
    )


def _normalize_search_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", ascii_value).lower().split())


def _filing_status(classification: SourceDocumentClassification) -> str:
    if "amended_or_replacement" in classification.filing_status_hints:
        return "amended_or_replacement"
    if "amendment" in classification.filing_status_hints:
        return "amended_or_replacement"
    return "original"


def _assurance_type(classification: SourceDocumentClassification) -> str:
    normalized_hints = {_normalize_search_text(hint) for hint in classification.assurance_hints}
    if "reviewed" in normalized_hints or "soat xet" in normalized_hints or "soat_xet" in classification.assurance_hints:
        return "reviewed"
    if "audited" in normalized_hints or "kiem toan" in normalized_hints:
        return "audited"
    return "unaudited"
