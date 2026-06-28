from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol


ALLOWED_ROLES = {
    "financial_statement",
    "amendment_context_attachment",
    "cover_disclosure_letter",
    "cover_or_disclosure",
    "variance_explanation",
    "business_overview_out_of_scope",
    "business_overview",
    "out_of_scope",
    "unknown",
}
ALLOWED_LANGUAGES = {"vi", "en", "mixed", "unknown"}
ALLOWED_BASIS_FAMILIES = {"consolidated", "separate", "parent", "non_consolidated", "unknown"}
MIN_CLASSIFICATION_CONFIDENCE = 0.5
MAX_SNIPPET_CHARS = 160


@dataclass(frozen=True)
class SourceDocumentClassifierInput:
    vietstock_entry_metadata: dict[str, Any]
    source_document_metadata: dict[str, Any]
    filename: str
    first_few_page_evidence: dict[str, Any]


@dataclass(frozen=True)
class SourceDocumentClassification:
    source_document_id: str
    state: str
    document_role: str
    language: str
    basis_family: str
    visible_filing_label: str | None
    company_evidence: list[str]
    period_evidence: list[str]
    quarter_evidence: list[str]
    filing_status_hints: list[str]
    amendment_hints: list[str]
    assurance_hints: list[str]
    is_full_financial_statement: bool
    is_supporting_document: bool
    confidence: float
    visible_evidence_snippets: list[str]
    validation_errors: list[str]
    is_canonical_source_selection: bool = False
    approves_basis_switch: bool = False
    repairs_extraction: bool = False
    is_source_authority: bool = False


@dataclass(frozen=True)
class SourceDocumentClassificationResult:
    status: str
    classifications: list[SourceDocumentClassification]


class SourceDocumentClassifier(Protocol):
    def classify(self, classifier_input: SourceDocumentClassifierInput) -> dict[str, Any] | SourceDocumentClassification:
        ...


class DeterministicSourceDocumentClassifier:
    def classify(self, classifier_input: SourceDocumentClassifierInput) -> dict[str, Any]:
        evidence = classifier_input.first_few_page_evidence
        metadata = classifier_input.source_document_metadata
        text = _normalize(
            " ".join(
                [
                    classifier_input.filename,
                    str(classifier_input.vietstock_entry_metadata.get("title") or ""),
                    " ".join(evidence.get("visible_title_snippets") or []),
                    " ".join(evidence.get("role_clues") or []),
                ]
            )
        )
        role = _document_role(text, evidence)
        if role in {"out_of_scope", "cover_or_disclosure"} and metadata.get("document_type") == "main_financial_statement":
            role = "financial_statement"
        basis_family = _first_allowed(evidence.get("basis_clues") or metadata.get("basis_clues") or [], ALLOWED_BASIS_FAMILIES)
        language = _language(evidence, text)
        confidence = _confidence(role, evidence, text)
        snippets = _short_snippets(evidence.get("visible_title_snippets") or [])
        if not snippets and role == "financial_statement":
            snippets = _short_snippets([classifier_input.vietstock_entry_metadata.get("title")])
        company_evidence = _short_snippets(evidence.get("company_clues") or [])
        ticker = classifier_input.vietstock_entry_metadata.get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            ticker_evidence = ticker.strip().upper()
            if ticker_evidence not in company_evidence:
                company_evidence.append(ticker_evidence)
        period_evidence = _short_snippets(evidence.get("period_clues") or [])
        if role == "financial_statement":
            for clue in _short_snippets(metadata.get("reported_period_clues") or []):
                if clue not in period_evidence:
                    period_evidence.append(clue)

        return {
            "source_document_id": metadata["source_document_id"],
            "document_role": role,
            "language": language,
            "basis_family": basis_family,
            "visible_filing_label": snippets[0] if snippets else None,
            "company_evidence": company_evidence,
            "period_evidence": period_evidence,
            "quarter_evidence": [clue for clue in period_evidence if "Q" in str(clue).upper()][:3],
            "filing_status_hints": _filing_status_hints(text),
            "amendment_hints": _amendment_hints(text),
            "assurance_hints": _assurance_hints(text),
            "is_full_financial_statement": role == "financial_statement",
            "is_supporting_document": role in {
                "amendment_context_attachment",
                "cover_disclosure_letter",
                "cover_or_disclosure",
                "variance_explanation",
            },
            "confidence": confidence,
            "visible_evidence_snippets": snippets,
        }


def classify_filing_package_source_documents(
    package: Any,
    *,
    classifier: SourceDocumentClassifier,
) -> SourceDocumentClassificationResult:
    raw = package.raw if hasattr(package, "raw") else package
    evidence_by_id = {
        evidence.get("source_document_id"): evidence
        for evidence in raw.get("source_evidence", [])
        if isinstance(evidence, dict)
    }
    classifications = []
    for document in raw.get("source_documents", []):
        classifier_input = build_source_document_classifier_input(raw, document, evidence_by_id.get(document.get("document_id"), {}))
        try:
            output = classifier.classify(classifier_input)
        except Exception as exc:
            output = {"source_document_id": document.get("document_id"), "_validation_error": str(exc)}
        classifications.append(validate_source_document_classification(output))
    status = "completed" if all(item.state == "classified" for item in classifications) else "completed_with_inconclusive_classifications"
    return SourceDocumentClassificationResult(status=status, classifications=classifications)


def build_source_document_classifier_input(
    raw_package: dict[str, Any],
    document: dict[str, Any],
    evidence: dict[str, Any],
) -> SourceDocumentClassifierInput:
    source_document_id = str(document.get("document_id") or "")
    filename = _direct_or_member_filename(document)
    return SourceDocumentClassifierInput(
        vietstock_entry_metadata={
            "file_id": (document.get("vietstock_entry") or {}).get("file_id"),
            "title": (document.get("vietstock_entry") or {}).get("title"),
            "url": (document.get("vietstock_entry") or {}).get("url"),
            "ticker": (raw_package.get("company") or {}).get("ticker"),
            "company_name": (raw_package.get("company") or {}).get("name"),
            "period": raw_package.get("period"),
            "quarter": raw_package.get("quarter"),
            "report_basis": raw_package.get("report_basis"),
        },
        source_document_metadata={
            "source_document_id": source_document_id,
            "document_type": document.get("document_type"),
            "reported_period_clues": list(document.get("reported_period_clues") or []),
            "basis_clues": list(document.get("basis_clues") or []),
            "amendment_clues": {
                "is_amended_or_replacement": bool((document.get("amendment_clues") or {}).get("is_amended_or_replacement")),
                "clues": list((document.get("amendment_clues") or {}).get("clues") or []),
            },
            "file_type": document.get("file_type"),
        },
        filename=filename,
        first_few_page_evidence=_bounded_evidence(evidence, source_document_id=source_document_id),
    )


def validate_source_document_classification(raw: dict[str, Any] | SourceDocumentClassification) -> SourceDocumentClassification:
    if isinstance(raw, SourceDocumentClassification):
        return raw
    errors = []
    source_document_id = raw.get("source_document_id") if isinstance(raw, dict) else None
    if not isinstance(raw, dict):
        errors.append("classification output must be an object")
        raw = {}
    role = raw.get("document_role")
    language = raw.get("language")
    basis_family = raw.get("basis_family")
    confidence = raw.get("confidence")
    if not isinstance(source_document_id, str) or not source_document_id:
        errors.append("source_document_id is required")
        source_document_id = "unknown"
    if role not in ALLOWED_ROLES:
        errors.append("document_role is invalid")
        role = "unknown"
    if language not in ALLOWED_LANGUAGES:
        errors.append("language is invalid")
        language = "unknown"
    if basis_family not in ALLOWED_BASIS_FAMILIES:
        errors.append("basis_family is invalid")
        basis_family = "unknown"
    if not isinstance(confidence, (int, float)):
        errors.append("confidence is required")
        confidence = 0.0
    snippets = _short_snippets(raw.get("visible_evidence_snippets") if isinstance(raw.get("visible_evidence_snippets"), list) else [])
    state = "classified"
    if errors or float(confidence) < MIN_CLASSIFICATION_CONFIDENCE:
        state = "inconclusive"
    return SourceDocumentClassification(
        source_document_id=source_document_id,
        state=state,
        document_role=role,
        language=language,
        basis_family=basis_family,
        visible_filing_label=_short_optional(raw.get("visible_filing_label")),
        company_evidence=_short_string_list(raw.get("company_evidence")),
        period_evidence=_short_string_list(raw.get("period_evidence")),
        quarter_evidence=_short_string_list(raw.get("quarter_evidence")),
        filing_status_hints=_short_string_list(raw.get("filing_status_hints")),
        amendment_hints=_short_string_list(raw.get("amendment_hints")),
        assurance_hints=_short_string_list(raw.get("assurance_hints")),
        is_full_financial_statement=bool(raw.get("is_full_financial_statement")) and state == "classified",
        is_supporting_document=bool(raw.get("is_supporting_document")) and state == "classified",
        confidence=float(confidence),
        visible_evidence_snippets=snippets,
        validation_errors=errors,
    )


def _bounded_evidence(evidence: dict[str, Any], *, source_document_id: str) -> dict[str, Any]:
    return {
        "source_document_id": source_document_id,
        "status": evidence.get("status"),
        "inspected_page_span": evidence.get("inspected_page_span"),
        "visible_title_snippets": _short_snippets(evidence.get("visible_title_snippets") or []),
        "company_clues": _short_snippets(evidence.get("company_clues") or []),
        "period_clues": _short_snippets(evidence.get("period_clues") or []),
        "basis_clues": list(evidence.get("basis_clues") or []),
        "language_clues": list(evidence.get("language_clues") or []),
        "role_clues": list(evidence.get("role_clues") or []),
        "confidence": evidence.get("confidence"),
        "warnings": _short_snippets(evidence.get("warnings") or []),
    }


def _direct_or_member_filename(document: dict[str, Any]) -> str:
    member = document.get("vietstock_member") or {}
    if member.get("filename"):
        return str(member["filename"])
    entry_url = (document.get("vietstock_entry") or {}).get("url")
    return str(entry_url or "")


def _document_role(text: str, evidence: dict[str, Any]) -> str:
    role_clues = set(evidence.get("role_clues") or [])
    if "business_overview" in role_clues or "tong quan tinh hinh kinh doanh" in text:
        return "business_overview"
    if "disclosure" in role_clues or "cong bo thong tin" in text or "information disclosure" in text:
        return "cover_or_disclosure"
    if ("thuyet minh" in text or "dieu chinh" in text) and not _proves_full_replacement(text):
        return "amendment_context_attachment"
    if "variance_explanation" in role_clues or "giai trinh" in text or "explanation" in text:
        return "variance_explanation"
    if "financial_statement" in role_clues or "bao cao tai chinh" in text or "financial statement" in text or "bctc" in text:
        return "financial_statement"
    return "out_of_scope"


def _proves_full_replacement(text: str) -> bool:
    return "thay the toan bo" in text or "full replacement" in text


def _language(evidence: dict[str, Any], text: str) -> str:
    clues = evidence.get("language_clues") or []
    if "en" in clues and "vi" in clues:
        return "mixed"
    if "en" in clues:
        return "en"
    if "vi" in clues:
        return "vi"
    if "financial statements" in text:
        return "en"
    if "bao cao" in text or "cong bo" in text or "bctc" in text:
        return "vi"
    return "unknown"


def _confidence(role: str, evidence: dict[str, Any], text: str) -> float:
    metadata_supports_financial_statement = role == "financial_statement" and (
        "bao cao tai chinh" in text or "bctc" in text
    )
    if evidence.get("status") != "completed":
        if metadata_supports_financial_statement:
            return 0.6
        return 0.0
    evidence_confidence = evidence.get("confidence")
    if evidence_confidence == "none":
        if metadata_supports_financial_statement:
            return 0.6
        return 0.0
    if evidence_confidence == "low":
        return 0.45
    if role in {"financial_statement", "cover_or_disclosure", "business_overview", "amendment_context_attachment", "variance_explanation"}:
        return 0.86
    return 0.6


def _filing_status_hints(text: str) -> list[str]:
    if any(marker in text for marker in ["dieu chinh", "bo sung", "sua doi", "thay the", "adjusted", "amended"]):
        return ["amendment"]
    return ["original"]


def _amendment_hints(text: str) -> list[str]:
    return [marker for marker in ["dieu chinh", "bo sung", "sua doi", "thay the", "adjusted", "amended"] if marker in text][:4]


def _assurance_hints(text: str) -> list[str]:
    hints = []
    if "soat xet" in text or "reviewed" in text:
        hints.append("reviewed")
    if "kiem toan" in text or "audited" in text:
        hints.append("audited")
    return hints


def _first_allowed(values: list[Any], allowed: set[str]) -> str:
    for value in values:
        if isinstance(value, str) and value in allowed:
            return value
    return "unknown"


def _short_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _short_snippets(value)


def _short_optional(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _sanitize_snippet(value)


def _short_snippets(values: list[Any]) -> list[str]:
    snippets = []
    for value in values:
        if not isinstance(value, str):
            continue
        snippet = _sanitize_snippet(value)
        if snippet and snippet not in snippets:
            snippets.append(snippet)
    return snippets[:6]


def _sanitize_snippet(value: str) -> str:
    cleaned = re.sub(r"\b(?:x|y)=\d+(?:\.\d+)?\b", "", value)
    cleaned = re.sub(r"\bpage_number\s*=\s*\d+\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_SNIPPET_CHARS].strip()


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_value).lower()
