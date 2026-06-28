from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from vinacount.canonical_source_selection import (
    NON_CONSOLIDATED_BASIS_FAMILIES,
    CanonicalSourceSelectionResult,
    select_canonical_source_from_classifications,
)
from vinacount.filing_package import CanonicalSourceSelection, FilingPackage, SourceDocument
from vinacount.runtime_contract import HITL_BOUNDARY
from vinacount.source_document_classifier import (
    DeterministicSourceDocumentClassifier,
    SourceDocumentClassification,
    classify_filing_package_source_documents,
)
from vinacount.vietstock_fetched_source import (
    VietstockFetchedFilingPackageResult,
    fetch_vietstock_filing_package,
)


SessionFactory = Callable[[], Any]
Clock = Callable[[], datetime]


class VietstockRuntimeSourceDiscoveryAdapter:
    """Resolve the two Runtime source slots through the governed Vietstock path."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory | None = None,
        output_dir: Path | str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._session_factory = session_factory or requests.Session
        self._output_dir = Path(output_dir or "/tmp/vinacount_runtime_vietstock_sources")
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def discover(self, filing_intent: dict[str, Any]) -> dict[str, Any]:
        ticker = str(filing_intent["company_identifier"]).strip().upper()
        target_year = int(filing_intent["target_fiscal_year"])
        quarter = int(filing_intent["target_quarter"])
        public_basis = str(filing_intent["report_basis_preference"])
        fetched_at = self._clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        session = self._session_factory()

        slots = []
        package_warnings = []
        internal_packages = {}
        for role, fiscal_year in (
            ("target", target_year),
            ("prior_year_same_quarter", target_year - 1),
        ):
            slot, warnings, package = self._discover_slot(
                role=role,
                ticker=ticker,
                fiscal_year=fiscal_year,
                quarter=quarter,
                public_basis=public_basis,
                fetched_at=fetched_at,
                session=session,
            )
            slots.append(slot)
            package_warnings.extend(warnings)
            if package is not None:
                internal_packages[role] = copy.deepcopy(package.raw)

        confirmable = all(slot["status"] == "ready_for_review" for slot in slots)
        return {
            "status": "ready_for_review" if confirmable else "partially_rejected",
            "confirmable": confirmable,
            "hitl_boundary": HITL_BOUNDARY,
            "slots": slots,
            "package_warnings": _deduplicate_warnings(package_warnings),
            "_internal_live_extraction_source_packages": internal_packages,
        }

    def _discover_slot(
        self,
        *,
        role: str,
        ticker: str,
        fiscal_year: int,
        quarter: int,
        public_basis: str,
        fetched_at: str,
        session: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], FilingPackage | None]:
        try:
            fetched = fetch_vietstock_filing_package(
                ticker=ticker,
                company_name=ticker,
                year=fiscal_year,
                quarter=f"Q{quarter}",
                report_basis=public_basis,
                output_dir=self._output_dir,
                session=session,
                fetched_at=fetched_at,
            )
        except Exception as exc:
            return _unavailable_slot(role, "source_discovery_unavailable", _safe_error_message(exc)), [], None

        if fetched.package is None:
            code, message = _fetch_failure(fetched, ticker=ticker, fiscal_year=fiscal_year, quarter=quarter)
            slot = _unavailable_slot(role, code, message)
            return slot, copy.deepcopy(slot["warnings"]), None

        package = fetched.package
        classification_result = classify_filing_package_source_documents(
            package,
            classifier=DeterministicSourceDocumentClassifier(),
        )
        selection = select_canonical_source_from_classifications(
            package,
            classification_result.classifications,
        )
        if selection.status == "hitl_needed" and selection.code in {
            "ambiguous_financial_statement_candidates",
            "ambiguous_reviewed_supersession_order",
        }:
            candidate_documents = _runtime_candidate_documents(
                package,
                selection,
                classification_result.classifications,
                ticker=ticker,
                fiscal_year=fiscal_year,
                quarter=quarter,
                public_basis=public_basis,
            )
            code = selection.code or "source_identity_ambiguity"
            slot = _unavailable_slot(
                role,
                code,
                _selection_message(code, ticker=ticker, fiscal_year=fiscal_year, quarter=quarter),
                candidate_documents=candidate_documents,
            )
            return (
                slot,
                [*copy.deepcopy(slot["warnings"]), *_record_warnings(fetched, role)],
                package,
            )
        selection_failure = _selection_failure(
            package,
            selection,
            classification_result.classifications,
            ticker=ticker,
            fiscal_year=fiscal_year,
            quarter=quarter,
            public_basis=public_basis,
        )
        if selection_failure is not None:
            code, message = selection_failure
            slot = _unavailable_slot(role, code, message)
            return slot, [*copy.deepcopy(slot["warnings"]), *_record_warnings(fetched, role)], None

        candidate = _runtime_candidate(
            package,
            selection,
            classification_result.classifications,
            ticker=ticker,
            fiscal_year=fiscal_year,
            quarter=quarter,
            public_basis=public_basis,
        )
        warnings = [
            *_record_warnings(fetched, role),
            *_selection_warnings(selection, role),
        ]
        if candidate["audit_references"].get("source_container_type") == "zip":
            warnings.append(
                _warning(
                    "vietstock_package_member_selected",
                    "Nguồn Vietstock là gói ZIP; tài liệu chuẩn là thành viên PDF đã được chọn.",
                    role=role,
                    severity="info",
                )
            )
        return {
            "role": role,
            "status": "ready_for_review",
            "candidate": candidate,
            "candidate_documents": [],
            "rejection": None,
            "warnings": warnings,
        }, warnings, package


def _selection_failure(
    package: FilingPackage,
    selection: CanonicalSourceSelectionResult,
    classifications: list[SourceDocumentClassification],
    *,
    ticker: str,
    fiscal_year: int,
    quarter: int,
    public_basis: str,
) -> tuple[str, str] | None:
    if selection.status != "selected" or selection.selection is None:
        if any(
            item.state == "classified"
            and item.document_role == "financial_statement"
            and item.is_full_financial_statement
            and item.language == "vi"
            and not _basis_matches(public_basis, item.basis_family)
            for item in classifications
        ):
            return "wrong_basis", "Cơ sở lập báo cáo hiển thị không khớp yêu cầu."
        code = selection.code or "source_package_unavailable"
        return code, _selection_message(code, ticker=ticker, fiscal_year=fiscal_year, quarter=quarter)

    canonical_id = selection.selection.canonical_source_document_id
    canonical_document = next(
        (item for item in package.source_documents if item.document_id == canonical_id),
        None,
    )
    classification = next(
        (item for item in classifications if item.source_document_id == canonical_id),
        None,
    )
    if canonical_document is None or classification is None or classification.state != "classified":
        return "source_identity_inconclusive", "Không thể xác minh danh tính tài liệu nguồn đã chọn."
    if not _company_matches(canonical_document, classification.company_evidence, ticker):
        return "wrong_company", "Tên công ty hiển thị trong tài liệu không khớp mã công ty yêu cầu."
    if not _period_matches(classification.period_evidence, fiscal_year=fiscal_year, quarter=quarter):
        return "wrong_period", "Kỳ báo cáo hiển thị trong tài liệu không khớp kỳ yêu cầu."
    if not _basis_matches(public_basis, classification.basis_family):
        return "wrong_basis", "Cơ sở lập báo cáo hiển thị không khớp yêu cầu."
    if classification.language not in {"vi", "mixed"}:
        return "wrong_language", "Tài liệu chuẩn không phải báo cáo tài chính tiếng Việt."
    if classification.document_role != "financial_statement" or not classification.is_full_financial_statement:
        return (
            "not_full_financial_statement",
            "Tài liệu được chọn không phải báo cáo tài chính đầy đủ.",
        )
    return None


def _runtime_candidate(
    package: FilingPackage,
    selection: CanonicalSourceSelectionResult,
    classifications: list[SourceDocumentClassification],
    *,
    ticker: str,
    fiscal_year: int,
    quarter: int,
    public_basis: str,
) -> dict[str, Any]:
    selected = selection.selection
    assert selected is not None
    canonical_id = selected.canonical_source_document_id
    document = next(item for item in package.source_documents if item.document_id == canonical_id)
    classification = next(item for item in classifications if item.source_document_id == canonical_id)
    raw = document.raw
    member = raw.get("vietstock_member") or {}
    container = raw.get("vietstock_container") or {}
    source_name = str(document.provenance.get("source_name") or "Vietstock")
    visible_company = next(
        (value for value in classification.company_evidence if _normalize(value) == _normalize(ticker)),
        classification.company_evidence[0] if classification.company_evidence else ticker,
    )
    visible_period = _visible_period(classification, fiscal_year=fiscal_year, quarter=quarter)
    visible_basis = _visible_basis(classification.basis_family)
    visible_label = classification.visible_filing_label or (
        f"Báo cáo tài chính {visible_basis} quý {quarter} năm {fiscal_year}"
    )
    artifact_path = Path(raw["local_artifact"]["path"])
    evidence = list(classification.visible_evidence_snippets)
    evidence.extend(
        [
            f"Định danh công ty: {visible_company}",
            f"Định danh kỳ báo cáo: {visible_period}",
            f"Cơ sở lập báo cáo: {visible_basis}",
        ]
    )
    audit_references = {
        "package_id": package.package_id,
        "event_id": (package.raw.get("filing_event") or {}).get("event_id"),
        "canonical_source_document_id": canonical_id,
        "source_document_fingerprint_sha256": raw["fingerprint"]["hash_value"],
        "source_identity_check_status": "ready",
        "filing_visible_report_basis": classification.basis_family,
        "public_report_basis": public_basis,
        "selected_package_member_filename": member.get("filename"),
        "selected_package_member_file_size_bytes": member.get("file_size_bytes"),
        "source_container_type": container.get("file_type"),
        "source_container_file_id": container.get("file_id"),
        "source_container_fingerprint_sha256": container.get("hash_value"),
    }
    return {
        "source_document_id": canonical_id,
        "company_name_vi": visible_company,
        "ticker": ticker,
        "period_label": f"Q{quarter} {fiscal_year}",
        "quarter": quarter,
        "fiscal_year": fiscal_year,
        "report_basis": public_basis,
        "filing_status": selected.filing_status,
        "document_type": "quarterly_bctc",
        "language": classification.language,
        "source_origin": "Vietstock",
        "source_name": "Vietstock" if source_name.lower() == "vietstock" else source_name,
        "source_url": str(document.provenance["source_url"]),
        "is_searchable_version": _is_searchable_member(member.get("filename")),
        "file_size_bytes": artifact_path.stat().st_size,
        "page_count": 0,
        "visible_filing_label": visible_label,
        "first_page_identity": {
            "visible_company_name": visible_company,
            "visible_period": visible_period,
            "visible_basis_clue": visible_basis,
        },
        "classification_evidence": _deduplicate_strings(evidence),
        "audit_references": audit_references,
    }


def _runtime_candidate_documents(
    package: FilingPackage,
    selection: CanonicalSourceSelectionResult,
    classifications: list[SourceDocumentClassification],
    *,
    ticker: str,
    fiscal_year: int,
    quarter: int,
    public_basis: str,
) -> list[dict[str, Any]]:
    candidate_ids = _hitl_candidate_document_ids(
        package,
        classifications,
        ticker=ticker,
        fiscal_year=fiscal_year,
        quarter=quarter,
        public_basis=public_basis,
    )
    if (
        not candidate_ids
        and selection.code == "ambiguous_reviewed_supersession_order"
        and selection.records
    ):
        candidate_ids = [
            str(record.source_document_id)
            for record in selection.records
            if record.source_document_id
        ]
    return [
        _runtime_candidate_for_document_id(
            package,
            classifications,
            source_document_id=source_document_id,
            ticker=ticker,
            fiscal_year=fiscal_year,
            quarter=quarter,
            public_basis=public_basis,
        )
        for source_document_id in candidate_ids
    ]


def _hitl_candidate_document_ids(
    package: FilingPackage,
    classifications: list[SourceDocumentClassification],
    *,
    ticker: str,
    fiscal_year: int,
    quarter: int,
    public_basis: str,
) -> list[str]:
    documents_by_id = {document.document_id: document for document in package.source_documents}
    candidate_ids = []
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
        if not _basis_matches(public_basis, classification.basis_family):
            continue
        if not _company_matches(document, classification.company_evidence, ticker):
            continue
        if not _period_matches(classification.period_evidence, fiscal_year=fiscal_year, quarter=quarter):
            continue
        candidate_ids.append(classification.source_document_id)
    return candidate_ids


def _runtime_candidate_for_document_id(
    package: FilingPackage,
    classifications: list[SourceDocumentClassification],
    *,
    source_document_id: str,
    ticker: str,
    fiscal_year: int,
    quarter: int,
    public_basis: str,
) -> dict[str, Any]:
    classifications_by_id = {item.source_document_id: item for item in classifications}
    classification = classifications_by_id[source_document_id]
    selection = CanonicalSourceSelectionResult(
        status="selected",
        code=None,
        selection=CanonicalSourceSelection(
            canonical_source_document_id=source_document_id,
            filing_status=_filing_status_for_candidate(package, classification),
            supporting_document_ids=[
                document.document_id for document in package.supporting_documents
            ],
            requested_report_basis=public_basis,
            selected_basis_family=classification.basis_family,
            visible_filing_label_by_document_id={
                item.source_document_id: item.visible_filing_label
                for item in classifications
                if item.visible_filing_label
            },
            assurance_type=_assurance_type_for_candidate(classification),
            superseded_source_document_ids=[],
            amendment_context_attachment_ids=[],
            supporting_document_links=[],
        ),
        records=[],
        ignored_document_ids=[],
    )
    return _runtime_candidate(
        package,
        selection,
        classifications,
        ticker=ticker,
        fiscal_year=fiscal_year,
        quarter=quarter,
        public_basis=public_basis,
    )


def _filing_status_for_candidate(
    package: FilingPackage,
    classification: SourceDocumentClassification,
) -> str:
    document = next(
        item for item in package.source_documents if item.document_id == classification.source_document_id
    )
    text = _normalize(" ".join([*classification.filing_status_hints, *classification.amendment_hints]))
    if document.document_type == "amended_or_replacement_financial_statement" or document.is_amended_or_replacement:
        return "amended_or_replacement"
    if "soat xet" in text or "reviewed" in text:
        return "reviewed"
    return "original"


def _assurance_type_for_candidate(classification: SourceDocumentClassification) -> str:
    text = _normalize(" ".join(classification.assurance_hints))
    if "kiem toan" in text or "audited" in text:
        return "audited"
    if "soat xet" in text or "reviewed" in text:
        return "reviewed"
    return "unaudited"


def _fetch_failure(
    fetched: VietstockFetchedFilingPackageResult,
    *,
    ticker: str,
    fiscal_year: int,
    quarter: int,
) -> tuple[str, str]:
    if fetched.records:
        record = fetched.records[0]
        return record.code, _safe_error_message(record.message)
    return (
        "source_package_unavailable",
        f"Không tìm thấy báo cáo Vietstock cho {ticker} Q{quarter} {fiscal_year}.",
    )


def _selection_message(code: str, *, ticker: str, fiscal_year: int, quarter: int) -> str:
    messages = {
        "ambiguous_financial_statement_candidates": (
            "Có nhiều tài liệu báo cáo tài chính cùng danh tính nhưng chưa thể chọn an toàn."
        ),
        "ambiguous_reviewed_supersession_order": (
            "Không thể xác định an toàn thứ tự thay thế giữa các báo cáo cùng danh tính."
        ),
        "missing_requested_basis_vietnamese_financial_statement": (
            "Không có báo cáo tài chính tiếng Việt đúng cơ sở lập báo cáo."
        ),
        "searchable_identity_not_confirmed": "Không thể xác minh danh tính của bản báo cáo tra cứu.",
        "inconclusive_reviewed_full_fs_identity": (
            "Danh tính của báo cáo soát xét có khả năng thay thế chưa rõ ràng."
        ),
    }
    return messages.get(code, f"Không thể chọn nguồn Vietstock an toàn cho {ticker} Q{quarter} {fiscal_year}.")


def _unavailable_slot(
    role: str,
    code: str,
    message: str,
    *,
    candidate_documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "status": "unavailable",
        "candidate": None,
        "candidate_documents": candidate_documents or [],
        "rejection": None,
        "warnings": [_warning(code, message, role=role)],
    }


def _record_warnings(
    fetched: VietstockFetchedFilingPackageResult,
    role: str,
) -> list[dict[str, Any]]:
    return [
        _warning(record.code, _safe_error_message(record.message), role=role)
        for record in fetched.records
        if record.severity != "info"
    ]


def _selection_warnings(
    selection: CanonicalSourceSelectionResult,
    role: str,
) -> list[dict[str, Any]]:
    messages = {
        "corrected_value_resolution_required": (
            "Nguồn có tài liệu điều chỉnh liên quan; bước trích xuất sau xác nhận "
            "phải giải quyết giá trị hiệu lực trước khi phân tích."
        ),
    }
    return [
        _warning(
            record.code,
            messages.get(record.code, _safe_error_message(record.message)),
            role=role,
        )
        for record in selection.records
        if record.severity in {"warning", "hitl_needed", "recoverable_data_error", "error"}
    ]


def _warning(
    code: str,
    message: str,
    *,
    role: str,
    severity: str = "warning",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "stage_id": "source_discovery",
        "source_slot_role": role,
        "artifact_refs": [],
    }


def _company_matches(document: SourceDocument, evidence: list[str], ticker: str) -> bool:
    normalized_ticker = _normalize(ticker)
    entry = document.raw.get("vietstock_entry") or {}
    member = document.raw.get("vietstock_member") or {}
    metadata_identity_text = " ".join(
        str(value or "")
        for value in (
            entry.get("title"),
            entry.get("url"),
            document.provenance.get("source_url"),
            member.get("filename"),
        )
    )
    if not re.search(
        rf"(?:^|\s){re.escape(normalized_ticker)}(?:$|\s)",
        _normalize(metadata_identity_text),
    ):
        return False
    explicit_tickers = {
        match.group(1)
        for value in evidence
        for match in re.finditer(r"\b(?:Cong ty|Công ty)\s+([A-Z]{2,6})\b", value)
    }
    explicit_tickers -= {"CO", "CP", "CTCP", "TNHH", "TMCP"}
    if explicit_tickers and ticker.upper() not in explicit_tickers:
        return False
    if any(
        re.search(rf"(?:^|\s){re.escape(normalized_ticker)}(?:$|\s)", _normalize(value))
        for value in evidence
    ):
        return True
    return bool(evidence) and not explicit_tickers


def _period_matches(evidence: list[str], *, fiscal_year: int, quarter: int) -> bool:
    normalized = " ".join(_normalize(value) for value in evidence)
    visible_pairs = {
        (int(year), int(q))
        for q, year in re.findall(r"(?:quy|q)\s*([1-4])\D{0,20}(20\d{2})", normalized)
    }
    visible_pairs.update(
        (int(year), int(q))
        for year, q in re.findall(r"(20\d{2})\s*q([1-4])", normalized)
    )
    if any(year == fiscal_year and visible_quarter != quarter for year, visible_quarter in visible_pairs):
        return False
    return (
        f"{fiscal_year} q{quarter}" in normalized
        or f"q{quarter} {fiscal_year}" in normalized
        or f"quy {quarter} nam {fiscal_year}" in normalized
    )


def _basis_matches(public_basis: str, classified_basis: str) -> bool:
    if public_basis == classified_basis:
        return True
    return (
        public_basis in NON_CONSOLIDATED_BASIS_FAMILIES
        and classified_basis in NON_CONSOLIDATED_BASIS_FAMILIES
    )


def _visible_period(
    classification: SourceDocumentClassification,
    *,
    fiscal_year: int,
    quarter: int,
) -> str:
    for clue in classification.period_evidence:
        if _period_matches([clue], fiscal_year=fiscal_year, quarter=quarter):
            return clue
    return f"Quý {quarter} năm {fiscal_year}"


def _visible_basis(basis: str) -> str:
    if basis == "consolidated":
        return "hợp nhất"
    if basis == "parent":
        return "công ty mẹ"
    return "riêng lẻ"


def _is_searchable_member(filename: Any) -> bool:
    normalized = _normalize(str(filename or ""))
    return "tra cuu" in normalized or "searchable" in normalized


def _safe_error_message(message: Any) -> str:
    text = re.sub(r"\s+", " ", str(message)).strip()
    text = re.sub(r"(?:[A-Za-z]:\\|/)[^\s]+", "[internal location]", text)
    return text[:240] or "Không thể hoàn tất khám phá nguồn Vietstock."


def _deduplicate_strings(values: list[str]) -> list[str]:
    result = []
    for value in values:
        value = re.sub(r"\s+", " ", str(value)).strip()
        if value and value not in result:
            result.append(value)
    return result or ["Đã xác minh định danh tài liệu nguồn Vietstock."]


def _deduplicate_warnings(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for warning in warnings:
        key = (warning["code"], warning.get("source_slot_role"), warning["message"])
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", ascii_value).lower().split())
