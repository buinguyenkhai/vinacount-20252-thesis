from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALLOWED_REPORT_BASES = {"consolidated", "separate", "parent"}
ALLOWED_DOCUMENT_TYPES = {
    "main_financial_statement",
    "variance_explanation",
    "variance_explanation_attachment",
    "cover_disclosure_letter",
    "cover_letter",
    "amendment_context_attachment",
    "business_overview_out_of_scope",
    "amended_or_replacement_financial_statement",
}
ALLOWED_SOURCE_KINDS = {"local", "fetched"}

REQUIRED_PACKAGE_FIELDS = {
    "package_id",
    "company",
    "period",
    "report_basis",
    "filing_event",
    "source_documents",
}
REQUIRED_COMPANY_FIELDS = {"name"}
REQUIRED_FILING_EVENT_FIELDS = {"event_id", "filing_status"}
REQUIRED_SOURCE_DOCUMENT_FIELDS = {
    "document_id",
    "document_type",
    "provenance",
    "local_artifact",
    "fingerprint",
    "reported_period_clues",
    "basis_clues",
    "amendment_clues",
}
REQUIRED_PROVENANCE_FIELDS = {"source_kind", "source_name", "source_url", "fetched_at"}
REQUIRED_LOCAL_ARTIFACT_FIELDS = {"artifact_id", "path"}
REQUIRED_FINGERPRINT_FIELDS = {"hash_algorithm", "hash_value"}
REQUIRED_AMENDMENT_CLUE_FIELDS = {"is_amended_or_replacement", "clues"}


@dataclass(frozen=True)
class SourceDocument:
    raw: dict[str, Any]

    @property
    def document_id(self) -> str:
        return self.raw["document_id"]

    @property
    def document_type(self) -> str:
        return self.raw["document_type"]

    @property
    def provenance(self) -> dict[str, Any]:
        return self.raw["provenance"]

    @property
    def reported_period_clues(self) -> list[Any]:
        return self.raw["reported_period_clues"]

    @property
    def basis_clues(self) -> list[Any]:
        return self.raw["basis_clues"]

    @property
    def is_amended_or_replacement(self) -> bool:
        return self.raw["amendment_clues"]["is_amended_or_replacement"]


@dataclass(frozen=True)
class LocalSourceDocument:
    document_id: str
    document_type: str
    artifact_id: str
    path: str
    hash_value: str
    reported_period_clues: list[str]
    basis_clues: list[str]
    hash_algorithm: str = "sha256"
    source_name: str = "local"
    is_amended_or_replacement: bool = False
    amendment_clues: list[str] | None = None


@dataclass(frozen=True)
class FilingPackage:
    raw: dict[str, Any]
    source_documents: list[SourceDocument]

    @property
    def package_id(self) -> str:
        return self.raw["package_id"]

    @property
    def company_name(self) -> str:
        return self.raw["company"]["name"]

    @property
    def period(self) -> str:
        return self.raw["period"]

    @property
    def report_basis(self) -> str:
        return self.raw["report_basis"]

    @property
    def financial_statement_candidates(self) -> list[SourceDocument]:
        return [
            document
            for document in self.source_documents
            if document.document_type
            in {"main_financial_statement", "amended_or_replacement_financial_statement"}
        ]

    @property
    def supporting_documents(self) -> list[SourceDocument]:
        return [
            document
            for document in self.source_documents
            if document.document_type in {
                "variance_explanation",
                "variance_explanation_attachment",
                "cover_disclosure_letter",
                "cover_letter",
                "amendment_context_attachment",
            }
        ]

    @property
    def source_evidence(self) -> list[dict[str, Any]]:
        return self.raw.get("source_evidence", [])


@dataclass(frozen=True)
class CanonicalSourceSelection:
    canonical_source_document_id: str
    filing_status: str
    supporting_document_ids: list[str]
    requested_report_basis: str | None = None
    selected_basis_family: str | None = None
    visible_filing_label_by_document_id: dict[str, str] | None = None
    assurance_type: str | None = None
    superseded_source_document_ids: list[str] | None = None
    amendment_context_attachment_ids: list[str] | None = None
    supporting_document_links: list[dict[str, Any]] | None = None


class CanonicalSourceSelectionError(ValueError):
    pass


def ingest_local_filing_package(
    *,
    package_id: str,
    company: dict[str, Any],
    period: str,
    quarter: str,
    report_basis: str,
    filing_event: dict[str, Any],
    source_documents: list[LocalSourceDocument],
) -> FilingPackage:
    _require_non_empty_string(quarter, "FilingPackage quarter")
    raw = {
        "package_id": package_id,
        "company": company,
        "period": period,
        "quarter": quarter,
        "report_basis": report_basis,
        "filing_event": filing_event,
        "source_documents": [
            _local_source_document_to_raw(document) for document in source_documents
        ],
    }
    return validate_filing_package(raw)


def _local_source_document_to_raw(document: LocalSourceDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "document_type": document.document_type,
        "provenance": {
            "source_kind": "local",
            "source_name": document.source_name,
            "source_url": None,
            "fetched_at": None,
        },
        "local_artifact": {
            "artifact_id": document.artifact_id,
            "path": document.path,
        },
        "fingerprint": {
            "hash_algorithm": document.hash_algorithm,
            "hash_value": document.hash_value,
        },
        "reported_period_clues": document.reported_period_clues,
        "basis_clues": document.basis_clues,
        "amendment_clues": {
            "is_amended_or_replacement": document.is_amended_or_replacement,
            "clues": document.amendment_clues or [],
        },
    }


def select_canonical_source(package: FilingPackage) -> CanonicalSourceSelection:
    financial_statement_candidates = _exact_financial_statement_candidates(package)
    if not financial_statement_candidates:
        raise CanonicalSourceSelectionError(_missing_candidate_reason(package))

    amended_candidates = [
        document
        for document in financial_statement_candidates
        if document.document_type == "amended_or_replacement_financial_statement"
        and document.is_amended_or_replacement
    ]
    original_candidates = [
        document
        for document in financial_statement_candidates
        if document.document_type == "main_financial_statement"
    ]

    if amended_candidates:
        if len(amended_candidates) != 1:
            raise CanonicalSourceSelectionError("ambiguous_financial_statement_candidates")
        canonical_document = amended_candidates[0]
        filing_status = "amended_or_replacement"
    else:
        if len(original_candidates) != 1:
            raise CanonicalSourceSelectionError("ambiguous_financial_statement_candidates")
        canonical_document = original_candidates[0]
        filing_status = "original"

    return CanonicalSourceSelection(
        canonical_source_document_id=canonical_document.document_id,
        filing_status=filing_status,
        supporting_document_ids=[document.document_id for document in package.supporting_documents],
        requested_report_basis=package.report_basis,
    )


def _exact_financial_statement_candidates(package: FilingPackage) -> list[SourceDocument]:
    return [
        document
        for document in package.financial_statement_candidates
        if package.period in document.reported_period_clues
        and package.report_basis in document.basis_clues
    ]


def _missing_candidate_reason(package: FilingPackage) -> str:
    if not package.financial_statement_candidates:
        return "missing_financial_statement"
    if any(package.period in document.reported_period_clues for document in package.financial_statement_candidates):
        return "basis_mismatch"
    return "period_mismatch"


def validate_filing_package(raw: dict[str, Any]) -> FilingPackage:
    _require_fields(raw, REQUIRED_PACKAGE_FIELDS, "FilingPackage")

    company = raw["company"]
    if not isinstance(company, dict):
        raise ValueError("FilingPackage company must be an object")
    _require_fields(company, REQUIRED_COMPANY_FIELDS, "FilingPackage company")
    _require_non_empty_string(company["name"], "FilingPackage company.name")

    _require_non_empty_string(raw["package_id"], "FilingPackage package_id")
    _require_non_empty_string(raw["period"], "FilingPackage period")
    if raw["report_basis"] not in ALLOWED_REPORT_BASES:
        raise ValueError("FilingPackage report_basis must be consolidated, separate, or parent")

    filing_event = raw["filing_event"]
    if not isinstance(filing_event, dict):
        raise ValueError("FilingPackage filing_event must be an object")
    _require_fields(filing_event, REQUIRED_FILING_EVENT_FIELDS, "FilingPackage filing_event")
    _require_non_empty_string(filing_event["event_id"], "FilingPackage filing_event.event_id")
    _require_non_empty_string(filing_event["filing_status"], "FilingPackage filing_event.filing_status")

    source_documents = raw["source_documents"]
    if not isinstance(source_documents, list) or not source_documents:
        raise ValueError("FilingPackage source_documents must be a non-empty list")

    validated_documents = [_validate_source_document(document) for document in source_documents]
    document_ids = [document.document_id for document in validated_documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("FilingPackage source_documents contain a duplicate source document ID")
    return FilingPackage(raw=raw, source_documents=validated_documents)


def _validate_source_document(raw: dict[str, Any]) -> SourceDocument:
    if not isinstance(raw, dict):
        raise ValueError("FilingPackage source document must be an object")
    _require_fields(raw, REQUIRED_SOURCE_DOCUMENT_FIELDS, "FilingPackage source document")
    _require_non_empty_string(raw["document_id"], "FilingPackage source document document_id")
    if raw["document_type"] not in ALLOWED_DOCUMENT_TYPES:
        raise ValueError("FilingPackage source document has unsupported document_type")

    provenance = raw["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("FilingPackage source document provenance must be an object")
    _require_fields(provenance, REQUIRED_PROVENANCE_FIELDS, "FilingPackage source document provenance")
    if provenance["source_kind"] not in ALLOWED_SOURCE_KINDS:
        raise ValueError("FilingPackage source document provenance source_kind must be local or fetched")
    _require_non_empty_string(provenance["source_name"], "FilingPackage source document provenance.source_name")
    if provenance["source_kind"] == "fetched":
        _require_non_empty_string(provenance["source_url"], "FilingPackage fetched provenance source_url")
        _require_non_empty_string(provenance["fetched_at"], "FilingPackage fetched provenance fetched_at")

    local_artifact = raw["local_artifact"]
    if not isinstance(local_artifact, dict):
        raise ValueError("FilingPackage source document local_artifact must be an object")
    _require_fields(local_artifact, REQUIRED_LOCAL_ARTIFACT_FIELDS, "FilingPackage source document local_artifact")
    _require_non_empty_string(local_artifact["artifact_id"], "FilingPackage source document local_artifact.artifact_id")
    _require_non_empty_string(local_artifact["path"], "FilingPackage source document local_artifact.path")

    fingerprint = raw["fingerprint"]
    if not isinstance(fingerprint, dict):
        raise ValueError("FilingPackage source document fingerprint must be an object")
    _require_fields(fingerprint, REQUIRED_FINGERPRINT_FIELDS, "FilingPackage source document fingerprint")
    _require_non_empty_string(fingerprint["hash_algorithm"], "FilingPackage source document fingerprint.hash_algorithm")
    _require_non_empty_string(fingerprint["hash_value"], "FilingPackage source document fingerprint.hash_value")

    if not isinstance(raw["reported_period_clues"], list):
        raise ValueError("FilingPackage source document reported_period_clues must be a list")
    if not isinstance(raw["basis_clues"], list):
        raise ValueError("FilingPackage source document basis_clues must be a list")

    amendment_clues = raw["amendment_clues"]
    if not isinstance(amendment_clues, dict):
        raise ValueError("FilingPackage source document amendment_clues must be an object")
    _require_fields(amendment_clues, REQUIRED_AMENDMENT_CLUE_FIELDS, "FilingPackage source document amendment_clues")
    if not isinstance(amendment_clues["is_amended_or_replacement"], bool):
        raise ValueError("FilingPackage source document amendment flag must be boolean")
    if not isinstance(amendment_clues["clues"], list):
        raise ValueError("FilingPackage source document amendment clues must be a list")

    return SourceDocument(raw=raw)


def _require_fields(raw: dict[str, Any], required_fields: set[str], label: str) -> None:
    missing = required_fields - raw.keys()
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _require_non_empty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
