from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vinacount.filing_package import FilingPackage


REQUIRED_RAW_EXTRACTION_ARTIFACT_FIELDS = {
    "artifact_id",
    "filing_package_id",
    "source_document_id",
    "extraction_method",
    "extraction_version",
}


@dataclass(frozen=True)
class RawExtractionArtifact:
    raw: dict[str, Any]

    @property
    def artifact_id(self) -> str:
        return self.raw["artifact_id"]

    @property
    def filing_package_id(self) -> str:
        return self.raw["filing_package_id"]

    @property
    def source_document_id(self) -> str:
        return self.raw["source_document_id"]

    @property
    def qa_records(self) -> list[dict[str, Any]]:
        return [
            {
                "record_id": warning["warning_id"],
                "record_type": "parser_warning",
                "message": warning["message"],
            }
            for warning in self.raw.get("parser_warnings", [])
        ] + [
            {
                "record_id": error["error_id"],
                "record_type": "extraction_error",
                "message": error["message"],
                "recoverable": error["recoverable"],
            }
            for error in self.raw.get("extraction_errors", [])
        ]


def validate_raw_extraction_artifact(
    raw: dict[str, Any],
    filing_package: FilingPackage,
) -> RawExtractionArtifact:
    _require_fields(
        raw,
        REQUIRED_RAW_EXTRACTION_ARTIFACT_FIELDS,
        "RawExtractionArtifact",
    )
    _require_non_empty_string(raw["artifact_id"], "RawExtractionArtifact artifact_id")
    _require_non_empty_string(
        raw["filing_package_id"],
        "RawExtractionArtifact filing_package_id",
    )
    _require_non_empty_string(
        raw["source_document_id"],
        "RawExtractionArtifact source_document_id",
    )
    _require_non_empty_string(
        raw["extraction_method"],
        "RawExtractionArtifact extraction_method",
    )
    _require_non_empty_string(
        raw["extraction_version"],
        "RawExtractionArtifact extraction_version",
    )
    if raw["filing_package_id"] != filing_package.package_id:
        raise ValueError("RawExtractionArtifact filing_package_id must match FilingPackage")

    source_document_ids = {document.document_id for document in filing_package.source_documents}
    if raw["source_document_id"] not in source_document_ids:
        raise ValueError("RawExtractionArtifact source_document_id must exist in FilingPackage")

    return RawExtractionArtifact(raw=raw)


def _require_fields(raw: dict[str, Any], required_fields: set[str], label: str) -> None:
    missing = required_fields - raw.keys()
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")


def _require_non_empty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
