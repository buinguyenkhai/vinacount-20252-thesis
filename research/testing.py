from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vinacount.filing_package import FilingPackage, SourceDocument
from vinacount.raw_extraction_artifact import RawExtractionArtifact, validate_raw_extraction_artifact


@dataclass(frozen=True)
class FakeOcrExtractionAdapter:
    raw_html: str
    raw_tables: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    parser_warnings: list[dict[str, Any]] = field(default_factory=list)
    extraction_errors: list[dict[str, Any]] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "fake_nanonets_ocr_3_docstrange_html"
    extraction_version: str = "v1"

    def extract_source_document(
        self,
        *,
        filing_package: FilingPackage,
        source_document_id: str,
    ) -> RawExtractionArtifact:
        source_document = _find_source_document(filing_package, source_document_id)
        if source_document is None:
            raise ValueError("RawExtractionArtifact source_document_id must exist in FilingPackage")
        raw_artifact = {
            "artifact_id": f"RAW_EXT_{source_document_id}",
            "filing_package_id": filing_package.package_id,
            "source_document_id": source_document_id,
            "extraction_method": self.extraction_method,
            "extraction_version": self.extraction_version,
            "raw_html": self.raw_html,
            "raw_tables": self.raw_tables,
            "diagnostics": self.diagnostics,
            "parser_warnings": self.parser_warnings,
            "extraction_errors": self.extraction_errors,
            "provider_metadata": self.provider_metadata,
            "source_document_provenance": source_document.provenance,
            "source_local_artifact": source_document.raw["local_artifact"],
        }
        return validate_raw_extraction_artifact(raw_artifact, filing_package)


def _find_source_document(
    filing_package: FilingPackage,
    source_document_id: str,
) -> SourceDocument | None:
    for source_document in filing_package.source_documents:
        if source_document.document_id == source_document_id:
            return source_document
    return None
