from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vinacount.filing_package import FilingPackage, SourceDocument
from vinacount.raw_extraction_artifact import (
    RawExtractionArtifact,
    validate_raw_extraction_artifact,
)

RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION = "raw_extraction_artifact.v1"


@dataclass(frozen=True)
class NanonetsOcr3DocstrangeConfig:
    live_ocr_enabled: bool = False
    api_key: str | None = None
    timeout_seconds: int = 60
    max_retries: int = 2
    model: str = "ocr-3-docstrange"


@dataclass(frozen=True)
class NanonetsOcr3DocstrangeAdapter:
    config: NanonetsOcr3DocstrangeConfig = field(default_factory=NanonetsOcr3DocstrangeConfig)
    client: Any | None = None
    extraction_method: str = "nanonets_ocr_3_docstrange_html"
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
        if not self.config.live_ocr_enabled:
            return self._configuration_error_artifact(
                filing_package=filing_package,
                source_document=source_document,
                message="Nanonets OCR-3 / DocStrange live OCR is disabled.",
            )
        if not self.config.api_key:
            return self._configuration_error_artifact(
                filing_package=filing_package,
                source_document=source_document,
                message="Nanonets OCR-3 / DocStrange API key is missing.",
            )
        if self.client is None:
            return self._configuration_error_artifact(
                filing_package=filing_package,
                source_document=source_document,
                message="Nanonets OCR-3 / DocStrange provider client is missing.",
            )
        try:
            provider_response = self.client(
                api_key=self.config.api_key,
                pdf_path=source_document.raw["local_artifact"]["path"],
                output_format="html",
                include_metadata=["bounding_boxes", "confidence_score"],
                timeout_seconds=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
                model=self.config.model,
            )
        except Exception as exc:
            return self._provider_error_artifact(
                filing_package=filing_package,
                source_document=source_document,
                error_type=type(exc).__name__,
            )
        return self._successful_artifact(
            filing_package=filing_package,
            source_document=source_document,
            provider_response=provider_response,
        )

    def _configuration_error_artifact(
        self,
        *,
        filing_package: FilingPackage,
        source_document: SourceDocument,
        message: str,
    ) -> RawExtractionArtifact:
        source_document_id = source_document.document_id
        raw_artifact = {
            "artifact_id": f"RAW_EXT_{source_document_id}",
            "filing_package_id": filing_package.package_id,
            "source_document_id": source_document_id,
            "schema_version": RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION,
            "extraction_method": self.extraction_method,
            "extraction_version": self.extraction_version,
            "raw_html": "",
            "raw_tables": [],
            "diagnostics": [
                {
                    "provider": "nanonets_ocr_3_docstrange",
                    "model": self.config.model,
                    "timeout_seconds": self.config.timeout_seconds,
                    "max_retries": self.config.max_retries,
                    "live_ocr_enabled": self.config.live_ocr_enabled,
                }
            ],
            "parser_warnings": [],
            "extraction_errors": [
                {
                    "error_id": f"ERR_{source_document_id}_NANONETS_CONFIG",
                    "message": message,
                    "recoverable": True,
                }
            ],
            "provider_metadata": {
                "provider": "nanonets_ocr_3_docstrange",
                "model": self.config.model,
                "output_format": "html",
                "include_metadata": ["bounding_boxes", "confidence_score"],
                "status": "configuration_error",
            },
            "source_document_fingerprint": source_document.raw["fingerprint"],
            "source_document_provenance": source_document.provenance,
            "source_local_artifact": source_document.raw["local_artifact"],
        }
        return validate_raw_extraction_artifact(raw_artifact, filing_package)

    def _provider_error_artifact(
        self,
        *,
        filing_package: FilingPackage,
        source_document: SourceDocument,
        error_type: str,
    ) -> RawExtractionArtifact:
        source_document_id = source_document.document_id
        raw_artifact = {
            "artifact_id": f"RAW_EXT_{source_document_id}",
            "filing_package_id": filing_package.package_id,
            "source_document_id": source_document_id,
            "schema_version": RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION,
            "extraction_method": self.extraction_method,
            "extraction_version": self.extraction_version,
            "raw_html": "",
            "raw_tables": [],
            "diagnostics": [
                {
                    "provider": "nanonets_ocr_3_docstrange",
                    "model": self.config.model,
                    "timeout_seconds": self.config.timeout_seconds,
                    "max_retries": self.config.max_retries,
                }
            ],
            "parser_warnings": [],
            "extraction_errors": [
                {
                    "error_id": f"ERR_{source_document_id}_NANONETS_PROVIDER",
                    "message": f"Nanonets OCR-3 / DocStrange provider error: {error_type}",
                    "recoverable": True,
                }
            ],
            "provider_metadata": {
                "provider": "nanonets_ocr_3_docstrange",
                "model": self.config.model,
                "output_format": "html",
                "include_metadata": ["bounding_boxes", "confidence_score"],
                "status": "provider_error",
                "timeout_seconds": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
            },
            "source_document_fingerprint": source_document.raw["fingerprint"],
            "source_document_provenance": source_document.provenance,
            "source_local_artifact": source_document.raw["local_artifact"],
        }
        return validate_raw_extraction_artifact(raw_artifact, filing_package)

    def _successful_artifact(
        self,
        *,
        filing_package: FilingPackage,
        source_document: SourceDocument,
        provider_response: dict[str, Any],
    ) -> RawExtractionArtifact:
        source_document_id = source_document.document_id
        raw_artifact = {
            "artifact_id": f"RAW_EXT_{source_document_id}",
            "filing_package_id": filing_package.package_id,
            "source_document_id": source_document_id,
            "schema_version": RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION,
            "extraction_method": self.extraction_method,
            "extraction_version": self.extraction_version,
            "raw_html": provider_response.get("raw_html", ""),
            "raw_tables": provider_response.get("raw_tables", []),
            "diagnostics": provider_response.get("confidence_metadata", [])
            + provider_response.get("bounding_boxes", []),
            "parser_warnings": provider_response.get("parser_warnings", []),
            "extraction_errors": [],
            "provider_metadata": {
                "provider": "nanonets_ocr_3_docstrange",
                "model": self.config.model,
                "run_id": provider_response.get("run_id"),
                "output_format": "html",
                "include_metadata": ["bounding_boxes", "confidence_score"],
                "status": "completed",
                "timeout_seconds": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
            },
            "source_document_fingerprint": source_document.raw["fingerprint"],
            "source_document_provenance": source_document.provenance,
            "source_local_artifact": source_document.raw["local_artifact"],
        }
        if isinstance(provider_response.get("extraction_candidates"), dict):
            raw_artifact["extraction_candidates"] = provider_response["extraction_candidates"]
        return validate_raw_extraction_artifact(raw_artifact, filing_package)


def _find_source_document(
    filing_package: FilingPackage,
    source_document_id: str,
) -> SourceDocument | None:
    for source_document in filing_package.source_documents:
        if source_document.document_id == source_document_id:
            return source_document
    return None
