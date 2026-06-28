from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceEvidenceRecord:
    severity: str
    code: str
    message: str


def extract_source_evidence_for_documents(
    source_documents: list[dict[str, Any]],
    *,
    max_pages: int = 5,
) -> tuple[list[dict[str, Any]], list[SourceEvidenceRecord]]:
    evidence = []
    records = []
    for document in source_documents:
        artifact_path = Path(document["local_artifact"]["path"])
        try:
            content = artifact_path.read_bytes()
            artifact = _extract_document_evidence(document, content, max_pages=max_pages)
            evidence.append(artifact)
            if artifact["status"] != "completed":
                records.append(
                    SourceEvidenceRecord(
                        severity="recoverable_data_error",
                        code="source_quality_evidence_inconclusive",
                        message=f"Source evidence extraction was inconclusive for {document['document_id']}",
                    )
                )
        except Exception as exc:
            evidence.append(_failed_evidence(document["document_id"], max_pages=max_pages, warning=str(exc)))
            records.append(
                SourceEvidenceRecord(
                    severity="recoverable_data_error",
                    code="source_quality_evidence_extraction_failed",
                    message=f"Source evidence extraction failed for {document['document_id']}: {exc}",
                )
            )
    return evidence, records


def _extract_document_evidence(
    document: dict[str, Any],
    content: bytes,
    *,
    max_pages: int,
) -> dict[str, Any]:
    pages = _bounded_pages(content, max_pages=max_pages)
    if not pages:
        return _failed_evidence(
            document["document_id"],
            max_pages=max_pages,
            warning="no extractable text in bounded source evidence window",
        )

    combined = "\n".join(pages)
    snippets = _title_snippets(pages)
    warnings = []
    if not snippets:
        warnings.append("no visible title-like snippets in bounded source evidence window")
    return {
        "source_document_id": document["document_id"],
        "status": "completed",
        "inspected_page_span": {"start_page": 1, "end_page": len(pages)},
        "extraction_method": "embedded_pdf_text_probe",
        "visible_title_snippets": snippets,
        "company_clues": _company_clues(combined),
        "period_clues": _period_clues(combined),
        "basis_clues": _basis_clues(combined),
        "language_clues": _language_clues(combined),
        "role_clues": _role_clues(combined),
        "confidence": _confidence(snippets, combined),
        "warnings": warnings,
    }


def _bounded_pages(content: bytes, *, max_pages: int) -> list[str]:
    extracted_text = _extract_pdf_text(content, max_pages=max_pages)
    if extracted_text:
        pages = [_compact_text(page) for page in extracted_text.split("\f")]
        return [page for page in pages if page][:max_pages]

    text = content.decode("utf-8", errors="ignore")
    text = re.sub(r"^%PDF(?:-\d+(?:\.\d+)?)?\s*", "", text).strip()
    pages = [_compact_text(page) for page in text.split("\f")]
    return [page for page in pages if page][:max_pages]


def _extract_pdf_text(content: bytes, *, max_pages: int) -> str:
    if not content.lstrip().startswith(b"%PDF") or shutil.which("pdftotext") is None:
        return ""
    with tempfile.TemporaryDirectory(prefix="vinacount_source_evidence_") as temp_dir:
        pdf_path = Path(temp_dir) / "source.pdf"
        pdf_path.write_bytes(content)
        try:
            result = subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    "1",
                    "-l",
                    str(max_pages),
                    "-layout",
                    str(pdf_path),
                    "-",
                ],
                check=False,
                capture_output=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="ignore").strip()


def _title_snippets(pages: list[str]) -> list[str]:
    snippets = []
    for page in pages:
        for line in re.split(r"[\r\n]+", page):
            line = _compact_text(line)
            if not line:
                continue
            normalized = _normalize(line)
            if _looks_like_title(normalized):
                snippets.append(line[:160])
                break
    return snippets[:6]


def _looks_like_title(normalized: str) -> bool:
    markers = [
        "bao cao tai chinh",
        "financial statement",
        "cong bo thong tin",
        "information disclosure",
        "giai trinh",
        "explanation",
    ]
    return any(marker in normalized for marker in markers)


def _company_clues(text: str) -> list[str]:
    clues = []
    for match in re.finditer(r"\b(?:Cong ty|Ngan hang)[^\n\f]{0,100}", text, flags=re.IGNORECASE):
        clue = _compact_text(match.group(0))
        if clue and clue not in clues:
            clues.append(clue[:120])
    return clues[:3]


def _period_clues(text: str) -> list[str]:
    normalized = _normalize(text)
    clues = []
    for match in re.finditer(r"(?:quy|q)\s*([1-4])\D{0,20}(20\d{2})", normalized):
        clue = f"Q{match.group(1)} {match.group(2)}"
        if clue not in clues:
            clues.append(clue)
        canonical_clue = f"{match.group(2)}-Q{match.group(1)}"
        if canonical_clue not in clues:
            clues.append(canonical_clue)
    for year in re.findall(r"\b20\d{2}\b", normalized):
        if year not in clues:
            clues.append(year)
    return clues[:4]


def _basis_clues(text: str) -> list[str]:
    normalized = _normalize(text)
    clues = []
    if "hop nhat" in normalized or "consolidated" in normalized:
        clues.append("consolidated")
    if "cong ty me" in normalized or "parent company" in normalized:
        clues.append("parent")
    if "rieng" in normalized or "separate" in normalized or "standalone" in normalized:
        clues.append("separate")
    return clues


def _language_clues(text: str) -> list[str]:
    normalized = _normalize(text)
    clues = []
    if any(marker in normalized for marker in ["bao cao", "cong ty", "quy", "cong bo thong tin"]):
        clues.append("vi")
    if any(marker in normalized for marker in ["financial statements", "separate financial", "consolidated financial"]):
        clues.append("en")
    return clues


def _role_clues(text: str) -> list[str]:
    normalized = _normalize(text)
    clues = []
    if "bao cao tai chinh" in normalized or "financial statement" in normalized:
        clues.append("financial_statement")
    if "cong bo thong tin" in normalized or "information disclosure" in normalized:
        clues.append("disclosure")
    if "giai trinh" in normalized or "explanation" in normalized:
        clues.append("variance_explanation")
    return clues


def _confidence(snippets: list[str], text: str) -> str:
    roles = _role_clues(text)
    if snippets and "financial_statement" in roles:
        return "medium"
    if snippets:
        return "low"
    return "none"


def _failed_evidence(source_document_id: str, *, max_pages: int, warning: str) -> dict[str, Any]:
    return {
        "source_document_id": source_document_id,
        "status": "recoverable_data_error",
        "inspected_page_span": {"start_page": 1, "end_page": max_pages},
        "extraction_method": "embedded_pdf_text_probe",
        "visible_title_snippets": [],
        "company_clues": [],
        "period_clues": [],
        "basis_clues": [],
        "language_clues": [],
        "role_clues": [],
        "confidence": "none",
        "warnings": [warning],
    }


def _compact_text(value: str) -> str:
    printable = "".join(char if char.isprintable() or char.isspace() else " " for char in value)
    return re.sub(r"\s+", " ", printable).strip()


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_value).lower()
