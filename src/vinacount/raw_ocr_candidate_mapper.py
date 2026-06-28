from __future__ import annotations

import re
import unicodedata
from typing import Any

from vinacount.filing_package import FilingPackage
from vinacount.nanonets_provider_client import _raw_tables_from_html
from vinacount.raw_extraction_artifact import RawExtractionArtifact


MAPPER_VERSION = "raw_ocr_candidate_mapper.v2"


class RawOcrCandidateMappingError(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def map_raw_ocr_artifact_to_extraction_candidates(
    artifact: RawExtractionArtifact,
    *,
    filing_package: FilingPackage,
    report_profile: str | None = None,
) -> dict[str, Any]:
    """Map provider-normalized raw OCR tables into ReportMemory extraction candidates."""

    profile = report_profile or _profile_from_existing_metadata(artifact.raw) or "unknown"
    if profile not in {"standard_corporate", "securities"}:
        raise RawOcrCandidateMappingError(
            "unsupported_report_profile",
            "Raw OCR candidate mapping supports standard_corporate and securities report profiles.",
        )
    raw_tables = _raw_tables_with_navigation(artifact.raw)
    if not isinstance(raw_tables, list) or not raw_tables:
        raise RawOcrCandidateMappingError(
            "raw_tables_missing",
            "Raw OCR candidate mapping requires provider-normalized raw tables.",
        )
    tables = _statement_tables(
        raw_tables,
        period=filing_package.period,
        source_document_id=artifact.source_document_id,
        report_profile=profile,
    )
    if not tables:
        raise RawOcrCandidateMappingError(
            "structured_statement_tables_not_found",
            "Raw OCR tables did not contain supported statement rows.",
        )
    return {
        "metadata": {
            "company_name": filing_package.company_name,
            "ticker": filing_package.raw["company"].get("ticker"),
            "period": filing_package.period,
            "report_profile": profile,
            "report_basis": filing_package.report_basis,
            "business_context_tags": [profile],
            "report_assurance_type": "unaudited",
            "currency": "VND",
            "unit": "vnd",
            "language": "vi",
            "report_period_type": "quarterly",
            "mapper_version": MAPPER_VERSION,
            "extraction_limitations": [
                "deterministic_raw_table_candidate_mapping",
                "current_period_columns_only",
            ],
        },
        "sections": _sections_for_tables(tables),
        "statement_tables": tables,
        "notes": [],
        "variance_explanations": [],
        "evidence_surface_status": _table_only_surface_statuses(
            filing_package=filing_package,
            source_document_id=artifact.source_document_id,
        ),
        "mapper_diagnostics": {
            "mapper_version": MAPPER_VERSION,
            "raw_tables_total": len(raw_tables),
            "statement_tables_total": len(tables),
            "candidate_rows_total": sum(len(table["rows"]) for table in tables),
        },
    }


def _raw_tables_with_navigation(raw: dict[str, Any]) -> Any:
    raw_tables = raw.get("raw_tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        return raw_tables
    if any(isinstance(table, dict) and _positive_int(table.get("page_number")) for table in raw_tables):
        return raw_tables
    raw_html = raw.get("raw_html")
    if not isinstance(raw_html, str) or not raw_html.strip():
        return raw_tables
    parsed_tables = _raw_tables_from_html(raw_html)
    if len(parsed_tables) != len(raw_tables):
        return raw_tables
    enriched_tables = []
    for raw_table, parsed_table in zip(raw_tables, parsed_tables, strict=True):
        if not isinstance(raw_table, dict):
            enriched_tables.append(raw_table)
            continue
        enriched = dict(raw_table)
        page_number = _positive_int(parsed_table.get("page_number"))
        if page_number is not None:
            enriched["page_number"] = page_number
        enriched_tables.append(enriched)
    return enriched_tables


def _table_only_surface_statuses(
    *,
    filing_package: FilingPackage,
    source_document_id: str,
) -> list[dict[str, Any]]:
    report_ref_prefix = f"{filing_package.raw['company'].get('ticker')}_{_period_slug(filing_package.period)}"
    surfaces = [
        "notes",
        "variance_explanations",
        "related_party_evidence",
        "accounting_policy_evidence",
        "extraction_quality",
    ]
    return [
        {
            "surface": surface,
            "state": "unsupported_by_extraction_path",
            "source_document_id": source_document_id,
            "evidence_ref": f"{report_ref_prefix}:LIMIT_{surface.upper()}_UNSUPPORTED",
            "producer": "deterministic_raw_ocr_mapper",
            "producer_version": MAPPER_VERSION,
            "reason_code": "deterministic_raw_ocr_mapper_table_only",
            "message": (
                "Deterministic raw OCR mapping produced table evidence only; this evidence "
                "surface was not extracted into source-bound ReportMemory objects."
            ),
        }
        for surface in surfaces
    ]


def _profile_from_existing_metadata(raw: dict[str, Any]) -> str | None:
    candidates = raw.get("extraction_candidates")
    if isinstance(candidates, dict):
        metadata = candidates.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("report_profile"), str):
            return metadata["report_profile"]
    return None


def _statement_tables(
    raw_tables: list[dict[str, Any]],
    *,
    period: str,
    source_document_id: str,
    report_profile: str,
) -> list[dict[str, Any]]:
    balance_rows = []
    income_quarter_rows = []
    income_ytd_rows = []
    cash_flow_rows = []
    for raw_table in raw_tables:
        cells = raw_table.get("cells")
        if not isinstance(cells, list):
            continue
        normalized_table_text = _normalize(" ".join(str(cell) for row in cells if isinstance(row, list) for cell in row))
        cash_flow_context = "luu chuyen tien" in normalized_table_text
        year_to_date_context = "luy ke" in normalized_table_text
        ambiguous_current_period_columns = _has_ambiguous_current_period_columns(cells)
        raw_table_id = str(raw_table.get("raw_table_id") or "RAW_TABLE")
        raw_page_number = _positive_int(raw_table.get("page_number"))
        for row_index, row in enumerate(cells):
            parsed = _candidate_row(
                row,
                period=period,
                source_document_id=source_document_id,
                raw_table_id=raw_table_id,
                raw_row_index=row_index,
                raw_page_number=raw_page_number,
                cash_flow_context=cash_flow_context,
                year_to_date_context=year_to_date_context,
                ambiguous_current_period_columns=ambiguous_current_period_columns,
            )
            if parsed is None:
                continue
            if parsed["standard_account"] == "fvtpl_unrealized_gain":
                income_ytd_rows.append(parsed)
            elif parsed["account_group"] in {"asset_quality", "receivables_credit"}:
                balance_rows.append(parsed)
            elif parsed["standard_account"] == "revenue":
                income_quarter_rows.append(parsed)
            elif parsed["standard_account"] == "profit_after_tax":
                income_ytd_rows.append(parsed)
            elif parsed["standard_account"] == "operating_cash_flow":
                cash_flow_rows.append(parsed)

    tables = []
    if balance_rows:
        tables.append(
            _table(
                table_type="balance_sheet",
                section_type="balance_sheet",
                period_basis="balance_sheet_date" if report_profile == "securities" else "quarter",
                period=period,
                source_document_id=source_document_id,
                rows=_dedupe_rows(balance_rows),
            )
        )
    income_rows = _dedupe_rows(sorted(income_quarter_rows, key=_row_priority))
    if income_rows:
        tables.append(
            _table(
                table_type="income_statement",
                section_type="income_statement",
                period_basis="quarter",
                period=period,
                source_document_id=source_document_id,
                rows=income_rows,
            )
        )
    ytd_rows = _dedupe_rows(income_ytd_rows)
    if ytd_rows:
        tables.append(
            _table(
                table_type="income_statement",
                section_type="income_statement",
                period_basis="year_to_date",
                period=period,
                source_document_id=source_document_id,
                rows=ytd_rows,
            )
        )
    if cash_flow_rows:
        tables.append(
            _table(
                table_type="cash_flow_statement",
                section_type="cash_flow_statement",
                period_basis="year_to_date",
                period=period,
                source_document_id=source_document_id,
                rows=_dedupe_rows(cash_flow_rows),
            )
        )
    return tables


def _candidate_row(
    row: Any,
    *,
    period: str,
    source_document_id: str,
    raw_table_id: str,
    raw_row_index: int,
    raw_page_number: int | None,
    cash_flow_context: bool = False,
    year_to_date_context: bool = False,
    ambiguous_current_period_columns: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(row, list):
        return None
    text_cells = [str(cell).strip() for cell in row]
    code = _account_code(text_cells)
    label = _label(text_cells, code)
    normalized_label = _normalize(label)
    account = _standard_account(code, normalized_label, cash_flow_context=cash_flow_context)
    if account is None:
        return None
    value = _selected_value(
        text_cells,
        account,
        code,
        year_to_date_context=year_to_date_context,
        ambiguous_current_period_columns=ambiguous_current_period_columns,
    )
    if value is None:
        return None
    row_id = f"ROW_{account.upper()}_{_period_slug(period)}"
    cell_id = f"CELL_{account.upper()}_{_period_slug(period)}"
    source_excerpt = _source_excerpt(text_cells)
    provenance = {
        "source_document_id": source_document_id,
        "raw_table_id": raw_table_id,
        "raw_row_index": raw_row_index,
        "mapper_version": MAPPER_VERSION,
    }
    cell = {
        "cell_id": cell_id,
        "period": period,
        "value": value,
        "source_document_id": source_document_id,
    }
    if raw_page_number is not None:
        provenance["page_number"] = raw_page_number
        cell["source_page_number"] = raw_page_number
    if source_excerpt:
        provenance["source_excerpt"] = source_excerpt
        cell["source_excerpt"] = source_excerpt
    row_payload = {
        "row_id": row_id,
        "account_code": code,
        "standard_account": account,
        "account_group": _account_group(account),
        "label": _english_label(account),
        "original_label": label,
        "source_document_id": source_document_id,
        "evidence_provenance": provenance,
        "cells": [cell],
    }
    if raw_page_number is not None:
        row_payload["source_page_number"] = raw_page_number
    if source_excerpt:
        row_payload["source_excerpt"] = source_excerpt
    return row_payload


def _source_excerpt(text_cells: list[str]) -> str | None:
    excerpt = " | ".join(cell for cell in text_cells if cell)
    if not excerpt:
        return None
    return excerpt[:500]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _account_code(cells: list[str]) -> str | None:
    for cell in cells[:4]:
        if re.fullmatch(r"\d{1,3}(?:\.\d+)?[a-z]?", cell.strip(), flags=re.IGNORECASE):
            return cell.strip()
    return None


def _label(cells: list[str], code: str | None) -> str:
    for cell in cells:
        if not cell or cell == code:
            continue
        normalized = _normalize(cell)
        if re.fullmatch(r"[ivx]+", normalized) or re.fullmatch(r"[a-z]", normalized):
            continue
        if _parse_number(cell) is None:
            return cell
    return ""


def _standard_account(code: str | None, normalized_label: str, *, cash_flow_context: bool = False) -> str | None:
    if code == "131" or "phai thu ngan han cua khach hang" in normalized_label:
        return "trade_receivables"
    if code == "117" and "cac khoan phai thu" in normalized_label:
        return "trade_receivables"
    if code == "60" and cash_flow_context:
        return "operating_cash_flow"
    if code == "114" and "cac khoan cho vay" in normalized_label:
        return "margin_lending"
    if code == "116" and "du phong suy giam" in normalized_label:
        return "margin_impairment"
    if code == "112" and "fvtpl" in normalized_label:
        return "fvtpl_assets"
    if code == "115" and "san sang de ban" in normalized_label:
        return "afs_assets"
    if code in {"113", "212.1"} and "nam giu den ngay dao han" in normalized_label:
        return "htm_assets"
    if code == "01.2" and _is_fvtpl_unrealized_gain_label(normalized_label):
        return "fvtpl_unrealized_gain"
    if "doanh thu thuan" in normalized_label:
        return "revenue"
    if code in {"1", "01", "10", "20"} and "doanh thu" in normalized_label:
        return "revenue"
    if (
        "loi nhuan sau thue" in normalized_label
        and "chua phan phoi" not in normalized_label
        and "cong ty me" not in normalized_label
        and "co dong khong kiem soat" not in normalized_label
    ):
        return "profit_after_tax"
    if code == "60" and "loi nhuan sau thue" in normalized_label:
        return "profit_after_tax"
    if code == "200" and "loi nhuan" in normalized_label and "sau thue" in normalized_label:
        return "profit_after_tax"
    if code == "20" and ("luu chuyen tien thuan tu hoat dong kinh doanh" in normalized_label or cash_flow_context):
        return "operating_cash_flow"
    return None


def _selected_value(
    cells: list[str],
    account: str,
    code: str | None,
    *,
    year_to_date_context: bool = False,
    ambiguous_current_period_columns: bool = False,
) -> int | None:
    value_cells = cells
    if code is not None and code in cells:
        value_cells = cells[cells.index(code) + 1 :]
    numbers = [_parse_number(cell) for cell in value_cells]
    values = [number for number in numbers if number is not None]
    if len(values) >= 2 and abs(values[0]) < 1000 and any(abs(value) >= 1000 for value in values[1:]):
        values = values[1:]
    if not values:
        return None
    if ambiguous_current_period_columns and len(values) >= 2:
        raise RawOcrCandidateMappingError(
            "ambiguous_current_period_columns",
            "Raw OCR table has multiple current-period value columns without a deterministic selector.",
        )
    if account == "fvtpl_unrealized_gain" and year_to_date_context and len(values) >= 2:
        if len(values) >= 4:
            return values[2]
        return values[-1]
    if account == "profit_after_tax" and len(values) >= 3:
        return values[2]
    return values[0]


def _has_ambiguous_current_period_columns(cells: list[Any]) -> bool:
    for row in cells[:3]:
        if not isinstance(row, list):
            continue
        normalized_cells = [_normalize(str(cell)) for cell in row]
        current_markers = sum(
            1
            for cell in normalized_cells
            if "nam nay" in cell or "ky nay" in cell
        )
        prior_markers = sum(1 for cell in normalized_cells if "nam truoc" in cell)
        ytd_markers = sum(1 for cell in normalized_cells if "luy ke" in cell)
        if current_markers >= 2 and prior_markers == 0 and ytd_markers == 0:
            return True
    return False


def _is_fvtpl_unrealized_gain_label(normalized_label: str) -> bool:
    if "danh gia lai" in normalized_label and "fvtpl" in normalized_label:
        return True
    if "chenh lech tang" not in normalized_label or "danh gia" not in normalized_label:
        return False
    financial_asset_context = "tstc" in normalized_label or "tai san tai chinh" in normalized_label
    profit_loss_context = (
        "lai lo" in normalized_label
        or "thong qua lo" in normalized_label
        or "thong qua lai lo" in normalized_label
    )
    return financial_asset_context and profit_loss_context


def _parse_number(value: str) -> int | None:
    text = str(value).strip()
    if not text or text in {"-", "–"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = text.strip("()").replace(".", "").replace(",", "")
    if not re.fullmatch(r"-?\d+", cleaned):
        return None
    parsed = int(cleaned)
    return -parsed if negative else parsed


def _table(
    *,
    table_type: str,
    section_type: str,
    period_basis: str,
    period: str,
    source_document_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    section_id = f"SEC_{section_type.upper()}_{_period_slug(period)}"
    table_id = f"TBL_{table_type.upper()}_{period_basis.upper()}_{_period_slug(period)}"
    return {
        "table_id": table_id,
        "section_id": section_id,
        "table_type": table_type,
        "title": table_type.replace("_", " ").title(),
        "period_basis": period_basis,
        "source_document_id": source_document_id,
        "rows": rows,
    }


def _sections_for_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    sections = []
    for table in tables:
        if table["section_id"] in seen:
            continue
        seen.add(table["section_id"])
        section_type = table["section_id"].removeprefix("SEC_").rsplit("_", maxsplit=2)[0].lower()
        sections.append(
            {
                "section_id": table["section_id"],
                "section_type": section_type,
                "title": table["title"],
                "status": "included",
            }
        )
    return sections


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = row["standard_account"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _row_priority(row: dict[str, Any]) -> tuple[int, str]:
    original_label = _normalize(row.get("original_label", ""))
    if row["standard_account"] == "revenue" and (
        "doanh thu thuan" in original_label or "cong doanh thu hoat dong" in original_label
    ):
        return (0, row["standard_account"])
    return (1, row["standard_account"])


def _account_group(account: str) -> str:
    if account in {"revenue", "profit_after_tax"}:
        return "revenue_income"
    if account in {"trade_receivables", "margin_lending", "margin_impairment"}:
        return "receivables_credit"
    if account == "operating_cash_flow":
        return "cashflow"
    if account in {"fvtpl_assets", "afs_assets", "htm_assets", "fvtpl_unrealized_gain"}:
        return "asset_quality"
    return "unknown"


def _english_label(account: str) -> str:
    return {
        "revenue": "Revenue",
        "trade_receivables": "Trade receivables",
        "profit_after_tax": "Profit after tax",
        "operating_cash_flow": "Operating cash flow",
        "margin_lending": "Margin lending",
        "margin_impairment": "Impairment allowance for margin lending",
        "fvtpl_assets": "FVTPL financial assets",
        "afs_assets": "AFS financial assets",
        "htm_assets": "HTM investments",
        "fvtpl_unrealized_gain": "Unrealized FVTPL gain",
    }[account]


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFD", value.strip().casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^\w\s.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _period_slug(period: str) -> str:
    return period.replace("-", "_")
