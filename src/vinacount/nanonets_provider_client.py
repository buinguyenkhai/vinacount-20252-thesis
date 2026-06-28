"""Nanonets OCR-3 / DocStrange provider client wiring."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

from vinacount.env_loader import load_dotenv_if_available
from vinacount.ocr_adapter import (
    NanonetsOcr3DocstrangeAdapter,
    NanonetsOcr3DocstrangeConfig,
)


NANONETS_API_KEY_ENV = "NANONETS_API_KEY"
NANONETS_BASE_URL = "https://extraction-api.nanonets.com/api/v1"


@dataclass
class NanonetsOcr3DocstrangeProviderClient:
    session: Any | None = None
    base_url: str = NANONETS_BASE_URL
    poll_interval_seconds: float = 5.0
    max_poll_seconds: int = 900
    extraction_mode: str = "async"

    def __call__(
        self,
        *,
        api_key: str,
        pdf_path: str,
        output_format: str,
        include_metadata: list[str],
        timeout_seconds: int,
        max_retries: int,
        model: str,
    ) -> dict[str, Any]:
        session = self.session or requests.Session()
        model_type = _model_type(model)
        if self.extraction_mode == "sync":
            payload = self._sync_extract(
                session=session,
                api_key=api_key,
                pdf_path=pdf_path,
                output_format=output_format,
                include_metadata=include_metadata,
                timeout_seconds=timeout_seconds,
                model_type=model_type,
            )
        else:
            payload = self._async_extract(
                session=session,
                api_key=api_key,
                pdf_path=pdf_path,
                output_format=output_format,
                include_metadata=include_metadata,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                model_type=model_type,
            )
        return _provider_response_from_payload(payload, output_format=output_format)

    def _sync_extract(
        self,
        *,
        session: Any,
        api_key: str,
        pdf_path: str,
        output_format: str,
        include_metadata: list[str],
        timeout_seconds: int,
        model_type: str,
    ) -> dict[str, Any]:
        with Path(pdf_path).open("rb") as handle:
            response = session.post(
                f"{self.base_url}/extract/sync",
                headers=_headers(api_key),
                files={"file": (Path(pdf_path).name, handle, "application/pdf")},
                data=_form_data(output_format, include_metadata, model_type),
                timeout=timeout_seconds,
            )
        response.raise_for_status()
        return response.json()

    def _async_extract(
        self,
        *,
        session: Any,
        api_key: str,
        pdf_path: str,
        output_format: str,
        include_metadata: list[str],
        timeout_seconds: int,
        max_retries: int,
        model_type: str,
    ) -> dict[str, Any]:
        with Path(pdf_path).open("rb") as handle:
            response = session.post(
                f"{self.base_url}/extract/async",
                headers=_headers(api_key),
                files={"file": (Path(pdf_path).name, handle, "application/pdf")},
                data=_form_data(output_format, include_metadata, model_type),
                timeout=timeout_seconds,
            )
        response.raise_for_status()
        queued = response.json()
        record_id = queued.get("record_id")
        if not record_id:
            raise RuntimeError("Nanonets async extraction did not return record_id")

        deadline = time.monotonic() + self.max_poll_seconds
        last_payload = queued
        transient_poll_failures = 0
        while time.monotonic() <= deadline:
            poll_response = session.get(
                f"{self.base_url}/extract/results/{record_id}",
                headers=_headers(api_key),
                timeout=timeout_seconds,
            )
            try:
                poll_response.raise_for_status()
            except Exception:
                transient_poll_failures += 1
                if transient_poll_failures > max_retries:
                    raise
                time.sleep(self.poll_interval_seconds)
                continue
            transient_poll_failures = 0
            last_payload = poll_response.json()
            status = last_payload.get("status")
            if status == "completed":
                return last_payload
            if status == "failed":
                raise RuntimeError(f"Nanonets extraction failed: {last_payload.get('message')}")
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Nanonets extraction timed out for record_id={record_id}: {last_payload}")


def make_nanonets_ocr3_docstrange_adapter_from_env(
    *,
    live_ocr_enabled: bool = True,
    timeout_seconds: int = 900,
    max_retries: int = 2,
    client: Any | None = None,
) -> NanonetsOcr3DocstrangeAdapter:
    load_dotenv_if_available()
    api_key = os.environ.get(NANONETS_API_KEY_ENV)
    provider_client = client or NanonetsOcr3DocstrangeProviderClient(
        max_poll_seconds=timeout_seconds
    )
    return NanonetsOcr3DocstrangeAdapter(
        config=NanonetsOcr3DocstrangeConfig(
            live_ocr_enabled=live_ocr_enabled,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            model="ocr-3-docstrange",
        ),
        client=provider_client,
    )


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _form_data(
    output_format: str,
    include_metadata: list[str],
    model_type: str,
) -> dict[str, str]:
    return {
        "output_format": output_format,
        "include_metadata": ",".join(include_metadata),
        "model_type": model_type,
    }


def _model_type(model: str) -> str:
    if model == "ocr-3-docstrange":
        return "nanonets-ocr-3"
    return model


def _provider_response_from_payload(
    payload: dict[str, Any],
    *,
    output_format: str,
) -> dict[str, Any]:
    result = payload.get("result") or {}
    formatted = result.get(output_format) or result.get("html") or {}
    content, metadata = _formatted_content_and_metadata(formatted)
    return {
        "run_id": payload.get("record_id"),
        "raw_html": content,
        "raw_tables": _raw_tables_from_html(content),
        "confidence_metadata": [
            {
                "provider": "nanonets_ocr_3_docstrange",
                "record_id": payload.get("record_id"),
                "confidence_score": _confidence_score(metadata),
                "pages_processed": payload.get("pages_processed"),
                "processing_time": payload.get("processing_time"),
            }
        ],
        "bounding_boxes": [
            {
                "provider": "nanonets_ocr_3_docstrange",
                "record_id": payload.get("record_id"),
                "bounding_boxes": metadata.get("bounding_boxes"),
            }
        ],
        "parser_warnings": [],
    }


def _formatted_content_and_metadata(formatted: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(formatted, dict):
        return "", {}
    content = formatted.get("content") or ""
    metadata = formatted.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(content, str):
        return str(content), metadata
    nested = _json_object(content)
    if nested is None:
        return content, metadata
    nested_content = nested.get("content")
    if isinstance(nested_content, str):
        content = nested_content
    nested_metadata = nested.get("metadata")
    if isinstance(nested_metadata, dict):
        metadata = {**nested_metadata, **metadata}
    return content, metadata


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _confidence_score(metadata: dict[str, Any]) -> Any:
    confidence_score = metadata.get("confidence_score")
    if confidence_score is not None:
        return confidence_score
    if metadata.get("overall_confidence") is None and metadata.get("page_confidence") is None:
        return None
    return {
        "overall": metadata.get("overall_confidence"),
        "page_confidence": metadata.get("page_confidence"),
    }


def _raw_tables_from_html(raw_html: str) -> list[dict[str, Any]]:
    parser = _HtmlTableParser()
    parser.feed(raw_html or "")
    tables = []
    for index, table in enumerate(parser.tables, start=1):
        raw_table = {
            "raw_table_id": f"RAW_TABLE_NANONETS_{index:03d}",
            "cells": table["cells"],
        }
        if table.get("page_number") is not None:
            raw_table["page_number"] = table["page_number"]
        tables.append(raw_table)
    return tables


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, Any]] = []
        self._current_page_number: int | None = None
        self._active_table: list[list[str]] | None = None
        self._active_table_page_number: int | None = None
        self._active_row: list[str] | None = None
        self._active_cell: list[str] | None = None
        self._active_heading: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._active_table = []
            self._active_table_page_number = self._current_page_number
        elif tag == "tr" and self._active_table is not None:
            self._active_row = []
        elif tag in {"td", "th"} and self._active_row is not None:
            self._active_cell = []
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._active_heading = []

    def handle_data(self, data: str) -> None:
        if self._active_cell is not None:
            self._active_cell.append(data)
        elif self._active_heading is not None:
            self._active_heading.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._active_row is not None and self._active_cell is not None:
            self._active_row.append(" ".join("".join(self._active_cell).split()))
            self._active_cell = None
        elif tag == "tr" and self._active_table is not None and self._active_row is not None:
            if self._active_row:
                self._active_table.append(self._active_row)
            self._active_row = None
        elif tag == "table" and self._active_table is not None:
            if self._active_table:
                self.tables.append(
                    {
                        "cells": self._active_table,
                        "page_number": self._active_table_page_number,
                    }
                )
            self._active_table = None
            self._active_table_page_number = None
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._active_heading is not None:
            page_number = _page_number_from_heading("".join(self._active_heading))
            if page_number is not None:
                self._current_page_number = page_number
            self._active_heading = None


def _page_number_from_heading(value: str) -> int | None:
    match = re.fullmatch(r"\s*page\s+(\d+)\s*", value, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))
