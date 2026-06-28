import asyncio
import json
import tempfile
from pathlib import Path

import httpx

from vinacount.runtime_api import DemoSourceDiscoveryAdapter as BaseDemoSourceDiscoveryAdapter
from vinacount.runtime_api import create_public_demo_runtime_app, create_runtime_app
from vinacount.runtime_contract import (
    assert_public_runtime_payload_safe,
    stable_json_dumps,
    validate_final_report_endpoint,
    validate_runtime_run_view,
)
from vinacount.runtime_orchestration import DemoCacheLookupAdapter
from vinacount.runtime_run_registry import FilesystemArtifactBodyStore, RuntimeRunRegistry


def test_public_runtime_smoke_serves_report_and_confirmed_source_pdf():
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_store = FilesystemArtifactBodyStore(Path(temp_dir) / "artifacts")
        registry = RuntimeRunRegistry()
        app = create_runtime_app(
            registry=registry,
            artifact_store=artifact_store,
            source_discovery=PublicSmokeSourceDiscoveryAdapter(),
            cache_lookup=DemoCacheLookupAdapter(),
            run_id_factory=lambda: "run_public_smoke_001",
            report_generation_mode="deterministic_template",
            detector_mode="deterministic_local",
            auto_advance=False,
        )
        created_response = _request(
            app,
            "POST",
            "/runtime-runs",
            json={
                "company_identifier": "VCF",
                "target_fiscal_year": 2025,
                "target_quarter": 3,
                "report_basis_preference": "consolidated",
            },
        )
        assert created_response.status_code == 201
        created = validate_runtime_run_view(created_response.json())
        assert created["status"] == "awaiting_source_confirmation"
        assert_public_runtime_payload_safe(created)

        confirm_response = _request(
            app,
            "POST",
            "/runtime-runs/run_public_smoke_001/source-confirmation",
            json={"action": "confirm_sources"},
        )
        assert confirm_response.status_code == 200
        confirmed = validate_runtime_run_view(confirm_response.json())
        assert confirmed["status"] == "analyzing"

        completed = validate_runtime_run_view(
            app.state.runtime_service.advance_runtime_run_to_terminal("run_public_smoke_001")
        )
        assert completed["status"] == "completed"
        assert completed["final_report"]["available"] is True
        assert_public_runtime_payload_safe(completed)

        _attach_smoke_source_pdfs(
            registry=registry,
            artifact_store=artifact_store,
            run_id="run_public_smoke_001",
            run_view=completed,
        )

        report_response = _request(app, "GET", completed["final_report"]["href"])
        assert report_response.status_code == 200
        report_endpoint = validate_final_report_endpoint(report_response.json())
        assert report_endpoint["run_id"] == "run_public_smoke_001"
        assert report_endpoint["report_markdown"].strip()
        assert_public_runtime_payload_safe(report_endpoint)

        target = next(
            slot["candidate"]
            for slot in completed["source_confirmation"]["slots"]
            if slot["role"] == "target"
        )
        pdf_response = _request(
            app,
            "GET",
            f"/runtime-runs/run_public_smoke_001/source-documents/{target['source_document_id']}/pdf",
        )
        assert pdf_response.status_code == 200
        assert pdf_response.headers["content-type"] == "application/pdf"
        assert pdf_response.content.startswith(b"%PDF")

        public_header_text = stable_json_dumps(dict(pdf_response.headers))
        _assert_public_text_safe(public_header_text)


def test_public_demo_factory_runs_to_final_report_without_provider_keys():
    with tempfile.TemporaryDirectory() as temp_dir:
        app = create_public_demo_runtime_app(
            registry=RuntimeRunRegistry(),
            artifact_store=FilesystemArtifactBodyStore(Path(temp_dir) / "artifacts"),
            run_id_factory=lambda: "run_public_demo_factory_001",
            auto_advance=False,
        )

        created_response = _request(
            app,
            "POST",
            "/runtime-runs",
            json={
                "company_identifier": "VCF",
                "target_fiscal_year": 2025,
                "target_quarter": 3,
                "report_basis_preference": "consolidated",
            },
        )
        assert created_response.status_code == 201
        created = validate_runtime_run_view(created_response.json())
        assert created["status"] == "awaiting_source_confirmation"
        assert created["source_confirmation"]["confirmable"] is True
        assert_public_runtime_payload_safe(created)

        confirm_response = _request(
            app,
            "POST",
            "/runtime-runs/run_public_demo_factory_001/source-confirmation",
            json={"action": "confirm_sources"},
        )
        assert confirm_response.status_code == 200
        confirmed = validate_runtime_run_view(confirm_response.json())
        assert confirmed["status"] == "analyzing"
        assert_public_runtime_payload_safe(confirmed)

        completed = validate_runtime_run_view(
            app.state.runtime_service.advance_runtime_run_to_terminal(
                "run_public_demo_factory_001"
            )
        )
        assert completed["status"] == "completed"

        report_response = _request(app, "GET", completed["final_report"]["href"])
        assert report_response.status_code == 200
        report_endpoint = validate_final_report_endpoint(report_response.json())
        assert report_endpoint["report_markdown"].startswith(
            "# Financial Reporting Risk-Signal Review"
        )
        assert_public_runtime_payload_safe(report_endpoint)


def _attach_smoke_source_pdfs(
    *,
    registry: RuntimeRunRegistry,
    artifact_store: FilesystemArtifactBodyStore,
    run_id: str,
    run_view: dict,
) -> None:
    for slot in run_view["source_confirmation"]["slots"]:
        candidate = slot["candidate"]
        body = _smoke_pdf_body(candidate)
        expected_sha256 = candidate["audit_references"]["source_document_fingerprint_sha256"]
        assert __import__("hashlib").sha256(body).hexdigest() == expected_sha256
        ref = artifact_store.put_bytes(
            run_id=run_id,
            artifact_id=f"public_smoke_source_{slot['role']}",
            kind="vietstock_source_pdf",
            body=body,
            schema_version="public_smoke_source_pdf.v1",
            version="public_smoke_v1",
            metadata={
                "report_role": slot["role"],
                "source_document_id": candidate["source_document_id"],
                "source_document_fingerprint_sha256": expected_sha256,
                "source_url": candidate["source_url"],
                "source_origin": candidate["source_origin"],
                "cache_outcome": "public_smoke_seed",
            },
        )
        registry.add_artifact_ref(run_id, ref)


def _smoke_pdf_body(candidate: dict) -> bytes:
    return (
        "%PDF-1.4\n"
        f"% Vinacount public smoke source: {candidate['source_document_id']}\n"
        "1 0 obj << /Type /Catalog >> endobj\n"
        "%%EOF\n"
    ).encode("utf-8")


def _assert_public_text_safe(text: str) -> None:
    lowered = text.lower()
    forbidden = [
        "raw_ocr",
        "detector_packet",
        "developer_audit",
        "artifact_manifest",
        "prompt",
        "provider_body",
        "llm_request",
        "llm_response",
        "/home/",
        "c:\\",
    ]
    for token in forbidden:
        assert token not in lowered


def _request(app, method: str, url: str, *, json: dict | None = None) -> httpx.Response:
    async def send() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.request(method, url, json=json)

    return asyncio.run(send())


class PublicSmokeSourceDiscoveryAdapter(BaseDemoSourceDiscoveryAdapter):
    def discover(self, filing_intent: dict) -> dict:
        payload = super().discover(filing_intent)
        for slot in payload["slots"]:
            candidate = slot["candidate"]
            body = _smoke_pdf_body(candidate)
            candidate["audit_references"]["source_document_fingerprint_sha256"] = __import__(
                "hashlib"
            ).sha256(body).hexdigest()
            candidate["file_size_bytes"] = len(body)
        return json.loads(json.dumps(payload))
