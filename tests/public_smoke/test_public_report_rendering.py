import tempfile
from pathlib import Path

from vinacount.runtime_api import DemoSourceDiscoveryAdapter, create_runtime_app
from vinacount.runtime_contract import assert_public_runtime_payload_safe, validate_final_report_endpoint
from vinacount.runtime_orchestration import DemoCacheLookupAdapter
from vinacount.runtime_run_registry import FilesystemArtifactBodyStore, RuntimeRunRegistry


def test_public_report_rendering_is_bounded_and_reviewer_safe():
    with tempfile.TemporaryDirectory() as temp_dir:
        app = create_runtime_app(
            registry=RuntimeRunRegistry(),
            artifact_store=FilesystemArtifactBodyStore(Path(temp_dir) / "artifacts"),
            source_discovery=DemoSourceDiscoveryAdapter(),
            cache_lookup=DemoCacheLookupAdapter(),
            run_id_factory=lambda: "run_public_report_001",
            report_generation_mode="deterministic_template",
            detector_mode="deterministic_local",
            auto_advance=False,
        )
        service = app.state.runtime_service

        service.create_runtime_run(
            {
                "company_identifier": "VCF",
                "target_fiscal_year": 2025,
                "target_quarter": 3,
                "report_basis_preference": "consolidated",
            }
        )
        service.update_source_confirmation(
            "run_public_report_001",
            {"action": "confirm_sources"},
        )
        service.advance_runtime_run_to_terminal("run_public_report_001")

        endpoint = validate_final_report_endpoint(
            service.get_final_report("run_public_report_001")
        )
        report = endpoint["report_json"]

        assert endpoint["report_markdown"].startswith("# Financial Reporting Risk-Signal Review")
        assert report["method_and_scope"]
        assert report["limitations"]
        assert report["overall_assessment"]["human_review_recommended"] is True
        assert report["grouped_findings"] or report["weak_or_limited_signals"]
        assert_public_runtime_payload_safe(endpoint)

        public_text = endpoint["report_markdown"].lower()
        for forbidden in [
            "raw ocr",
            "developer audit",
            "detector packet",
            "provider body",
            "chain of thought",
            "/home/",
            "c:\\",
        ]:
            assert forbidden not in public_text
