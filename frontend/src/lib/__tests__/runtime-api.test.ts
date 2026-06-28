import { afterEach, describe, expect, it, vi } from "vitest";
import { createRun } from "../runtime-api";

describe("createRun", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends the backend report synthesis model field", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          schema_version: "runtime_run_view.v1",
          run_id: "run_frontend_payload_001",
          created_at: "2026-06-20T00:00:00Z",
          updated_at: "2026-06-20T00:00:00Z",
          status: "awaiting_source_confirmation",
          recoverable: false,
          can_resume: false,
          elapsed_seconds: 0,
          filing_intent: {
            company_identifier: "NKG",
            company_name_vi: null,
            target_fiscal_year: 2021,
            target_quarter: 3,
            report_basis_preference: "consolidated",
          },
          runtime_config: {
            report_synthesis_model: {
              id: "deepseek-v4-flash",
              label: "DeepSeek V4 Flash",
              provider: "deepseek",
              selection: "user_selected",
            },
          },
          source_confirmation: {
            status: "ready_for_review",
            confirmable: true,
            hitl_boundary: "user_confirms_discovered_sources_before_analysis",
            slots: [],
            package_warnings: [],
          },
          stages: [],
          current_stage: "source_confirmation",
          warnings: [],
          allowed_actions: [],
          final_report: {
            available: false,
            report_id: null,
            generated_at: null,
            format: null,
            href: null,
          },
          error: null,
        }),
        { status: 201, headers: { "Content-Type": "application/json" } },
      ),
    );

    await createRun({
      company_identifier: "NKG",
      target_fiscal_year: 2021,
      target_quarter: 3,
      report_basis_preference: "consolidated",
      report_synthesis_model_id: "deepseek-v4-flash",
    });

    const request = fetchMock.mock.calls[0]?.[1] as RequestInit;
    const body = JSON.parse(String(request.body));
    expect(body.report_synthesis_model_id).toBe("deepseek-v4-flash");
    expect(body).not.toHaveProperty("orchestrator_model_id");
  });
});
