import { describe, it, expect } from "vitest";
import type { CanonicalFinalReport } from "@/types/runtime";

import fixtureWithLimitations from "@/fixtures/report_endpoint/final_report_with_evidence_surface_limitations.json";
import fixtureLimitationsOnly from "@/fixtures/report_endpoint/final_report_limitations_only.json";
import fixtureHappyPath from "@/fixtures/report_endpoint/final_report_vinacount_signal_2025_q3.json";

function cast(json: unknown): CanonicalFinalReport {
  return json as CanonicalFinalReport;
}

const EVIDENCE_REF_PATTERN = /\s*Evidence ref:\s*(\S+?)\.?$/;

function classifyLimitationType(text: string): string {
  if (EVIDENCE_REF_PATTERN.test(text)) return "evidence_surface";
  return "scope";
}

function parseEvidenceRef(description: string): { text: string; evidenceRef: string | null } {
  const match = description.match(EVIDENCE_REF_PATTERN);
  if (!match) return { text: description, evidenceRef: null };
  return {
    text: description.slice(0, match.index!).trim(),
    evidenceRef: match[1],
  };
}

describe("evidence-surface limitation classification", () => {
  it("classifies evidence-surface limitations by Evidence ref pattern", () => {
    expect(
      classifyLimitationType(
        "Related-party evidence was not extracted into source-bound ReportMemory objects. Evidence ref: VINACOUNT_SIGNAL_2025_Q3:LIMIT_RELATED_PARTY_UNSUPPORTED.",
      ),
    ).toBe("evidence_surface");
  });

  it("classifies scope limitations without evidence ref as scope", () => {
    expect(classifyLimitationType("External context is excluded.")).toBe("scope");
    expect(classifyLimitationType("The report is limited to fixture-based report evidence.")).toBe(
      "scope",
    );
  });

  it("handles trailing period on evidence ref", () => {
    expect(
      classifyLimitationType(
        "Notes evidence surface status is not_extracted_yet. Evidence ref: VINACOUNT_SIGNAL_2025_Q3:LIMIT_NOTES_NOT_EXTRACTED.",
      ),
    ).toBe("evidence_surface");
  });
});

describe("evidence ref parsing", () => {
  it("extracts evidence ref from limitation string", () => {
    const result = parseEvidenceRef(
      "Related-party evidence was not extracted into source-bound ReportMemory objects. Evidence ref: VINACOUNT_SIGNAL_2025_Q3:LIMIT_RELATED_PARTY_UNSUPPORTED.",
    );
    expect(result.text).toBe(
      "Related-party evidence was not extracted into source-bound ReportMemory objects.",
    );
    expect(result.evidenceRef).toBe(
      "VINACOUNT_SIGNAL_2025_Q3:LIMIT_RELATED_PARTY_UNSUPPORTED",
    );
  });

  it("returns null evidence ref for scope limitations", () => {
    const result = parseEvidenceRef("External context is excluded.");
    expect(result.text).toBe("External context is excluded.");
    expect(result.evidenceRef).toBeNull();
  });

  it("handles ambiguous_failed_closed evidence ref", () => {
    const result = parseEvidenceRef(
      "Extraction quality evidence surface status is ambiguous_failed_closed. Evidence ref: VINACOUNT_SIGNAL_2025_Q3:LIMIT_EXTRACTION_QUALITY_AMBIGUOUS.",
    );
    expect(result.evidenceRef).toBe(
      "VINACOUNT_SIGNAL_2025_Q3:LIMIT_EXTRACTION_QUALITY_AMBIGUOUS",
    );
  });
});

describe("fixture: findings plus evidence-surface limitations", () => {
  const report = cast(fixtureWithLimitations);
  const limitations = report.report_json.limitations as string[];

  it("has both scope and evidence-surface limitations", () => {
    const types = limitations.map(classifyLimitationType);
    expect(types).toContain("scope");
    expect(types).toContain("evidence_surface");
  });

  it("has at least one grouped finding", () => {
    const findings = report.report_json.grouped_findings as unknown[];
    expect(findings.length).toBeGreaterThanOrEqual(1);
  });

  it("evidence-surface limitation mentions related-party evidence", () => {
    const surfaceLimitations = limitations.filter(
      (l) => classifyLimitationType(l) === "evidence_surface",
    );
    expect(surfaceLimitations.some((l) => l.includes("Related-party evidence"))).toBe(true);
  });

  it("evidence-surface limitation mentions accounting-policy evidence", () => {
    const surfaceLimitations = limitations.filter(
      (l) => classifyLimitationType(l) === "evidence_surface",
    );
    expect(surfaceLimitations.some((l) => l.includes("Accounting-policy evidence"))).toBe(true);
  });
});

describe("fixture: limitations only (no findings)", () => {
  const report = cast(fixtureLimitationsOnly);
  const limitations = report.report_json.limitations as string[];

  it("has no grouped findings", () => {
    const findings = report.report_json.grouped_findings as unknown[];
    expect(findings.length).toBe(0);
  });

  it("has multiple evidence-surface limitations", () => {
    const surfaceCount = limitations.filter(
      (l) => classifyLimitationType(l) === "evidence_surface",
    ).length;
    expect(surfaceCount).toBeGreaterThanOrEqual(3);
  });

  it("overall status is insufficient_evidence_for_overall_assessment", () => {
    const overall = report.report_json.overall_assessment as Record<string, unknown>;
    expect(overall.overall_review_status).toBe("insufficient_evidence_for_overall_assessment");
  });
});

describe("fixture: happy path without evidence-surface limitations", () => {
  const report = cast(fixtureHappyPath);
  const limitations = report.report_json.limitations as string[];

  it("has no evidence-surface limitations", () => {
    const surfaceCount = limitations.filter(
      (l) => classifyLimitationType(l) === "evidence_surface",
    ).length;
    expect(surfaceCount).toBe(0);
  });
});

describe("sanitization: forbidden strings never appear in fixtures", () => {
  const FORBIDDEN_PATTERNS = [
    "raw_ocr",
    "provider_metadata",
    "prompt",
    "detector_packet",
    "/home/",
    "artifacts/",
    "developer_audit_bundle",
    "DEEPSEEK_API_KEY",
    "VINACOUNT_SFT_VLLM_API_KEY",
    "127.0.0.1",
  ];

  const fixtures = [
    { name: "with_evidence_surface_limitations", data: fixtureWithLimitations },
    { name: "limitations_only", data: fixtureLimitationsOnly },
    { name: "happy_path", data: fixtureHappyPath },
  ];

  for (const fixture of fixtures) {
    describe(fixture.name, () => {
      const json = JSON.stringify(fixture.data);

      for (const pattern of FORBIDDEN_PATTERNS) {
        it(`does not contain "${pattern}"`, () => {
          expect(json).not.toContain(pattern);
        });
      }
    });
  }
});
