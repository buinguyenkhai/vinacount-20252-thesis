import { describe, it, expect } from "vitest";
import { parseFilingIntent } from "../filing-intent-parser";

describe("parseFilingIntent", () => {
  it("parses full English input", () => {
    const result = parseFilingIntent("VCF Q3 2025");
    expect(result).toEqual({
      ok: true,
      fields: {
        company_identifier: "VCF",
        target_quarter: 3,
        target_fiscal_year: 2025,
      },
    });
  });

  it("parses full Vietnamese input", () => {
    const result = parseFilingIntent("FPT quý 4 2023");
    expect(result).toEqual({
      ok: true,
      fields: {
        company_identifier: "FPT",
        target_quarter: 4,
        target_fiscal_year: 2023,
      },
    });
  });

  it("parses with consolidated basis", () => {
    const result = parseFilingIntent("VCF Q1 2025 hợp nhất");
    expect(result).toEqual({
      ok: true,
      fields: {
        company_identifier: "VCF",
        target_quarter: 1,
        target_fiscal_year: 2025,
        report_basis_preference: "consolidated",
      },
    });
  });

  it("parses with separate basis", () => {
    const result = parseFilingIntent("HPG Q2 2024 riêng lẻ");
    expect(result).toEqual({
      ok: true,
      fields: {
        company_identifier: "HPG",
        target_quarter: 2,
        target_fiscal_year: 2024,
        report_basis_preference: "separate",
      },
    });
  });

  it("parses English basis words", () => {
    const result = parseFilingIntent("VIC Q1 2025 consolidated");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.report_basis_preference).toBe("consolidated");
    }
  });

  it("parses lowercase ticker", () => {
    const result = parseFilingIntent("vcf q3 2025");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("VCF");
      expect(result.fields.target_quarter).toBe(3);
    }
  });

  it("parses with 'năm' prefix on year", () => {
    const result = parseFilingIntent("FPT Q4 năm 2023");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.target_fiscal_year).toBe(2023);
    }
  });

  it("parses ticker-only (partial)", () => {
    const result = parseFilingIntent("VCF");
    expect(result).toEqual({
      ok: true,
      fields: { company_identifier: "VCF" },
    });
  });

  it("parses ticker + quarter without year", () => {
    const result = parseFilingIntent("FPT Q4");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("FPT");
      expect(result.fields.target_quarter).toBe(4);
      expect(result.fields.target_fiscal_year).toBeUndefined();
    }
  });

  it("returns error for empty input", () => {
    const result = parseFilingIntent("");
    expect(result).toEqual({ ok: false, message: "empty_input" });
  });

  it("returns error for whitespace-only input", () => {
    const result = parseFilingIntent("   ");
    expect(result).toEqual({ ok: false, message: "empty_input" });
  });

  it("returns error for unrecognizable input", () => {
    const result = parseFilingIntent("hello world");
    expect(result).toEqual({ ok: false, message: "no_fields_extracted" });
  });

  it("handles Vietnamese analysis prefix", () => {
    const result = parseFilingIntent("Phân tích VCF Q3 2025");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("VCF");
      expect(result.fields.target_quarter).toBe(3);
      expect(result.fields.target_fiscal_year).toBe(2025);
    }
  });

  it("handles 'analyze' prefix", () => {
    const result = parseFilingIntent("analyze FPT Q1 2024");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("FPT");
    }
  });

  it("handles 5-letter ticker", () => {
    const result = parseFilingIntent("VNMIL Q2 2025");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("VNMIL");
    }
  });

  // Draft confirmation flow tests

  it("ticker-only input does not include defaults", () => {
    const result = parseFilingIntent("NKG");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("NKG");
      expect(result.fields.target_quarter).toBeUndefined();
      expect(result.fields.target_fiscal_year).toBeUndefined();
      expect(result.fields.report_basis_preference).toBeUndefined();
    }
  });

  it("full input extracts all four fields with no extras", () => {
    const result = parseFilingIntent("NKG Q3 2021 hợp nhất");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields).toEqual({
        company_identifier: "NKG",
        target_quarter: 3,
        target_fiscal_year: 2021,
        report_basis_preference: "consolidated",
      });
    }
  });

  it("all-diacritics input with no extractable fields returns error", () => {
    const result = parseFilingIntent("báo cáo tài chính");
    expect(result.ok).toBe(false);
  });

  it("company name extracts false ticker that draft card can correct", () => {
    const result = parseFilingIntent("Thép Nam Kim Q3 2021");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("KIM");
      expect(result.fields.target_quarter).toBe(3);
    }
  });

  it("partial input returns only extracted fields", () => {
    const result = parseFilingIntent("NKG 2024");
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.fields.company_identifier).toBe("NKG");
      expect(result.fields.target_fiscal_year).toBe(2024);
      expect(result.fields.target_quarter).toBeUndefined();
      expect(result.fields.report_basis_preference).toBeUndefined();
    }
  });
});
