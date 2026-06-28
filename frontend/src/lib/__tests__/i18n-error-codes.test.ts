import { describe, it, expect } from "vitest";
import { STRINGS } from "../i18n";
import type { ErrorCode } from "@/types/runtime";

const EXTRACTION_ERROR_CODES: ErrorCode[] = [
  "source_artifact_unreachable",
  "ocr_config_missing",
  "ocr_provider_failed",
  "raw_extraction_invalid",
];

describe("i18n error code coverage", () => {
  for (const code of EXTRACTION_ERROR_CODES) {
    it(`error.${code} has EN and VI translations`, () => {
      const entry = STRINGS[`error.${code}`];
      expect(entry).toBeDefined();
      expect(entry.en).toBeTruthy();
      expect(entry.vi).toBeTruthy();
    });
  }

  it("progress.live_extraction_runs has EN and VI translations", () => {
    const entry = STRINGS["progress.live_extraction_runs"];
    expect(entry).toBeDefined();
    expect(entry.en).toBeTruthy();
    expect(entry.vi).toBeTruthy();
  });
});

const FILING_CACHE_WARNING_CODES = [
  "filing_cache_lookup_report_memory_reusable",
  "filing_cache_lookup_source_only",
  "filing_cache_lookup_stale_rebuild_required",
  "filing_cache_lookup_incomplete_source_pair",
  "filing_cache_lookup_incomplete_report_memory_pair",
  "filing_cache_lookup_miss",
  "cache_reused_report_memory_artifacts",
  "raw_ocr_cache_activity",
] as const;

describe("i18n filing cache warning code coverage", () => {
  for (const code of FILING_CACHE_WARNING_CODES) {
    it(`warning.${code} has EN and VI translations`, () => {
      const entry = STRINGS[`warning.${code}`];
      expect(entry).toBeDefined();
      expect(entry.en).toBeTruthy();
      expect(entry.vi).toBeTruthy();
    });

    it(`warning.${code} VI translation is not the raw enum text`, () => {
      const entry = STRINGS[`warning.${code}`];
      expect(entry.vi).not.toBe(code);
      expect(entry.vi).not.toBe(`warning.${code}`);
      expect(entry.vi).not.toContain("_");
    });
  }
});

const CACHE_EXTRACTION_COUNT_KEYS = [
  // Backend #185 cache_lookup stage counts
  "source_slots_checked",
  "reusable_report_memory_artifacts",
  "cache_misses",
  // Backend #185 extraction stage counts
  "report_memory_artifacts_reused",
  "raw_ocr_cache_hits",
  "raw_ocr_cache_misses",
  "live_extraction_runs",
  // Legacy fixture keys (kept for backwards compatibility)
  "cache_entries_checked",
  "cache_hits",
] as const;

describe("i18n cache/extraction count label coverage", () => {
  for (const key of CACHE_EXTRACTION_COUNT_KEYS) {
    it(`progress.${key} has EN and VI translations`, () => {
      const entry = STRINGS[`progress.${key}`];
      expect(entry).toBeDefined();
      expect(entry.en).toBeTruthy();
      expect(entry.vi).toBeTruthy();
    });

    it(`progress.${key} VI translation is not the raw key`, () => {
      const entry = STRINGS[`progress.${key}`];
      expect(entry.vi).not.toBe(key);
      expect(entry.vi).not.toBe(`progress.${key}`);
    });
  }
});
