import type {
  RuntimeRunView,
  FilingIntentValidationError,
  CanonicalFinalReport,
  FixtureMeta,
} from "@/types/runtime";

import step01 from "@/fixtures/happy_path/step_01_created.json";
import step02 from "@/fixtures/happy_path/step_02_discovering_sources.json";
import step03 from "@/fixtures/happy_path/step_03_awaiting_source_confirmation.json";
import step04 from "@/fixtures/happy_path/step_04_analyzing_cache_lookup.json";
import step05 from "@/fixtures/happy_path/step_05_analyzing_extraction.json";
import step06 from "@/fixtures/happy_path/step_06_analyzing_tool_analysis.json";
import step07 from "@/fixtures/happy_path/step_07_analyzing_detector_assessment.json";
import step08 from "@/fixtures/happy_path/step_08_analyzing_aggregation.json";
import step09 from "@/fixtures/happy_path/step_09_analyzing_report_generation.json";
import step10 from "@/fixtures/happy_path/step_10_completed.json";

import failedRecoverable from "@/fixtures/scenarios/failed_recoverable.json";
import failedNonRecoverable from "@/fixtures/scenarios/failed_non_recoverable.json";
import cancelled from "@/fixtures/scenarios/cancelled.json";
import cacheHitCompleted from "@/fixtures/scenarios/cache_hit_completed.json";
import sourceRejectedRetryable from "@/fixtures/scenarios/source_rejected_retryable.json";
import awaitingSourceConfirmation from "@/fixtures/scenarios/awaiting_source_confirmation.json";
import liveVietstockDirectPdf from "@/fixtures/scenarios/live_vietstock_direct_pdf.json";
import liveVietstockZipPackage from "@/fixtures/scenarios/live_vietstock_zip_package.json";
import oneSlotUnavailableRetry from "@/fixtures/scenarios/one_slot_unavailable_retry.json";
import bothSlotsUnavailable from "@/fixtures/scenarios/both_slots_unavailable.json";
import packageWarningsPresent from "@/fixtures/scenarios/package_warnings_present.json";
import notConfirmable from "@/fixtures/scenarios/not_confirmable.json";
import cacheSourceOnlyCompleted from "@/fixtures/scenarios/cache_source_only_completed.json";
import cacheLookupAmbiguousFailed from "@/fixtures/scenarios/cache_lookup_ambiguous_failed.json";
import cacheLookupInvalidBlockedFailed from "@/fixtures/scenarios/cache_lookup_invalid_blocked_failed.json";
import cacheLookupStaleRebuildCompleted from "@/fixtures/scenarios/cache_lookup_stale_rebuild_completed.json";

import invalidQuarter from "@/fixtures/filing_intent_errors/invalid_quarter.json";
import unresolvableCompany from "@/fixtures/filing_intent_errors/unresolvable_company_identifier.json";

import finalReportFixture from "@/fixtures/report_endpoint/final_report_vinacount_signal_2025_q3.json";

type FixtureRunView = RuntimeRunView & { _fixture_meta: FixtureMeta };

function castRun(data: unknown): FixtureRunView {
  return data as FixtureRunView;
}

export const HAPPY_PATH_STEPS: FixtureRunView[] = [
  step01, step02, step03, step04, step05,
  step06, step07, step08, step09, step10,
].map(castRun);

export type ScenarioKey =
  | "failed_recoverable"
  | "failed_non_recoverable"
  | "cancelled"
  | "cache_hit_completed"
  | "cache_source_only_completed"
  | "cache_lookup_ambiguous_failed"
  | "cache_lookup_invalid_blocked_failed"
  | "cache_lookup_stale_rebuild_completed"
  | "source_rejected_retryable"
  | "awaiting_source_confirmation"
  | "live_vietstock_direct_pdf"
  | "live_vietstock_zip_package"
  | "one_slot_unavailable_retry"
  | "both_slots_unavailable"
  | "package_warnings_present"
  | "not_confirmable";

export const SCENARIOS: Record<ScenarioKey, FixtureRunView> = {
  failed_recoverable: castRun(failedRecoverable),
  failed_non_recoverable: castRun(failedNonRecoverable),
  cancelled: castRun(cancelled),
  cache_hit_completed: castRun(cacheHitCompleted),
  cache_source_only_completed: castRun(cacheSourceOnlyCompleted),
  cache_lookup_ambiguous_failed: castRun(cacheLookupAmbiguousFailed),
  cache_lookup_invalid_blocked_failed: castRun(cacheLookupInvalidBlockedFailed),
  cache_lookup_stale_rebuild_completed: castRun(cacheLookupStaleRebuildCompleted),
  source_rejected_retryable: castRun(sourceRejectedRetryable),
  awaiting_source_confirmation: castRun(awaitingSourceConfirmation),
  live_vietstock_direct_pdf: castRun(liveVietstockDirectPdf),
  live_vietstock_zip_package: castRun(liveVietstockZipPackage),
  one_slot_unavailable_retry: castRun(oneSlotUnavailableRetry),
  both_slots_unavailable: castRun(bothSlotsUnavailable),
  package_warnings_present: castRun(packageWarningsPresent),
  not_confirmable: castRun(notConfirmable),
};

export const SCENARIO_LABELS: Record<ScenarioKey, string> = {
  failed_recoverable: "Paused (Resumable)",
  failed_non_recoverable: "Failed",
  cancelled: "Cancelled",
  cache_hit_completed: "Completed (Cached)",
  cache_source_only_completed: "Completed (Cache Rebuild)",
  cache_lookup_ambiguous_failed: "Cache Ambiguous (Failed)",
  cache_lookup_invalid_blocked_failed: "Cache Invalid Blocked (Failed)",
  cache_lookup_stale_rebuild_completed: "Stale Rebuild (Completed)",
  source_rejected_retryable: "Source Rejected",
  awaiting_source_confirmation: "Awaiting Confirmation",
  live_vietstock_direct_pdf: "Live Vietstock (Direct PDF)",
  live_vietstock_zip_package: "Live Vietstock (ZIP Package)",
  one_slot_unavailable_retry: "One Slot Unavailable",
  both_slots_unavailable: "Both Slots Unavailable",
  package_warnings_present: "Package Warnings",
  not_confirmable: "Not Confirmable",
};

export const FILING_INTENT_ERRORS: Record<string, FilingIntentValidationError> = {
  invalid_quarter: invalidQuarter as unknown as FilingIntentValidationError,
  unresolvable_company: unresolvableCompany as unknown as FilingIntentValidationError,
};

export const FINAL_REPORT: CanonicalFinalReport =
  finalReportFixture as unknown as CanonicalFinalReport;
