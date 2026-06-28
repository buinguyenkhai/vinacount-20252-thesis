import { describe, it, expect } from "vitest";
import { projectThread, type ThreadMessage } from "../thread-projection";
import type { RuntimeRunView } from "@/types/runtime";

import step01 from "@/fixtures/happy_path/step_01_created.json";
import step02 from "@/fixtures/happy_path/step_02_discovering_sources.json";
import step03 from "@/fixtures/happy_path/step_03_awaiting_source_confirmation.json";
import step05 from "@/fixtures/happy_path/step_05_analyzing_extraction.json";
import step10 from "@/fixtures/happy_path/step_10_completed.json";

import failedRecoverable from "@/fixtures/scenarios/failed_recoverable.json";
import failedNonRecoverable from "@/fixtures/scenarios/failed_non_recoverable.json";
import cancelled from "@/fixtures/scenarios/cancelled.json";
import cacheHit from "@/fixtures/scenarios/cache_hit_completed.json";

function cast(json: unknown): RuntimeRunView {
  return json as RuntimeRunView;
}

function kinds(messages: ThreadMessage[]): string[] {
  return messages.map((m) => m.kind);
}

describe("projectThread", () => {
  describe("happy path", () => {
    it("step 1 (created): user_input + unconfirmed confirmation", () => {
      const thread = projectThread(cast(step01));
      expect(kinds(thread)).toEqual(["user_input", "confirmation"]);
      expect(thread[0].kind).toBe("user_input");
      if (thread[1].kind === "confirmation") {
        expect(thread[1].confirmed).toBe(false);
      }
    });

    it("step 2 (discovering): user_input + confirmed + progress", () => {
      const thread = projectThread(cast(step02));
      expect(kinds(thread)).toEqual(["user_input", "confirmation", "progress"]);
      if (thread[1].kind === "confirmation") {
        expect(thread[1].confirmed).toBe(true);
      }
      if (thread[2].kind === "progress") {
        expect(thread[2].phases[0].status).toBe("active");
      }
    });

    it("step 3 (awaiting_source_confirmation): progress has source confirmation active", () => {
      const thread = projectThread(cast(step03));
      expect(kinds(thread)).toEqual(["user_input", "confirmation", "progress"]);
      if (thread[2].kind === "progress") {
        expect(thread[2].sourceConfirmationActive).toBe(true);
        expect(thread[2].sourceConfirmation).not.toBeNull();
      }
    });

    it("step 5 (extraction): source confirmation no longer active", () => {
      const thread = projectThread(cast(step05));
      expect(kinds(thread)).toEqual(["user_input", "confirmation", "progress"]);
      if (thread[2].kind === "progress") {
        expect(thread[2].sourceConfirmationActive).toBe(false);
      }
    });

    it("step 10 (completed): includes completion card", () => {
      const thread = projectThread(cast(step10));
      expect(kinds(thread)).toEqual([
        "user_input",
        "confirmation",
        "progress",
        "completion",
      ]);
      if (thread[3].kind === "completion") {
        expect(thread[3].finalReport.available).toBe(true);
        expect(thread[3].elapsedSeconds).toBe(480);
      }
    });
  });

  describe("scenarios", () => {
    it("failed_recoverable: includes failed card", () => {
      const thread = projectThread(cast(failedRecoverable));
      expect(kinds(thread)).toEqual([
        "user_input",
        "confirmation",
        "progress",
        "failed",
      ]);
      if (thread[3].kind === "failed") {
        expect(thread[3].error.recoverable).toBe(true);
        expect(thread[3].error.code).toBe("detector_timeout");
      }
    });

    it("failed_non_recoverable: includes failed card, not recoverable", () => {
      const thread = projectThread(cast(failedNonRecoverable));
      const last = thread[thread.length - 1];
      expect(last.kind).toBe("failed");
      if (last.kind === "failed") {
        expect(last.error.recoverable).toBe(false);
      }
    });

    it("cancelled: includes cancelled card", () => {
      const thread = projectThread(cast(cancelled));
      expect(kinds(thread)).toEqual([
        "user_input",
        "confirmation",
        "progress",
        "cancelled",
      ]);
    });

    it("cache_hit_completed: includes completion", () => {
      const thread = projectThread(cast(cacheHit));
      const last = thread[thread.length - 1];
      expect(last.kind).toBe("completion");
    });
  });

  describe("user_input message", () => {
    it("always includes filing intent", () => {
      const thread = projectThread(cast(step01));
      if (thread[0].kind === "user_input") {
        expect(thread[0].intent.company_identifier).toBe("VCF");
        expect(thread[0].intent.target_quarter).toBe(3);
        expect(thread[0].intent.target_fiscal_year).toBe(2025);
      }
    });
  });

  describe("progress message", () => {
    it("has 4 phases", () => {
      const thread = projectThread(cast(step05));
      if (thread[2].kind === "progress") {
        expect(thread[2].phases).toHaveLength(4);
        expect(thread[2].phases.map((p) => p.id)).toEqual([
          "finding_sources",
          "confirming_sources",
          "analyzing",
          "generating_report",
        ]);
      }
    });

    it("carries elapsed seconds", () => {
      const thread = projectThread(cast(step05));
      if (thread[2].kind === "progress") {
        expect(thread[2].elapsedSeconds).toBeGreaterThan(0);
      }
    });
  });
});
