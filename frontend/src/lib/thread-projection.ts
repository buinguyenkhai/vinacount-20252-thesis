import type {
  RuntimeRunView,
  FilingIntent,
  SourceConfirmation,
  AllowedAction,
  RuntimeError,
  FinalReportMeta,
  RuntimeWarning,
} from "@/types/runtime";
import { mapStagesToPhases, type PhaseState } from "./phase-mapping";

export type ThreadMessage =
  | { kind: "user_input"; intent: FilingIntent }
  | {
      kind: "confirmation";
      intent: FilingIntent;
      confirmed: boolean;
    }
  | {
      kind: "progress";
      phases: PhaseState[];
      sourceConfirmation: SourceConfirmation | null;
      sourceConfirmationActive: boolean;
      allowedActions: AllowedAction[];
      warnings: RuntimeWarning[];
      elapsedSeconds: number;
    }
  | {
      kind: "completion";
      intent: FilingIntent;
      finalReport: FinalReportMeta;
      elapsedSeconds: number;
      allowedActions: AllowedAction[];
    }
  | {
      kind: "failed";
      error: RuntimeError;
      allowedActions: AllowedAction[];
      warnings: RuntimeWarning[];
    }
  | { kind: "cancelled" };

export function projectThread(run: RuntimeRunView): ThreadMessage[] {
  const messages: ThreadMessage[] = [];

  messages.push({ kind: "user_input", intent: run.filing_intent });

  if (run.status === "created") {
    messages.push({
      kind: "confirmation",
      intent: run.filing_intent,
      confirmed: false,
    });
    return messages;
  }

  messages.push({
    kind: "confirmation",
    intent: run.filing_intent,
    confirmed: true,
  });

  const isAwaitingConfirmation =
    run.status === "awaiting_source_confirmation";
  const hasStages = run.stages.length > 0;

  const failedCacheLookup =
    run.status === "failed" &&
    run.error?.code === "cache_lookup_failed";

  if (hasStages) {
    const phases = mapStagesToPhases(run.stages, run.current_stage);
    const progressWarnings = failedCacheLookup
      ? run.warnings.filter((w) => w.stage_id !== "cache_lookup")
      : run.warnings;
    messages.push({
      kind: "progress",
      phases,
      sourceConfirmation: run.source_confirmation,
      sourceConfirmationActive: isAwaitingConfirmation,
      allowedActions: run.allowed_actions,
      warnings: progressWarnings,
      elapsedSeconds: run.elapsed_seconds,
    });
  }

  if (run.status === "completed" && run.final_report) {
    messages.push({
      kind: "completion",
      intent: run.filing_intent,
      finalReport: run.final_report,
      elapsedSeconds: run.elapsed_seconds,
      allowedActions: run.allowed_actions,
    });
  }

  if (run.status === "failed" && run.error) {
    messages.push({
      kind: "failed",
      error: run.error,
      allowedActions: run.allowed_actions,
      warnings: run.warnings,
    });
  }

  if (run.status === "cancelled") {
    messages.push({ kind: "cancelled" });
  }

  return messages;
}
