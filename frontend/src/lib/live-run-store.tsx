"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
} from "react";
import type {
  RuntimeRunView,
  CanonicalFinalReport,
  FilingIntentValidationError,
  RejectionReasonCode,
} from "@/types/runtime";
import { TERMINAL_STATUSES } from "@/types/runtime";
import {
  createRun,
  getRun,
  getReport,
  submitSourceConfirmation,
  resumeRun as apiResumeRun,
  stopRun as apiStopRun,
  ApiError,
  isValidationError,
  type CreateRunRequest,
} from "./runtime-api";

const POLL_INTERVAL_MS = 2500;

export type LiveRunPhase =
  | "idle"
  | "submitting"
  | "polling"
  | "terminal";

export interface LiveRunState {
  phase: LiveRunPhase;
  run: RuntimeRunView | null;
  report: CanonicalFinalReport | null;
  validationError: FilingIntentValidationError | null;
  error: string | null;
  actionPending: boolean;

  submitFilingIntent: (req: CreateRunRequest) => Promise<void>;
  loadSealedDemoRun: (runId: string) => Promise<void>;
  confirmSources: () => Promise<void>;
  rejectSource: (
    slotRole: string,
    reasonCode: RejectionReasonCode,
    comment?: string,
  ) => Promise<void>;
  retrySourceDiscovery: (slotRole: string) => Promise<void>;
  selectSourceCandidate: (slotRole: string, sourceDocumentId: string) => Promise<void>;
  resumeRun: () => Promise<void>;
  stopRun: () => Promise<void>;
  clearRun: () => void;
  clearValidationError: () => void;
}

const LiveRunContext = createContext<LiveRunState | null>(null);

export function LiveRunProvider({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<LiveRunPhase>("idle");
  const [run, setRun] = useState<RuntimeRunView | null>(null);
  const [report, setReport] = useState<CanonicalFinalReport | null>(null);
  const [validationError, setValidationError] =
    useState<FilingIntentValidationError | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionPending, setActionPending] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const runIdRef = useRef<string | null>(null);
  const reportFetchedRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const handleRunUpdate = useCallback(
    (updated: RuntimeRunView) => {
      setRun(updated);
      runIdRef.current = updated.run_id;

      if (TERMINAL_STATUSES.has(updated.status)) {
        setPhase("terminal");
        stopPolling();
      } else {
        setPhase("polling");
      }
    },
    [stopPolling],
  );

  const startPolling = useCallback(
    (runId: string) => {
      stopPolling();

      const poll = async () => {
        try {
          const updated = await getRun(runId);
          if (runIdRef.current !== runId) return;
          handleRunUpdate(updated);
        } catch {
          // Silently retry on next interval; transient network errors
          // shouldn't break the polling loop
        }
      };

      pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
    },
    [stopPolling, handleRunUpdate],
  );

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  // Dev-only: load a completed run by ?run=<run_id> query param
  const bootedRef = useRef(false);
  useEffect(() => {
    if (bootedRef.current) return;
    const params = new URLSearchParams(window.location.search);
    const runId = params.get("run");
    if (!runId) return;
    bootedRef.current = true;
    (async () => {
      try {
        const loaded = await getRun(runId);
        handleRunUpdate(loaded);
      } catch {
        // silently ignore — run not found or backend offline
      }
    })();
  }, [handleRunUpdate]);

  // Fetch report when available and not already fetched
  useEffect(() => {
    if (
      !run ||
      !run.final_report?.available ||
      reportFetchedRef.current === run.run_id
    )
      return;

    reportFetchedRef.current = run.run_id;

    getReport(run.run_id)
      .then(setReport)
      .catch(() => {
        // Report fetch failure is non-fatal; the run is still complete
        reportFetchedRef.current = null;
      });
  }, [run]);

  const submitFilingIntent = useCallback(
    async (req: CreateRunRequest) => {
      setPhase("submitting");
      setValidationError(null);
      setError(null);
      setReport(null);
      reportFetchedRef.current = null;

      try {
        const created = await createRun(req);
        handleRunUpdate(created);
        if (!TERMINAL_STATUSES.has(created.status)) {
          startPolling(created.run_id);
        }
      } catch (e) {
        if (e instanceof ApiError && isValidationError(e.body)) {
          setValidationError(e.body as FilingIntentValidationError);
          setPhase("idle");
        } else {
          setError(
            e instanceof Error ? e.message : "Failed to start analysis",
          );
          setPhase("idle");
        }
      }
    },
    [handleRunUpdate, startPolling],
  );

  const loadSealedDemoRun = useCallback(
    async (runId: string) => {
      stopPolling();
      setPhase("submitting");
      setValidationError(null);
      setError(null);
      setReport(null);
      reportFetchedRef.current = null;
      runIdRef.current = runId;

      try {
        const loaded = await getRun(runId);
        if (runIdRef.current !== runId) return;
        handleRunUpdate(loaded);
        if (!TERMINAL_STATUSES.has(loaded.status)) {
          startPolling(loaded.run_id);
        }
      } catch (e) {
        setError(
          e instanceof ApiError
            ? "Sealed demo run is unavailable. Start the runtime API with the thesis QA/QC registry before using the Try buttons."
            : e instanceof Error
              ? e.message
              : "Could not load the sealed demo run",
        );
        setRun(null);
        setPhase("idle");
        runIdRef.current = null;
      }
    },
    [handleRunUpdate, startPolling, stopPolling],
  );

  const withAction = useCallback(
    async (fn: () => Promise<RuntimeRunView>) => {
      if (!runIdRef.current) return;
      setActionPending(true);
      setError(null);
      try {
        const updated = await fn();
        handleRunUpdate(updated);
        if (!TERMINAL_STATUSES.has(updated.status)) {
          startPolling(updated.run_id);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Action failed");
      } finally {
        setActionPending(false);
      }
    },
    [handleRunUpdate, startPolling],
  );

  const confirmSources = useCallback(async () => {
    await withAction(() =>
      submitSourceConfirmation(runIdRef.current!, {
        action: "confirm_sources",
      }),
    );
  }, [withAction]);

  const rejectSource = useCallback(
    async (
      slotRole: string,
      reasonCode: RejectionReasonCode,
      comment?: string,
    ) => {
      await withAction(() =>
        submitSourceConfirmation(runIdRef.current!, {
          action: "reject_source",
          slot_role: slotRole,
          reason_code: reasonCode,
          ...(comment ? { comment } : {}),
        }),
      );
    },
    [withAction],
  );

  const retrySourceDiscovery = useCallback(
    async (slotRole: string) => {
      await withAction(() =>
        submitSourceConfirmation(runIdRef.current!, {
          action: "retry_source_discovery",
          slot_role: slotRole,
        }),
      );
    },
    [withAction],
  );

  const selectSourceCandidate = useCallback(
    async (slotRole: string, sourceDocumentId: string) => {
      await withAction(() =>
        submitSourceConfirmation(runIdRef.current!, {
          action: "select_source_candidate",
          slot_role: slotRole,
          source_document_id: sourceDocumentId,
        }),
      );
    },
    [withAction],
  );

  const resumeRun = useCallback(async () => {
    await withAction(() => apiResumeRun(runIdRef.current!));
  }, [withAction]);

  const stopRunAction = useCallback(async () => {
    await withAction(() => apiStopRun(runIdRef.current!));
  }, [withAction]);

  const clearRun = useCallback(() => {
    stopPolling();
    setPhase("idle");
    setRun(null);
    setReport(null);
    setValidationError(null);
    setError(null);
    setActionPending(false);
    runIdRef.current = null;
    reportFetchedRef.current = null;
  }, [stopPolling]);

  const clearValidationError = useCallback(() => {
    setValidationError(null);
  }, []);

  return (
    <LiveRunContext.Provider
      value={{
        phase,
        run,
        report,
        validationError,
        error,
        actionPending,
        submitFilingIntent,
        loadSealedDemoRun,
        confirmSources,
        rejectSource,
        retrySourceDiscovery,
        selectSourceCandidate,
        resumeRun,
        stopRun: stopRunAction,
        clearRun,
        clearValidationError,
      }}
    >
      {children}
    </LiveRunContext.Provider>
  );
}

export function useLiveRun(): LiveRunState {
  const ctx = useContext(LiveRunContext);
  if (!ctx)
    throw new Error("useLiveRun must be used within LiveRunProvider");
  return ctx;
}
