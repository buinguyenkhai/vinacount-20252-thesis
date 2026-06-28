"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import { useLiveRun } from "./live-run-store";
import { useFixtureStore } from "./fixture-store";
import { projectThread, type ThreadMessage } from "./thread-projection";
import type { RuntimeRunView, CanonicalFinalReport, EvidenceRef } from "@/types/runtime";

export type WorkspaceMode = "live" | "fixture";

export interface SelectedEvidence {
  evidenceRef: EvidenceRef;
  runId: string;
  findingTitle?: string;
  findingSummary?: string;
}

export interface WorkspaceState {
  mode: WorkspaceMode;
  setMode: (m: WorkspaceMode) => void;
  run: RuntimeRunView | null;
  report: CanonicalFinalReport | null;
  thread: ThreadMessage[];
  reportReaderOpen: boolean;
  modelId: string;
  modelUserSelected: boolean;
  openReportReader: () => void;
  closeReportReader: () => void;
  setModelId: (id: string) => void;
  selectedEvidence: SelectedEvidence | null;
  openEvidence: (ref: EvidenceRef, context?: { findingTitle?: string; findingSummary?: string }) => void;
  closeEvidence: () => void;
}

const WorkspaceContext = createContext<WorkspaceState | null>(null);

const MODEL_OPTIONS = [
  { id: "deepseek-v4-flash", label: "DeepSeek V4 Flash" },
  { id: "deepseek-v4-pro", label: "DeepSeek V4 Pro" },
] as const;

export { MODEL_OPTIONS };

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<WorkspaceMode>("live");
  const [reportReaderOpen, setReportReaderOpen] = useState(false);
  const [modelId, setModelIdRaw] = useState<string>(MODEL_OPTIONS[0].id);
  const [modelUserSelected, setModelUserSelected] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<SelectedEvidence | null>(null);

  const setModelId = useCallback((id: string) => {
    setModelIdRaw(id);
    setModelUserSelected(true);
  }, []);

  const liveRun = useLiveRun();
  const fixture = useFixtureStore();

  const run: RuntimeRunView | null =
    mode === "live" ? liveRun.run : fixture.run;

  const report: CanonicalFinalReport | null =
    mode === "live" ? liveRun.report : fixture.report;

  const thread = useMemo(
    () => (run ? projectThread(run) : []),
    [run],
  );

  const openReportReader = useCallback(() => setReportReaderOpen(true), []);

  const closeReportReader = useCallback(() => {
    setReportReaderOpen(false);
    setSelectedEvidence(null);
  }, []);

  const runId = run?.run_id ?? null;

  const openEvidence = useCallback(
    (ref: EvidenceRef, context?: { findingTitle?: string; findingSummary?: string }) => {
      if (!runId) return;
      setSelectedEvidence({ evidenceRef: ref, runId, ...context });
    },
    [runId],
  );

  const closeEvidence = useCallback(() => setSelectedEvidence(null), []);

  return (
    <WorkspaceContext.Provider
      value={{
        mode,
        setMode,
        run,
        report,
        thread,
        reportReaderOpen,
        modelId,
        modelUserSelected,
        openReportReader,
        closeReportReader,
        setModelId,
        selectedEvidence,
        openEvidence,
        closeEvidence,
      }}
    >
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace(): WorkspaceState {
  const ctx = useContext(WorkspaceContext);
  if (!ctx) throw new Error("useWorkspace must be used within WorkspaceProvider");
  return ctx;
}
