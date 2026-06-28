"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import type { RuntimeRunView, CanonicalFinalReport } from "@/types/runtime";
import {
  HAPPY_PATH_STEPS,
  SCENARIOS,
  SCENARIO_LABELS,
  FINAL_REPORT,
  type ScenarioKey,
} from "./fixture-data";

const STEP_LABELS: Record<string, string> = {
  created: "Created",
  discovering_sources: "Discovering Sources",
  awaiting_source_confirmation: "Source Confirmation",
  "analyzing cache_lookup": "Cache Lookup",
  "analyzing extraction": "Extraction",
  "analyzing tool_analysis": "Tool Analysis",
  "analyzing detector_assessment": "Detector Assessment",
  "analyzing aggregation": "Aggregation",
  "analyzing report_generation": "Report Generation",
  completed: "Completed",
};

function humanizeLabel(raw: string): string {
  return (
    STEP_LABELS[raw] ??
    raw
      .split(/[_ ]/)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ")
  );
}

type FixtureMode =
  | { kind: "happy_path"; stepIndex: number }
  | { kind: "scenario"; key: ScenarioKey };

interface FixtureStore {
  mode: FixtureMode;
  run: RuntimeRunView;
  report: CanonicalFinalReport | null;
  stepLabel: string;
  stepPosition: string;
  canPrev: boolean;
  canNext: boolean;
  prev: () => void;
  next: () => void;
  selectScenario: (key: ScenarioKey) => void;
  resetHappyPath: () => void;
}

const FixtureStoreContext = createContext<FixtureStore | null>(null);

export function FixtureStoreProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<FixtureMode>({
    kind: "happy_path",
    stepIndex: 0,
  });

  const run: RuntimeRunView =
    mode.kind === "happy_path"
      ? HAPPY_PATH_STEPS[mode.stepIndex]
      : SCENARIOS[mode.key];

  const report: CanonicalFinalReport | null =
    run.status === "completed" && run.final_report?.available
      ? FINAL_REPORT
      : null;

  const meta =
    mode.kind === "happy_path"
      ? (HAPPY_PATH_STEPS[mode.stepIndex] as unknown as { _fixture_meta: { step_label: string } })
          ._fixture_meta
      : null;

  const stepLabel =
    mode.kind === "happy_path"
      ? humanizeLabel(meta!.step_label)
      : SCENARIO_LABELS[mode.key];

  const stepPosition =
    mode.kind === "happy_path"
      ? `${mode.stepIndex + 1} of ${HAPPY_PATH_STEPS.length}`
      : "scenario";

  const canPrev = mode.kind === "happy_path" && mode.stepIndex > 0;
  const canNext =
    mode.kind === "happy_path" &&
    mode.stepIndex < HAPPY_PATH_STEPS.length - 1;

  const prev = useCallback(() => {
    setMode((m) =>
      m.kind === "happy_path" && m.stepIndex > 0
        ? { kind: "happy_path", stepIndex: m.stepIndex - 1 }
        : m
    );
  }, []);

  const next = useCallback(() => {
    setMode((m) =>
      m.kind === "happy_path" && m.stepIndex < HAPPY_PATH_STEPS.length - 1
        ? { kind: "happy_path", stepIndex: m.stepIndex + 1 }
        : m
    );
  }, []);

  const selectScenario = useCallback((key: ScenarioKey) => {
    setMode({ kind: "scenario", key });
  }, []);

  const resetHappyPath = useCallback(() => {
    setMode({ kind: "happy_path", stepIndex: 0 });
  }, []);

  return (
    <FixtureStoreContext.Provider
      value={{
        mode,
        run,
        report,
        stepLabel,
        stepPosition,
        canPrev,
        canNext,
        prev,
        next,
        selectScenario,
        resetHappyPath,
      }}
    >
      {children}
    </FixtureStoreContext.Provider>
  );
}

export function useFixtureStore(): FixtureStore {
  const ctx = useContext(FixtureStoreContext);
  if (!ctx) throw new Error("useFixtureStore must be used within FixtureStoreProvider");
  return ctx;
}
