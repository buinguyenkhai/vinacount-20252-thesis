// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";
import { describe, expect, it, vi } from "vitest";
import { WorkspaceLayout } from "../workspace-layout";

const mocks = vi.hoisted(() => ({
  loadSealedDemoRun: vi.fn(),
  submitFilingIntent: vi.fn(),
}));

vi.mock("@/lib/workspace-store", () => ({
  useWorkspace: () => ({
    mode: "live",
    run: null,
    modelId: "deepseek-v4-flash",
    modelUserSelected: false,
    reportReaderOpen: false,
  }),
}));

vi.mock("@/lib/live-run-store", () => ({
  useLiveRun: () => ({
    phase: "idle",
    run: null,
    report: null,
    validationError: null,
    error: null,
    actionPending: false,
    submitFilingIntent: mocks.submitFilingIntent,
    loadSealedDemoRun: mocks.loadSealedDemoRun,
    confirmSources: vi.fn(),
    rejectSource: vi.fn(),
    retrySourceDiscovery: vi.fn(),
    selectSourceCandidate: vi.fn(),
    resumeRun: vi.fn(),
    stopRun: vi.fn(),
    clearRun: vi.fn(),
    clearValidationError: vi.fn(),
  }),
}));

vi.mock("@/lib/fixture-store", () => ({
  useFixtureStore: () => ({
    resetHappyPath: vi.fn(),
  }),
}));

vi.mock("@/lib/i18n", () => ({
  useLocale: () => ({
    locale: "vi",
    t: (key: string) =>
      ({
        "empty.title": "Vinacount",
        "empty.subtitle": "Runtime analysis",
        "empty.method_note": "Cached-first runtime path",
        "empty.example_prefix": "Thử:",
      })[key] ?? key,
  }),
}));

vi.mock("../thread/thread-container", () => ({
  ThreadContainer: () => <div />,
}));

vi.mock("../chat-input", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock("../report-panel", () => ({
  ReportReader: () => <div />,
}));

vi.mock("../evidence-pane", () => ({
  EvidencePane: () => <div />,
}));

vi.mock("../fixture-player", () => ({
  FixturePlayer: () => <div />,
}));

vi.mock("../validation-banner", () => ({
  ValidationBanner: () => <div />,
}));

vi.mock("../filing-intent-draft", () => ({
  FilingIntentDraft: () => <div />,
}));

describe("WorkspaceLayout sealed replay examples", () => {
  it("loads the locked NKG run instead of creating a live run", () => {
    render(<WorkspaceLayout />);

    fireEvent.click(screen.getByRole("button", { name: "NKG Q3 2021 hợp nhất" }));

    expect(mocks.loadSealedDemoRun).toHaveBeenCalledWith(
      "run_qaqc_nkg_2021_q3_consolidated_report_memory_reuse",
    );
    expect(mocks.submitFilingIntent).not.toHaveBeenCalled();
  });
});
