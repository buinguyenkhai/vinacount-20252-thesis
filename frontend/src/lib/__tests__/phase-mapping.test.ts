import { describe, it, expect } from "vitest";
import { mapStagesToPhases, STAGE_TO_PHASE, PHASE_ORDER } from "../phase-mapping";
import type { Stage, StageId } from "@/types/runtime";
import { STAGE_ORDER } from "@/types/runtime";

function makeStage(
  id: StageId,
  status: Stage["status"],
  overrides?: Partial<Stage>
): Stage {
  return {
    stage_id: id,
    status,
    started_at: status !== "pending" ? "2026-06-17T09:00:00Z" : null,
    completed_at: status === "completed" ? "2026-06-17T09:01:00Z" : null,
    summary: null,
    progress: null,
    counts: null,
    warnings: [],
    ...overrides,
  };
}

function makeStages(activeStage?: StageId): Stage[] {
  return STAGE_ORDER.map((id) => {
    if (!activeStage) return makeStage(id, "pending");
    const activeIndex = STAGE_ORDER.indexOf(activeStage);
    const thisIndex = STAGE_ORDER.indexOf(id);
    if (thisIndex < activeIndex) return makeStage(id, "completed");
    if (thisIndex === activeIndex) return makeStage(id, "active");
    return makeStage(id, "pending");
  });
}

describe("STAGE_TO_PHASE", () => {
  it("maps all 8 stages to 4 phases", () => {
    expect(STAGE_TO_PHASE["source_discovery"]).toBe("finding_sources");
    expect(STAGE_TO_PHASE["source_confirmation"]).toBe("confirming_sources");
    expect(STAGE_TO_PHASE["cache_lookup"]).toBe("analyzing");
    expect(STAGE_TO_PHASE["extraction"]).toBe("analyzing");
    expect(STAGE_TO_PHASE["tool_analysis"]).toBe("analyzing");
    expect(STAGE_TO_PHASE["detector_assessment"]).toBe("analyzing");
    expect(STAGE_TO_PHASE["aggregation"]).toBe("analyzing");
    expect(STAGE_TO_PHASE["report_generation"]).toBe("generating_report");
  });
});

describe("mapStagesToPhases", () => {
  it("returns 4 phases", () => {
    const phases = mapStagesToPhases(makeStages(), null);
    expect(phases).toHaveLength(4);
    expect(phases.map((p) => p.id)).toEqual(PHASE_ORDER);
  });

  it("all pending when no stage is active", () => {
    const phases = mapStagesToPhases(makeStages(), null);
    expect(phases.every((p) => p.status === "pending")).toBe(true);
  });

  it("finding_sources is active when source_discovery is active", () => {
    const phases = mapStagesToPhases(
      makeStages("source_discovery"),
      "source_discovery"
    );
    expect(phases[0].status).toBe("active");
    expect(phases[1].status).toBe("pending");
    expect(phases[2].status).toBe("pending");
    expect(phases[3].status).toBe("pending");
  });

  it("confirming_sources is active when source_confirmation is active", () => {
    const phases = mapStagesToPhases(
      makeStages("source_confirmation"),
      "source_confirmation"
    );
    expect(phases[0].status).toBe("completed");
    expect(phases[1].status).toBe("active");
    expect(phases[2].status).toBe("pending");
  });

  it("analyzing is active when extraction is active", () => {
    const phases = mapStagesToPhases(
      makeStages("extraction"),
      "extraction"
    );
    expect(phases[0].status).toBe("completed");
    expect(phases[1].status).toBe("completed");
    expect(phases[2].status).toBe("active");
    expect(phases[3].status).toBe("pending");
  });

  it("analyzing stays active through detector_assessment", () => {
    const phases = mapStagesToPhases(
      makeStages("detector_assessment"),
      "detector_assessment"
    );
    expect(phases[2].id).toBe("analyzing");
    expect(phases[2].status).toBe("active");
  });

  it("generating_report is active when report_generation is active", () => {
    const phases = mapStagesToPhases(
      makeStages("report_generation"),
      "report_generation"
    );
    expect(phases[2].status).toBe("completed");
    expect(phases[3].status).toBe("active");
  });

  it("all completed when all stages completed", () => {
    const stages = STAGE_ORDER.map((id) => makeStage(id, "completed"));
    const phases = mapStagesToPhases(stages, null);
    expect(phases.every((p) => p.status === "completed")).toBe(true);
  });

  it("propagates summary from active stage", () => {
    const stages = makeStages("detector_assessment");
    stages[5] = makeStage("detector_assessment", "active", {
      summary: "Assessing 7 of 12 packets",
    });
    const phases = mapStagesToPhases(stages, "detector_assessment");
    expect(phases[2].summary).toBe("Assessing 7 of 12 packets");
  });

  it("propagates progress from active stage", () => {
    const stages = makeStages("detector_assessment");
    stages[5] = makeStage("detector_assessment", "active", {
      progress: { processed: 7, total: 12 },
    });
    const phases = mapStagesToPhases(stages, "detector_assessment");
    expect(phases[2].progress).toEqual({ processed: 7, total: 12 });
  });

  it("analyzing phase tracks completed sub-stages", () => {
    const stages = makeStages("tool_analysis");
    const phases = mapStagesToPhases(stages, "tool_analysis");
    expect(phases[2].stageCount).toBe(5);
    expect(phases[2].completedStageCount).toBe(2);
    expect(phases[2].activeStageId).toBe("tool_analysis");
    expect(phases[2].activeStageOrdinal).toBe(3);
  });

  it("handles failed stage", () => {
    const stages = makeStages("detector_assessment");
    stages[5] = makeStage("detector_assessment", "failed");
    const phases = mapStagesToPhases(stages, null);
    expect(phases[2].status).toBe("failed");
  });

  it("analyzing phase completes when extraction is skipped (cache hit)", () => {
    const stages: Stage[] = STAGE_ORDER.map((id) => {
      if (id === "extraction") return makeStage(id, "skipped");
      return makeStage(id, "completed");
    });
    const phases = mapStagesToPhases(stages, null);
    expect(phases[2].id).toBe("analyzing");
    expect(phases[2].status).toBe("completed");
  });

  it("analyzing phase tracks ordinals correctly when extraction is skipped", () => {
    const stages: Stage[] = [
      makeStage("source_discovery", "completed"),
      makeStage("source_confirmation", "completed"),
      makeStage("cache_lookup", "completed"),
      makeStage("extraction", "skipped"),
      makeStage("tool_analysis", "active"),
      makeStage("detector_assessment", "pending"),
      makeStage("aggregation", "pending"),
      makeStage("report_generation", "pending"),
    ];
    const phases = mapStagesToPhases(stages, "tool_analysis");
    expect(phases[2].status).toBe("active");
    expect(phases[2].activeStageId).toBe("tool_analysis");
  });
});
