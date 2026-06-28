import type { Stage, StageId, StageStatus } from "@/types/runtime";

export type PhaseId =
  | "finding_sources"
  | "confirming_sources"
  | "analyzing"
  | "generating_report";

export const PHASE_ORDER: readonly PhaseId[] = [
  "finding_sources",
  "confirming_sources",
  "analyzing",
  "generating_report",
];

export const STAGE_TO_PHASE: Record<StageId, PhaseId> = {
  source_discovery: "finding_sources",
  source_confirmation: "confirming_sources",
  cache_lookup: "analyzing",
  extraction: "analyzing",
  tool_analysis: "analyzing",
  detector_assessment: "analyzing",
  aggregation: "analyzing",
  report_generation: "generating_report",
};

const ANALYZING_STAGE_ORDER: readonly StageId[] = [
  "cache_lookup",
  "extraction",
  "tool_analysis",
  "detector_assessment",
  "aggregation",
];

export interface PhaseState {
  id: PhaseId;
  status: StageStatus;
  summary: string | null;
  progress: { processed: number; total: number } | null;
  stageCount: number;
  completedStageCount: number;
  activeStageId: StageId | null;
  activeStageOrdinal: number | null;
}

function derivePhaseStatus(stages: Stage[]): StageStatus {
  if (stages.some((s) => s.status === "failed")) return "failed";
  if (stages.some((s) => s.status === "active")) return "active";
  if (stages.every((s) => s.status === "completed")) return "completed";
  if (stages.every((s) => s.status === "skipped")) return "skipped";
  if (stages.every((s) => s.status === "cancelled")) return "cancelled";
  if (stages.some((s) => s.status === "completed" || s.status === "skipped")) {
    const remaining = stages.filter(
      (s) => s.status !== "completed" && s.status !== "skipped"
    );
    if (remaining.every((s) => s.status === "pending")) return "completed";
  }
  return "pending";
}

export function mapStagesToPhases(
  stages: Stage[],
  currentStage: StageId | null
): PhaseState[] {
  const stageMap = new Map(stages.map((s) => [s.stage_id, s]));

  return PHASE_ORDER.map((phaseId) => {
    let phaseStages: Stage[];

    if (phaseId === "finding_sources") {
      phaseStages = [stageMap.get("source_discovery")].filter(Boolean) as Stage[];
    } else if (phaseId === "confirming_sources") {
      phaseStages = [stageMap.get("source_confirmation")].filter(Boolean) as Stage[];
    } else if (phaseId === "analyzing") {
      phaseStages = ANALYZING_STAGE_ORDER.map((id) => stageMap.get(id)).filter(
        Boolean
      ) as Stage[];
    } else {
      phaseStages = [stageMap.get("report_generation")].filter(Boolean) as Stage[];
    }

    if (phaseStages.length === 0) {
      return {
        id: phaseId,
        status: "pending" as StageStatus,
        summary: null,
        progress: null,
        stageCount: 0,
        completedStageCount: 0,
        activeStageId: null,
        activeStageOrdinal: null,
      };
    }

    const activeStage = currentStage ? stageMap.get(currentStage) : null;
    const activePhaseForCurrent = currentStage
      ? STAGE_TO_PHASE[currentStage]
      : null;

    const summary =
      activePhaseForCurrent === phaseId && activeStage
        ? activeStage.summary
        : phaseStages.find((s) => s.status === "completed" && s.summary)
            ?.summary ?? null;

    const progressStage =
      activePhaseForCurrent === phaseId && activeStage?.progress
        ? activeStage
        : null;
    const activeStageOrdinal =
      activePhaseForCurrent === phaseId && activeStage
        ? phaseStages.findIndex((stage) => stage.stage_id === activeStage.stage_id) + 1
        : null;

    return {
      id: phaseId,
      status: derivePhaseStatus(phaseStages),
      summary,
      progress: progressStage?.progress ?? null,
      stageCount: phaseStages.length,
      completedStageCount: phaseStages.filter((s) => s.status === "completed")
        .length,
      activeStageId: activePhaseForCurrent === phaseId ? currentStage : null,
      activeStageOrdinal: activeStageOrdinal && activeStageOrdinal > 0 ? activeStageOrdinal : null,
    };
  });
}
