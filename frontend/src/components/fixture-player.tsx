"use client";

import { useEffect } from "react";
import { ChevronLeft, ChevronRight, Eye, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useFixtureStore } from "@/lib/fixture-store";
import {
  SCENARIO_LABELS,
  type ScenarioKey,
} from "@/lib/fixture-data";

const SCENARIO_KEYS = Object.keys(SCENARIO_LABELS) as ScenarioKey[];

export function FixturePlayer() {
  const {
    mode,
    stepLabel,
    stepPosition,
    canPrev,
    canNext,
    prev,
    next,
    selectScenario,
    resetHappyPath,
  } = useFixtureStore();

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLSelectElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;
      if (e.key === "ArrowLeft") prev();
      if (e.key === "ArrowRight") next();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [prev, next]);

  return (
    <div>
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary shrink-0">
          <Eye className="size-3" />
          Preview
        </span>

        <div className="flex items-center gap-1.5 min-w-0">
          {mode.kind === "happy_path" ? (
            <>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={prev}
                disabled={!canPrev}
                aria-label="Previous step"
              >
                <ChevronLeft className="size-3.5" />
              </Button>
              <span className="text-xs text-foreground font-medium truncate">
                {stepLabel}
              </span>
              <span className="text-xs text-muted-foreground tabular-nums shrink-0">
                {stepPosition}
              </span>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={next}
                disabled={!canNext}
                aria-label="Next step"
              >
                <ChevronRight className="size-3.5" />
              </Button>
            </>
          ) : (
            <>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={resetHappyPath}
                aria-label="Reset to walkthrough"
              >
                <RotateCcw className="size-3.5" />
              </Button>
              <span className="text-xs text-foreground font-medium truncate">
                {stepLabel}
              </span>
            </>
          )}
        </div>

        <div className="ml-auto shrink-0">
          <select
            value={mode.kind === "scenario" ? mode.key : ""}
            onChange={(e) => {
              const val = e.target.value;
              if (val === "") {
                resetHappyPath();
              } else {
                selectScenario(val as ScenarioKey);
              }
            }}
            className="rounded-lg border border-input bg-card px-2.5 py-1 text-xs text-foreground focus:outline-none focus:ring-2 focus:ring-ring/30"
            aria-label="Select scenario"
          >
            <option value="">Walkthrough</option>
            {SCENARIO_KEYS.map((key) => (
              <option key={key} value={key}>
                {SCENARIO_LABELS[key]}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}

export default FixturePlayer;
