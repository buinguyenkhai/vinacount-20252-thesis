"use client";

import { Plus } from "lucide-react";
import { useLocale } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";
import { useLiveRun } from "@/lib/live-run-store";
import { useFixtureStore } from "@/lib/fixture-store";
import { HeaderControls } from "@/components/header-controls";
import { Button } from "@/components/ui/button";

export function Header() {
  const { t } = useLocale();
  const { mode, run, closeReportReader } = useWorkspace();
  const liveRun = useLiveRun();
  const fixture = useFixtureStore();

  const hasRun =
    run !== null && (run.status !== "created" || run.stages.length > 0);

  async function handleNewAnalysis() {
    if (mode === "live") {
      const canStop = run?.allowed_actions?.some(
        (a) => a.action === "stop_run",
      );
      if (canStop) {
        await liveRun.stopRun();
      }
      liveRun.clearRun();
    } else {
      fixture.resetHappyPath();
    }
    closeReportReader();
  }

  return (
    <header className="border-b border-border bg-background shrink-0">
      <div className="max-w-3xl mx-auto px-6 py-4 flex items-center">
        <div className="flex items-center gap-2.5">
          <span className="text-base font-bold tracking-tight text-primary">
            {t("header.brand")}
          </span>
        </div>
        <div className="ml-auto flex items-center gap-3">
          {hasRun && (
            <Button size="sm" variant="outline" onClick={handleNewAnalysis}>
              <Plus className="size-3.5" data-icon="inline-start" />
              {t("header.new_analysis")}
            </Button>
          )}
          <HeaderControls />
        </div>
      </div>
    </header>
  );
}
