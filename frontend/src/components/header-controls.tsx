"use client";

import { useEffect, useState } from "react";
import { useLocale } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";

export function HeaderControls() {
  const { t, locale, setLocale } = useLocale();
  const { mode, setMode } = useWorkspace();
  const [devMode, setDevMode] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("dev") === "1") {
      setDevMode(true);
      // ?run= loads a real backend run in live mode; ?dev=1 alone enters fixture mode
      if (!params.get("run")) {
        setMode("fixture");
      }
    }
  }, [setMode]);

  return (
    <div className="flex items-center gap-2.5">
      {devMode && (
        <button
          onClick={() => setMode(mode === "fixture" ? "live" : "fixture")}
          className="rounded-lg border border-input bg-card px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
        >
          {mode === "fixture" ? "Live" : "Fixture"}
        </button>
      )}

      <div className="flex items-center rounded-lg border border-input bg-card text-xs overflow-hidden" role="radiogroup" aria-label={t("locale.switch_label")}>
        <button
          onClick={() => setLocale("en")}
          role="radio"
          aria-checked={locale === "en"}
          className={`px-2.5 py-1 transition-colors focus-visible:outline-2 focus-visible:outline-ring/50 focus-visible:outline-offset-[-2px] ${
            locale === "en"
              ? "bg-primary text-primary-foreground font-medium"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          EN
        </button>
        <button
          onClick={() => setLocale("vi")}
          role="radio"
          aria-checked={locale === "vi"}
          className={`px-2.5 py-1 transition-colors focus-visible:outline-2 focus-visible:outline-ring/50 focus-visible:outline-offset-[-2px] ${
            locale === "vi"
              ? "bg-primary text-primary-foreground font-medium"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          VI
        </button>
      </div>
    </div>
  );
}
