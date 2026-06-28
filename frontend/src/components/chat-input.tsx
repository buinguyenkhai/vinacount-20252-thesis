"use client";

import { useState, type FormEvent } from "react";
import { ArrowUp, Loader2 } from "lucide-react";
import { useLocale } from "@/lib/i18n";
import { useWorkspace, MODEL_OPTIONS } from "@/lib/workspace-store";
import { useLiveRun } from "@/lib/live-run-store";
import { useFixtureStore } from "@/lib/fixture-store";
import { parseFilingIntent, type ParsedFields } from "@/lib/filing-intent-parser";

interface ChatInputProps {
  onDraft?: (fields: ParsedFields, text: string) => void;
  hasDraft?: boolean;
}

export function ChatInput({ onDraft, hasDraft = false }: ChatInputProps) {
  const { t } = useLocale();
  const { mode, run, modelId, modelUserSelected, setModelId } = useWorkspace();
  const liveRun = useLiveRun();
  const fixture = useFixtureStore();
  const [value, setValue] = useState("");
  const [parseError, setParseError] = useState<string | null>(null);

  const status = run?.status;
  const isRunning =
    status === "discovering_sources" || status === "analyzing";
  const isAwaitingConfirmation = status === "awaiting_source_confirmation";
  const isSubmitting = mode === "live" && liveRun.phase === "submitting";
  const disabled = isRunning || isAwaitingConfirmation || isSubmitting || hasDraft;

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (disabled || !value.trim()) return;
    setParseError(null);

    if (mode === "fixture") {
      fixture.resetHappyPath();
      setValue("");
      return;
    }

    const result = parseFilingIntent(value);
    if (!result.ok) {
      setParseError(t("input.parse_error"));
      return;
    }

    if (onDraft) {
      onDraft(result.fields, value);
      setValue("");
      return;
    }
  }

  const placeholder = disabled
    ? isSubmitting
      ? t("loading.submitting")
      : isAwaitingConfirmation
        ? t("input.disabled_confirm")
        : hasDraft
          ? t("draft.title")
          : t("input.disabled_running")
    : t("input.placeholder");

  return (
    <form onSubmit={handleSubmit}>
      <div className="rounded-2xl border border-input bg-card shadow-sm overflow-hidden transition-colors focus-within:ring-2 focus-within:ring-ring/30 focus-within:border-ring">
        <input
          type="text"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            if (parseError) setParseError(null);
          }}
          placeholder={placeholder}
          disabled={disabled}
          autoComplete="off"
          aria-label={t("input.placeholder")}
          className="w-full bg-transparent px-4 pt-3.5 pb-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div className="flex items-center justify-between px-3 pb-2.5">
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            disabled={disabled}
            className="rounded-lg bg-muted px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring/30 transition-colors disabled:opacity-50 cursor-pointer"
            aria-label={t("model.aria_label")}
            title={t("model.tooltip")}
          >
            {MODEL_OPTIONS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          <button
            type="submit"
            disabled={disabled || !value.trim()}
            className="flex items-center justify-center size-8 rounded-xl bg-primary text-primary-foreground transition-opacity disabled:opacity-30 focus-visible:ring-2 focus-visible:ring-ring/50"
            aria-label={t("input.submit")}
          >
            {isSubmitting ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <ArrowUp className="size-4" />
            )}
          </button>
        </div>
      </div>
      {parseError && (
        <p className="mt-2 text-xs text-destructive px-1">
          {parseError}
        </p>
      )}
    </form>
  );
}
