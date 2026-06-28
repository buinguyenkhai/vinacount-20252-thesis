"use client";

import { useRef, useEffect } from "react";
import { useWorkspace } from "@/lib/workspace-store";
import { UserMessage } from "./user-message";
import { ConfirmationCard } from "./confirmation-card";
import { ProgressBlock } from "./progress-block";
import { CompletionCard } from "./completion-card";
import { FailedCard } from "./failed-card";
import { CancelledCard } from "./cancelled-card";

export function ThreadContainer() {
  const { thread } = useWorkspace();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [thread]);

  return (
    <div className="max-w-3xl mx-auto w-full flex flex-col gap-4 py-6 px-4 sm:px-6">
      {thread.map((msg, i) => {
        switch (msg.kind) {
          case "user_input":
            return <UserMessage key={i} intent={msg.intent} />;
          case "confirmation":
            return (
              <ConfirmationCard
                key={i}
                intent={msg.intent}
                confirmed={msg.confirmed}
              />
            );
          case "progress":
            return (
              <ProgressBlock
                key={i}
                phases={msg.phases}
                sourceConfirmation={msg.sourceConfirmation}
                sourceConfirmationActive={msg.sourceConfirmationActive}
                allowedActions={msg.allowedActions}
                warnings={msg.warnings}
                elapsedSeconds={msg.elapsedSeconds}
              />
            );
          case "completion":
            return (
              <CompletionCard
                key={i}
                intent={msg.intent}
                finalReport={msg.finalReport}
                elapsedSeconds={msg.elapsedSeconds}
                allowedActions={msg.allowedActions}
              />
            );
          case "failed":
            return (
              <FailedCard
                key={i}
                error={msg.error}
                allowedActions={msg.allowedActions}
                warnings={msg.warnings}
              />
            );
          case "cancelled":
            return <CancelledCard key={i} />;
        }
      })}
      <div ref={bottomRef} />
    </div>
  );
}
