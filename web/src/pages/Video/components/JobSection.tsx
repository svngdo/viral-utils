import { type ReactNode, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { statusLabels } from "@/pages/Video/display";
import type { JobEvent } from "@/pages/Video/types";

interface JobSectionProps {
  title: string;
  description: string;
  actionLabel: string;
  icon: ReactNode;
  unit: string;
  events: JobEvent[];
  onAction: () => void;
  actionDisabled?: boolean;
  cancelLabel?: string;
  cancelIcon?: ReactNode;
  onCancel?: () => void;
  cancelVisible?: boolean;
}

export default function JobSection({
  title,
  description,
  actionLabel,
  icon,
  unit,
  events,
  onAction,
  actionDisabled,
  cancelLabel,
  cancelIcon,
  onCancel,
  cancelVisible,
}: JobSectionProps) {
  const activityLogRef = useRef<HTMLDivElement | null>(null);
  const latestStatusEvent = [...events].reverse().find((event) => event.type === "status");
  const latestProgressEvent = [...events].reverse().find((event) => event.type === "progress");
  const activityEvents = events.filter((event) => event.type === "log");
  const hasActivity = events.length > 0 || actionDisabled;

  const status = latestStatusEvent?.status ?? "queued";
  const done = latestProgressEvent?.done ?? 0;
  const total = latestProgressEvent?.total ?? undefined;
  const hasKnownTotal = typeof total === "number" && total > 0;
  const progress = hasKnownTotal ? Math.round((done / total) * 100) : 0;
  const countLabel = hasKnownTotal ? `${done} / ${total} ${unit}` : `${done} ${unit}`;

  useEffect(() => {
    const activityLog = activityLogRef.current;
    if (!activityLog) return;

    activityLog.scrollTop = activityLog.scrollHeight;
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction>
          <div className="flex items-center gap-2">
            {cancelVisible && onCancel && cancelLabel ? (
              <Button type="button" variant="outline" onClick={onCancel}>
                {cancelIcon}
                {cancelLabel}
              </Button>
            ) : null}
            <Button onClick={onAction} disabled={actionDisabled}>
              {icon}
              {actionLabel}
            </Button>
          </div>
        </CardAction>
      </CardHeader>
      {hasActivity && (
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
              <div className="flex items-center gap-2">
                <span className="rounded-md border bg-muted/40 px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  {statusLabels[status]}
                </span>
                <span className="font-medium text-foreground">{countLabel}</span>
              </div>
              <span className="tabular-nums text-muted-foreground">
                {hasKnownTotal ? `${progress}%` : "Total pending"}
              </span>
            </div>
            <Progress value={progress} />
          </div>

          <div className="rounded-lg border bg-muted/20">
            <div className="border-b px-3 py-2 text-xs font-medium uppercase text-muted-foreground">
              RUN LOG
            </div>
            <div
              ref={activityLogRef}
              className="max-h-44 space-y-2 overflow-y-auto p-3 font-mono text-xs"
            >
              {activityEvents.map((event) => (
                <div key={event.id} className="grid grid-cols-[3.2rem_1fr] gap-2">
                  <span className="uppercase text-muted-foreground">{event.type}</span>
                  <span className="text-foreground">{event.message}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
