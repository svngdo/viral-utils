import { ScrollText } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import type { JobEvent, LogEvent } from "@/pages/Video/types";

interface DouyinJobToolbarStatusProps {
  syncEvents: JobEvent[];
  downloadEvents: JobEvent[];
}

interface JobSummary {
  kind: "sync" | "download";
  title: string;
  events: JobEvent[];
}

// Renders only the log events for one job stream.
function JobLogSection({ summary }: { summary: JobSummary }) {
  const logs = summary.events.filter((event): event is LogEvent => event.type === "log");
  if (!logs.length) return null;

  return (
    <div className="space-y-2">
      <div className="text-xs font-medium uppercase text-muted-foreground">{summary.title}</div>
      <div className="space-y-1 font-mono text-xs">
        {logs.map((event) => (
          <div key={event.id} className="text-foreground">
            {event.message}
          </div>
        ))}
      </div>
    </div>
  );
}

// Provides a compact popover for recent sync and download logs.
export default function DouyinJobToolbarStatus({
  syncEvents,
  downloadEvents,
}: DouyinJobToolbarStatusProps) {
  const [logOpen, setLogOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const jobs = useMemo<JobSummary[]>(
    () => [
      {
        kind: "sync",
        title: "Sync users",
        events: syncEvents,
      },
      {
        kind: "download",
        title: "Download videos",
        events: downloadEvents,
      },
    ],
    [downloadEvents, syncEvents],
  );
  const visibleJobs = jobs.filter((job) => job.events.length > 0);
  const logCount = visibleJobs.reduce(
    (count, job) => count + job.events.filter((event) => event.type === "log").length,
    0,
  );

  useEffect(() => {
    if (!logOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setLogOpen(false);
      }
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [logOpen]);

  useEffect(() => {
    if (!logOpen || !logCount) return;
    const log = logRef.current;
    if (!log) return;
    log.scrollTop = log.scrollHeight;
  }, [logCount, logOpen]);

  if (!logCount) return null;

  return (
    <div ref={menuRef} className="relative flex items-center">
      <Button
        type="button"
        variant="outline"
        size="sm"
        aria-expanded={logOpen}
        onClick={() => setLogOpen((open) => !open)}
      >
        <ScrollText />
        Run log
      </Button>

      {logOpen && (
        <div className="absolute top-10 right-0 z-20 w-[min(28rem,calc(100vw-2rem))] rounded-lg border bg-popover p-3 text-popover-foreground shadow-md">
          <div ref={logRef} className="max-h-72 space-y-4 overflow-y-auto">
            {visibleJobs.map((job) => (
              <JobLogSection key={job.kind} summary={job} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
