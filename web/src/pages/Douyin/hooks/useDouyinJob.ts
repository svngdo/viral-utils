import { useCallback, useEffect, useRef, useState } from "react";
import { apiAbsoluteUrl } from "@/lib/api";
import * as jobApi from "@/pages/Video/api";
import type { JobCreateResponse, JobEvent, JobStatus } from "@/pages/Video/types";

type JobKind = "sync" | "download";
type JobScope = "active" | "selected";

interface UseDouyinJobOptions {
  completeLogMessage: string;
  defaultStartErrorMessage: string;
  incompleteErrorMessage: string;
  kind: JobKind;
  pollErrorMessage: string;
  refreshErrorMessage: string;
  runningLogMessage: string;
  terminalLogPrefix: string;
  onCompleted: () => void | Promise<void>;
  onError: (message: string) => void;
  onLatestKindChange: (kind: JobKind) => void;
}

const TERMINAL_STATUSES: JobStatus[] = ["completed", "failed", "cancelled"];

export default function useDouyinJob({
  completeLogMessage,
  defaultStartErrorMessage,
  incompleteErrorMessage,
  kind,
  pollErrorMessage,
  refreshErrorMessage,
  runningLogMessage,
  terminalLogPrefix,
  onCompleted,
  onError,
  onLatestKindChange,
}: UseDouyinJobOptions) {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [scope, setScope] = useState<JobScope | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const finishedJobIdRef = useRef<string | null>(null);

  const closeEventSource = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }, []);

  useEffect(() => closeEventSource, [closeEventSource]);

  const appendLog = useCallback((id: string, message: string) => {
    setEvents((previous) => [...previous, { id, type: "log", message }]);
  }, []);

  const finishJob = useCallback(
    async (finishedJobId: string, status: JobStatus) => {
      if (finishedJobIdRef.current === finishedJobId) return;
      finishedJobIdRef.current = finishedJobId;
      closeEventSource();
      setJobId(null);
      setScope(null);

      if (status === "completed") {
        appendLog(`complete-${finishedJobId}`, completeLogMessage);
        try {
          await onCompleted();
        } catch {
          onError(refreshErrorMessage);
        }
      } else if (status === "failed") {
        onError(incompleteErrorMessage);
        appendLog(`failed-${finishedJobId}`, `${terminalLogPrefix} failed`);
      } else {
        appendLog(`cancelled-${finishedJobId}`, `${terminalLogPrefix} cancelled`);
      }
    },
    [
      appendLog,
      closeEventSource,
      completeLogMessage,
      incompleteErrorMessage,
      onCompleted,
      onError,
      refreshErrorMessage,
      terminalLogPrefix,
    ],
  );

  const startJob = useCallback(
    async (nextScope: JobScope, createJob: () => Promise<JobCreateResponse>) => {
      closeEventSource();
      finishedJobIdRef.current = null;
      onLatestKindChange(kind);
      try {
        const job = await createJob();
        setJobId(job.id);
        setScope(nextScope);
        setEvents([
          { id: `running-${job.id}`, type: "log", message: runningLogMessage },
          { id: `status-${job.id}`, type: "status", status: "running" },
        ]);

        const eventSource = new EventSource(apiAbsoluteUrl(job.events_url));
        eventSourceRef.current = eventSource;
        eventSource.onmessage = (event) => {
          const jobEvent = JSON.parse(event.data) as JobEvent;
          setEvents((previous) => [...previous, jobEvent]);
          if (jobEvent.type === "status" && TERMINAL_STATUSES.includes(jobEvent.status)) {
            void finishJob(job.id, jobEvent.status);
          }
        };
        eventSource.onerror = () => {
          closeEventSource();
          void jobApi
            .getJob(job.id)
            .then((currentJob) => {
              if (TERMINAL_STATUSES.includes(currentJob.status)) {
                return finishJob(job.id, currentJob.status);
              }
              onError(incompleteErrorMessage);
              setJobId(null);
              setScope(null);
            })
            .catch(() => {
              onError(pollErrorMessage);
              setJobId(null);
              setScope(null);
            });
        };
      } catch (error) {
        const message = error instanceof Error ? error.message : defaultStartErrorMessage;
        onError(message);
        setJobId(null);
        setScope(null);
      }
    },
    [
      closeEventSource,
      defaultStartErrorMessage,
      finishJob,
      incompleteErrorMessage,
      kind,
      onError,
      onLatestKindChange,
      pollErrorMessage,
      runningLogMessage,
    ],
  );

  const cancelJob = useCallback(async () => {
    if (!jobId) return;
    try {
      const job = await jobApi.cancelJob(jobId);
      setEvents((previous) => [
        ...previous,
        { id: `cancel-requested-${jobId}`, type: "log", message: "Cancel requested." },
        { id: `cancel-status-${jobId}`, type: "status", status: job.status },
      ]);
      if (TERMINAL_STATUSES.includes(job.status)) await finishJob(jobId, job.status);
    } catch (error) {
      onError(error instanceof Error ? error.message : `Could not cancel ${kind} job`);
    }
  }, [finishJob, jobId, kind, onError]);

  return {
    cancelJob,
    events,
    isRunning: Boolean(jobId),
    scope,
    startJob,
  };
}
