import { useCallback, useEffect, useRef, useState } from "react";
import * as jobApi from "@/pages/Video/api";
import type { JobEvent } from "@/pages/Video/types";

export default function useVideoJob() {
  const [fetchEvents] = useState<JobEvent[]>([]);
  const [processEvents, setProcessEvents] = useState<JobEvent[]>([]);
  const [processJobId, setProcessJobId] = useState<string | null>(null);
  const processEventSourceRef = useRef<EventSource | null>(null);

  const closeProcessEventSource = useCallback(() => {
    processEventSourceRef.current?.close();
    processEventSourceRef.current = null;
  }, []);

  useEffect(() => closeProcessEventSource, [closeProcessEventSource]);

  const handleFetchLatestVideos = async () => {
    const job = await jobApi.create_fetch_latest_videos_job();
    console.log(job);
  };

  const handleProcessVideos = async () => {
    closeProcessEventSource();
    const job = await jobApi.create_process_videos_job();
    setProcessEvents([
      {
        id: `process-started-${job.id}`,
        type: "status",
        status: "running",
      },
    ]);
    setProcessJobId(job.id);
    const evtSource = new EventSource(job.events_url);
    processEventSourceRef.current = evtSource;

    evtSource.onmessage = (event) => {
      const jobEvent = JSON.parse(event.data) as JobEvent;
      setProcessEvents((events) => [...events, jobEvent]);

      if (
        jobEvent.type === "status" &&
        ["completed", "failed", "cancelled"].includes(jobEvent.status)
      ) {
        closeProcessEventSource();
        setProcessJobId(null);
      }
    };

    evtSource.onerror = () => {
      closeProcessEventSource();
      setProcessJobId(null);
    };
  };

  const handleCancelProcessVideos = async () => {
    if (!processJobId) return;
    const job = await jobApi.cancel_job(processJobId);
    setProcessEvents((events) => [
      ...events,
      {
        id: `cancel-requested-${processJobId}`,
        type: "log",
        message: "Cancel requested.",
      },
      {
        id: `cancel-status-${processJobId}`,
        type: "status",
        status: job.status,
      },
    ]);
  };

  return {
    fetchEvents,
    handleFetchLatestVideos,
    processEvents,
    handleProcessVideos,
    handleCancelProcessVideos,
    isProcessRunning: Boolean(processJobId),
  };
}
