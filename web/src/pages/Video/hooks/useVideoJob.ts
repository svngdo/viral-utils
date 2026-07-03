import { useCallback, useEffect, useRef, useState } from "react";
import * as douyinJobApi from "@/pages/Douyin/api";
import * as jobApi from "@/pages/Video/api";
import type { JobEvent } from "@/pages/Video/types";

export default function useVideoJob() {
  const [fetchEvents, setFetchEvents] = useState<JobEvent[]>([]);
  const [processEvents, setProcessEvents] = useState<JobEvent[]>([]);
  const [fetchJobId, setFetchJobId] = useState<string | null>(null);
  const [processJobId, setProcessJobId] = useState<string | null>(null);
  const fetchEventSourceRef = useRef<EventSource | null>(null);
  const processEventSourceRef = useRef<EventSource | null>(null);

  const closeFetchEventSource = useCallback(() => {
    fetchEventSourceRef.current?.close();
    fetchEventSourceRef.current = null;
  }, []);

  const closeProcessEventSource = useCallback(() => {
    processEventSourceRef.current?.close();
    processEventSourceRef.current = null;
  }, []);

  useEffect(
    () => () => {
      closeFetchEventSource();
      closeProcessEventSource();
    },
    [closeFetchEventSource, closeProcessEventSource],
  );

  const handleFetchLatestVideos = async () => {
    closeFetchEventSource();
    const job = await douyinJobApi.createFetchActiveUsersJob();
    setFetchEvents([
      {
        id: `fetch-started-${job.id}`,
        type: "status",
        status: "running",
      },
    ]);
    setFetchJobId(job.id);
    const evtSource = new EventSource(job.events_url);
    fetchEventSourceRef.current = evtSource;

    evtSource.onmessage = (event) => {
      const jobEvent = JSON.parse(event.data) as JobEvent;
      setFetchEvents((events) => [...events, jobEvent]);

      if (
        jobEvent.type === "status" &&
        ["completed", "failed", "cancelled"].includes(jobEvent.status)
      ) {
        closeFetchEventSource();
        setFetchJobId(null);
      }
    };

    evtSource.onerror = () => {
      closeFetchEventSource();
      setFetchJobId(null);
    };
  };

  const handleProcessVideos = async () => {
    closeProcessEventSource();
    const job = await jobApi.createProcessVideosJob();
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
    const job = await jobApi.cancelJob(processJobId);
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

  const handleCancelFetchLatestVideos = async () => {
    if (!fetchJobId) return;
    const job = await jobApi.cancelJob(fetchJobId);
    setFetchEvents((events) => [
      ...events,
      {
        id: `cancel-requested-${fetchJobId}`,
        type: "log",
        message: "Cancel requested.",
      },
      {
        id: `cancel-status-${fetchJobId}`,
        type: "status",
        status: job.status,
      },
    ]);
  };

  return {
    fetchEvents,
    handleFetchLatestVideos,
    handleCancelFetchLatestVideos,
    processEvents,
    handleProcessVideos,
    handleCancelProcessVideos,
    isFetchRunning: Boolean(fetchJobId),
    isProcessRunning: Boolean(processJobId),
  };
}
