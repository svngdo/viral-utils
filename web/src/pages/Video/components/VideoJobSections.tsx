import { Download, Play, Square } from "lucide-react";
import JobSection from "@/pages/Video/components/JobSection";
import type { JobEvent } from "@/pages/Video/types";

interface VideoJobSectionsProps {
  fetchEvents: JobEvent[];
  processEvents: JobEvent[];
  onFetchLatestVideos: () => void;
  onCancelFetchLatestVideos: () => void;
  onProcessVideos: () => void;
  onCancelProcessVideos: () => void;
  isFetchRunning: boolean;
  isProcessRunning: boolean;
}

export default function VideoJobSections({
  fetchEvents,
  processEvents,
  onFetchLatestVideos,
  onCancelFetchLatestVideos,
  onProcessVideos,
  onCancelProcessVideos,
  isFetchRunning,
  isProcessRunning,
}: VideoJobSectionsProps) {
  return (
    <>
      <JobSection
        title="Fetch latest videos"
        description="Pull newly available videos and stream job events while the count updates."
        actionLabel="Fetch"
        icon={<Download />}
        unit="videos fetched"
        events={fetchEvents}
        onAction={onFetchLatestVideos}
        actionDisabled={isFetchRunning}
        cancelLabel="Cancel"
        cancelIcon={<Square />}
        onCancel={onCancelFetchLatestVideos}
        cancelVisible={isFetchRunning}
      />

      <JobSection
        title="Process videos"
        description="Prepare fetched videos and stream per-video processing events."
        actionLabel="Process"
        icon={<Play />}
        unit="videos processed"
        events={processEvents}
        onAction={onProcessVideos}
        actionDisabled={isProcessRunning}
        cancelLabel="Cancel"
        cancelIcon={<Square />}
        onCancel={onCancelProcessVideos}
        cancelVisible={isProcessRunning}
      />
    </>
  );
}
