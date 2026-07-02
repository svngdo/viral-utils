import { Download, Play, Square } from "lucide-react";
import JobSection from "@/pages/Video/components/JobSection";
import type { JobEvent } from "@/pages/Video/types";

interface VideoJobSectionsProps {
  fetchEvents: JobEvent[];
  processEvents: JobEvent[];
  onFetchLatestVideos: () => void;
  onProcessVideos: () => void;
  onCancelProcessVideos: () => void;
  isProcessRunning: boolean;
}

export default function VideoJobSections({
  fetchEvents,
  processEvents,
  onFetchLatestVideos,
  onProcessVideos,
  onCancelProcessVideos,
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
