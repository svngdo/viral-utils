import type { JobStatus } from "@/pages/Video/types";

export const statusLabels: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running",
  cancelling: "Cancelling",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};
