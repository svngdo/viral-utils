import { apiFetch } from "@/lib/api";
import type { JobCancelResponse, JobCreateResponse, JobResponse } from "@/pages/Video/types";

export const createProcessVideosJob = (): Promise<JobCreateResponse> =>
  apiFetch<JobCreateResponse>("/jobs/video/process-videos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

export const cancelJob = (jobId: string): Promise<JobCancelResponse> =>
  apiFetch<JobCancelResponse>(`/jobs/${jobId}/cancel`, {
    method: "POST",
  });

export const getJob = (jobId: string): Promise<JobResponse> =>
  apiFetch<JobResponse>(`/jobs/${jobId}`);
