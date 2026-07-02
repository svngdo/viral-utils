import { apiFetch } from "@/lib/api";
import type { JobCancelResponse, JobCreateResponse, JobResponse } from "@/pages/Video/types";

export const create_fetch_latest_videos_job = (): Promise<JobCreateResponse> =>
  apiFetch<JobCreateResponse>("/jobs/fetch-latest-videos", {
    method: "POST",
  });

export const create_process_videos_job = (): Promise<JobCreateResponse> =>
  apiFetch<JobCreateResponse>("/jobs/process-videos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

export const cancel_job = (jobId: string): Promise<JobCancelResponse> =>
  apiFetch<JobCancelResponse>(`/jobs/${jobId}/cancel`, {
    method: "POST",
  });

export const get_job = (jobId: string): Promise<JobResponse> =>
  apiFetch<JobResponse>(`/jobs/${jobId}`);
