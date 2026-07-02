export type JobEventType = "status" | "progress" | "log";

export type JobStatus = "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled";

export interface StatusEvent {
  id: string;
  type: "status";
  status: JobStatus;
}

export interface ProgressEvent {
  id: string;
  type: "progress";
  done: number;
  total: number;
}

export interface LogEvent {
  id: string;
  type: "log";
  message: string;
}

export type JobEvent = StatusEvent | ProgressEvent | LogEvent;

export interface JobCreateResponse {
  id: string;
  events_url: string;
}

export interface JobCancelResponse {
  id: string;
  status: JobStatus;
}

export interface JobResponse {
  id: string;
  status: JobStatus;
  error: string | null;
}
