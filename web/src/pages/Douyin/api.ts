import { apiFetch } from "@/lib/api";
import type {
  DouyinUser,
  DouyinUserCreate,
  DouyinUserStatus,
  DouyinUserUpdate,
  DouyinVideo,
  DouyinVideoPage,
  DouyinVideoUpdate,
} from "@/pages/Douyin/types";
import type { JobCreateResponse } from "@/pages/Video/types";

export const getUserStatuses = (): Promise<DouyinUserStatus[]> =>
  apiFetch<DouyinUserStatus[]>("/douyin/user-statuses");

export const getUsers = (): Promise<DouyinUser[]> => apiFetch<DouyinUser[]>("/douyin/users");

export const createUser = (data: DouyinUserCreate): Promise<DouyinUser> =>
  apiFetch<DouyinUser>("/douyin/users", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const getUserById = (id: number): Promise<DouyinUser> =>
  apiFetch<DouyinUser>(`/douyin/users/${id}`);

export const updateUser = (id: number, data: DouyinUserUpdate): Promise<DouyinUser> =>
  apiFetch<DouyinUser>(`/douyin/users/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const removeUser = (id: number): Promise<void> =>
  apiFetch<void>(`/douyin/users/${id}`, { method: "DELETE" });

export const createFetchActiveUsersJob = (): Promise<JobCreateResponse> =>
  apiFetch<JobCreateResponse>("/jobs/douyin/fetch-latest-videos", {
    method: "POST",
  });

export const createFetchUserVideosJob = (userId: number): Promise<JobCreateResponse> =>
  apiFetch<JobCreateResponse>(`/jobs/douyin/users/${userId}/fetch-videos`, {
    method: "POST",
  });

export const createFetchSelectedUsersJob = (userIds: number[]): Promise<JobCreateResponse> =>
  apiFetch<JobCreateResponse>("/jobs/douyin/users/fetch-videos", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_ids: userIds }),
  });

export const getVideoPage = ({
  limit,
  offset,
}: {
  limit: number;
  offset: number;
}): Promise<DouyinVideoPage> => {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return apiFetch<DouyinVideoPage>(`/douyin/videos/page?${params}`);
};

export const updateVideo = (id: number, data: DouyinVideoUpdate): Promise<DouyinVideo> =>
  apiFetch<DouyinVideo>(`/douyin/videos/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

export const removeVideo = (id: number): Promise<void> =>
  apiFetch<void>(`/douyin/videos/${id}`, { method: "DELETE" });
