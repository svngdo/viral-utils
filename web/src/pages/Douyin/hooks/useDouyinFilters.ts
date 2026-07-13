import { useMemo } from "react";
import {
  getUserDisplayName,
  getUserSecondaryName,
  getVideoDisplayTitle,
  getVideoSecondaryTitle,
} from "@/pages/Douyin/display";
import { formatDateTime } from "@/pages/Douyin/format";
import type { DouyinUser, DouyinUserStatus, DouyinVideo } from "@/pages/Douyin/types";
import type { System } from "@/pages/Systems/types";

function normalizeSearch(value: string) {
  return value.trim().toLowerCase();
}

function matchesSearch(values: unknown[], query: string) {
  if (!query) return true;
  return values
    .filter((value) => value !== null && value !== undefined)
    .some((value) => String(value).toLowerCase().includes(query));
}

interface UseDouyinFiltersOptions {
  users: DouyinUser[];
  videos: DouyinVideo[];
  systems: System[];
  selectedUserIds: Set<number>;
  userSearchQuery: string;
  userSystemFilter: number | "all" | "none";
  userStatusFilter: DouyinUserStatus | "all";
  videoSearchQuery: string;
}

export default function useDouyinFilters({
  users,
  videos,
  systems,
  selectedUserIds,
  userSearchQuery,
  userSystemFilter,
  userStatusFilter,
  videoSearchQuery,
}: UseDouyinFiltersOptions) {
  const systemNameById = useMemo(
    () => new Map(systems.map((system) => [system.id, system.name])),
    [systems],
  );
  const userNameById = useMemo(
    () => new Map(users.map((user) => [user.id, getUserDisplayName(user)])),
    [users],
  );
  const filteredUsers = useMemo(() => {
    const query = normalizeSearch(userSearchQuery);
    return users.filter((user) => {
      const matchesStatus = userStatusFilter === "all" || user.status === userStatusFilter;
      const matchesSystem =
        userSystemFilter === "all" ||
        (userSystemFilter === "none"
          ? user.system_id === null
          : user.system_id === userSystemFilter);
      return (
        matchesStatus &&
        matchesSystem &&
        matchesSearch(
          [
            user.id,
            getUserDisplayName(user),
            getUserSecondaryName(user),
            user.name,
            user.translated_name,
            user.sec_uid,
            user.status,
            user.topic,
            user.niche,
            user.sub_niche,
            user.micro_niche,
            user.note,
            user.system_id ? systemNameById.get(user.system_id) : null,
          ],
          query,
        )
      );
    });
  }, [systemNameById, userSearchQuery, userStatusFilter, userSystemFilter, users]);
  const filteredVideos = useMemo(() => {
    const query = normalizeSearch(videoSearchQuery);
    return videos.filter((video) =>
      matchesSearch(
        [
          video.id,
          video.aweme_id,
          getVideoDisplayTitle(video),
          getVideoSecondaryTitle(video),
          video.title,
          video.translated_title,
          userNameById.get(video.user_id),
          video.user_id,
          video.digg_count,
          formatDateTime(video.create_time),
          video.is_downloaded ? "yes downloaded true" : "no not downloaded false",
        ],
        query,
      ),
    );
  }, [userNameById, videoSearchQuery, videos]);
  const selectedVisibleCount = filteredUsers.filter((user) => selectedUserIds.has(user.id)).length;

  return {
    allVisibleUsersSelected:
      filteredUsers.length > 0 && selectedVisibleCount === filteredUsers.length,
    filteredUsers,
    filteredVideos,
    someVisibleUsersSelected:
      selectedVisibleCount > 0 && selectedVisibleCount < filteredUsers.length,
    systemNameById,
    userNameById,
  };
}
