import { useCallback, useState } from "react";
import * as douyinApi from "@/pages/Douyin/api";
import { DEFAULT_STATUSES, VIDEO_PAGE_SIZE } from "@/pages/Douyin/constants";
import type { DouyinUser, DouyinUserStatus, DouyinVideo } from "@/pages/Douyin/types";
import * as systemApi from "@/pages/Systems/api";
import type { System } from "@/pages/Systems/types";

export default function useDouyinData() {
  const [users, setUsers] = useState<DouyinUser[]>([]);
  const [videos, setVideos] = useState<DouyinVideo[]>([]);
  const [videoTotal, setVideoTotal] = useState(0);
  const [videoOffset, setVideoOffset] = useState(0);
  const [videosLoading, setVideosLoading] = useState(false);
  const [systems, setSystems] = useState<System[]>([]);
  const [statuses, setStatuses] = useState<DouyinUserStatus[]>(DEFAULT_STATUSES);

  const loadVideoPage = useCallback(async (offset: number) => {
    setVideosLoading(true);
    try {
      const page = await douyinApi.getVideoPage({ limit: VIDEO_PAGE_SIZE, offset });
      setVideos(page.items);
      setVideoTotal(page.total);
      setVideoOffset(page.offset);
    } finally {
      setVideosLoading(false);
    }
  }, []);

  const loadData = useCallback(async () => {
    const [nextUsers, nextVideoPage, nextSystems, nextStatuses] = await Promise.all([
      douyinApi.getUsers(),
      douyinApi.getVideoPage({ limit: VIDEO_PAGE_SIZE, offset: 0 }),
      systemApi.getAll(),
      douyinApi.getUserStatuses().catch(() => DEFAULT_STATUSES),
    ]);
    setUsers(nextUsers);
    setVideos(nextVideoPage.items);
    setVideoTotal(nextVideoPage.total);
    setVideoOffset(nextVideoPage.offset);
    setSystems(nextSystems);
    setStatuses(nextStatuses);
  }, []);

  return {
    canGoNext: videoOffset + VIDEO_PAGE_SIZE < videoTotal && !videosLoading,
    canGoPrevious: videoOffset > 0 && !videosLoading,
    firstVideoNumber: videoTotal ? videoOffset + 1 : 0,
    lastVideoNumber: Math.min(videoOffset + videos.length, videoTotal),
    loadData,
    loadVideoPage,
    setUsers,
    statuses,
    systems,
    users,
    videoOffset,
    videoTotal,
    videos,
    videosLoading,
  };
}
