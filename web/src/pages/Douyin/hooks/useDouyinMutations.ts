import type { Dispatch, SetStateAction } from "react";
import * as douyinApi from "@/pages/Douyin/api";
import { VIDEO_PAGE_SIZE } from "@/pages/Douyin/constants";
import type {
  DouyinUser,
  DouyinUserCreate,
  DouyinUserUpdate,
  DouyinVideo,
  DouyinVideoUpdate,
} from "@/pages/Douyin/types";

interface UseDouyinMutationsOptions {
  loadVideoPage: (offset: number) => Promise<void>;
  setActionError: Dispatch<SetStateAction<string>>;
  setUsers: Dispatch<SetStateAction<DouyinUser[]>>;
  videoOffset: number;
  videos: DouyinVideo[];
}

export default function useDouyinMutations({
  loadVideoPage,
  setActionError,
  setUsers,
  videoOffset,
  videos,
}: UseDouyinMutationsOptions) {
  const runMutation = async (fallbackMessage: string, action: () => Promise<void>) => {
    setActionError("");
    try {
      await action();
    } catch (error) {
      const message = error instanceof Error ? error.message : fallbackMessage;
      setActionError(message);
      throw new Error(message);
    }
  };

  const handleCreateUser = (data: DouyinUserCreate) =>
    runMutation("Could not create user", async () => {
      const created = await douyinApi.createUser(data);
      setUsers((previous) => [created, ...previous]);
    });

  const handleUpdateUser = (id: number, data: DouyinUserUpdate) =>
    runMutation("Could not update user", async () => {
      const updated = await douyinApi.updateUser(id, data);
      setUsers((previous) => previous.map((user) => (user.id === id ? updated : user)));
    });

  const handleDeleteUser = (id: number) =>
    runMutation("Could not delete user", async () => {
      await douyinApi.removeUser(id);
      setUsers((previous) => previous.filter((user) => user.id !== id));
      await loadVideoPage(videoOffset);
    });

  const handleUpdateVideo = (id: number, data: DouyinVideoUpdate) =>
    runMutation("Could not update video", async () => {
      await douyinApi.updateVideo(id, data);
      await loadVideoPage(videoOffset);
    });

  const handleDeleteVideo = (id: number) =>
    runMutation("Could not delete video", async () => {
      await douyinApi.removeVideo(id);
      const nextOffset =
        videos.length === 1 && videoOffset > 0
          ? Math.max(0, videoOffset - VIDEO_PAGE_SIZE)
          : videoOffset;
      await loadVideoPage(nextOffset);
    });

  return {
    handleCreateUser,
    handleDeleteUser,
    handleDeleteVideo,
    handleUpdateUser,
    handleUpdateVideo,
  };
}
