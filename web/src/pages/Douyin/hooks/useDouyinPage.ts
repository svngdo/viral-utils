import { useCallback, useEffect, useState } from "react";
import * as douyinApi from "@/pages/Douyin/api";
import { VIDEO_PAGE_SIZE } from "@/pages/Douyin/constants";
import useDouyinData from "@/pages/Douyin/hooks/useDouyinData";
import useDouyinFilters from "@/pages/Douyin/hooks/useDouyinFilters";
import useDouyinJob from "@/pages/Douyin/hooks/useDouyinJob";
import useDouyinMutations from "@/pages/Douyin/hooks/useDouyinMutations";
import useDouyinSelection from "@/pages/Douyin/hooks/useDouyinSelection";
import type { DouyinUserStatus, DouyinVideo } from "@/pages/Douyin/types";

export default function useDouyinPage() {
  const [activeTab, setActiveTab] = useState<"users" | "videos">("users");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [actionError, setActionError] = useState("");
  const [userSearchQuery, setUserSearchQuery] = useState("");
  const [userSystemFilter, setUserSystemFilter] = useState<number | "all" | "none">("all");
  const [userStatusFilter, setUserStatusFilter] = useState<DouyinUserStatus | "all">("all");
  const [videoSearchQuery, setVideoSearchQuery] = useState("");
  const [editingVideo, setEditingVideo] = useState<DouyinVideo | null>(null);
  const [deletingVideo, setDeletingVideo] = useState<DouyinVideo | null>(null);
  const [latestJobKind, setLatestJobKind] = useState<"sync" | "download" | null>(null);

  const {
    canGoNext,
    canGoPrevious,
    firstVideoNumber,
    lastVideoNumber,
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
  } = useDouyinData();

  const displayError = loadError || actionError;

  const handleActionError = useCallback((message: string) => {
    setActionError(message);
  }, []);

  const refreshAfterFetchJob = useCallback(async () => {
    const [nextUsers] = await Promise.all([douyinApi.getUsers(), loadVideoPage(videoOffset)]);
    setUsers(nextUsers);
  }, [loadVideoPage, setUsers, videoOffset]);

  const syncJob = useDouyinJob({
    completeLogMessage: "Sync complete",
    defaultStartErrorMessage: "Could not start sync job",
    incompleteErrorMessage: "Sync job did not complete",
    kind: "sync",
    pollErrorMessage: "Could not stream sync progress",
    refreshErrorMessage: "Could not refresh Douyin data",
    runningLogMessage: "Syncing...",
    terminalLogPrefix: "Sync",
    onCompleted: refreshAfterFetchJob,
    onError: handleActionError,
    onLatestKindChange: setLatestJobKind,
  });

  const downloadJob = useDouyinJob({
    completeLogMessage: "Download complete",
    defaultStartErrorMessage: "Could not start download job",
    incompleteErrorMessage: "Download job did not complete",
    kind: "download",
    pollErrorMessage: "Could not stream download progress",
    refreshErrorMessage: "Could not refresh Douyin videos",
    runningLogMessage: "Downloading...",
    terminalLogPrefix: "Download",
    onCompleted: () => loadVideoPage(videoOffset),
    onError: handleActionError,
    onLatestKindChange: setLatestJobKind,
  });

  const {
    handleCreateUser,
    handleDeleteUser,
    handleDeleteVideo,
    handleUpdateUser,
    handleUpdateVideo,
  } = useDouyinMutations({
    loadVideoPage,
    setActionError,
    setUsers,
    videoOffset,
    videos,
  });

  useEffect(() => {
    loadData()
      .catch(() => setLoadError("Could not load Douyin data"))
      .finally(() => setLoading(false));
  }, [loadData]);

  const { selectedUserCount, selectedUserIds, setSelectedUserIds, handleToggleUserSelected } =
    useDouyinSelection({ users });

  const {
    allVisibleUsersSelected,
    filteredUsers,
    filteredVideos,
    someVisibleUsersSelected,
    systemNameById,
    userNameById,
  } = useDouyinFilters({
    selectedUserIds,
    systems,
    userSearchQuery,
    userSystemFilter,
    userStatusFilter,
    users,
    videoSearchQuery,
    videos,
  });

  const handleToggleAllVisibleUsers = (selected: boolean) => {
    setSelectedUserIds((prev) => {
      const next = new Set(prev);
      for (const user of filteredUsers) {
        if (selected) {
          next.add(user.id);
        } else {
          next.delete(user.id);
        }
      }
      return next;
    });
  };

  const handleFetchActiveUsers = () => {
    setActionError("");
    void syncJob.startJob("active", () => douyinApi.createFetchActiveUsersJob());
  };

  const handleFetchSelectedUsers = () => {
    const userIds = [...selectedUserIds];
    if (!userIds.length) return;
    setActionError("");
    void syncJob.startJob("selected", () => douyinApi.createFetchSelectedUsersJob(userIds));
  };

  const handleDownloadActiveVideos = useCallback(() => {
    setActionError("");
    void downloadJob.startJob("active", () => douyinApi.createDownloadActiveVideosJob());
  }, [downloadJob]);

  const handleDownloadSelectedUsers = useCallback(() => {
    const userIds = [...selectedUserIds];
    if (!userIds.length) return;
    setActionError("");
    void downloadJob.startJob("selected", () => douyinApi.createDownloadSelectedUsersJob(userIds));
  }, [downloadJob, selectedUserIds]);

  const handlePreviousVideoPage = () => {
    void loadVideoPage(Math.max(0, videoOffset - VIDEO_PAGE_SIZE)).catch(() =>
      setActionError("Could not load videos"),
    );
  };

  const handleNextVideoPage = () => {
    void loadVideoPage(videoOffset + VIDEO_PAGE_SIZE).catch(() =>
      setActionError("Could not load videos"),
    );
  };

  return {
    activeTab,
    canGoNext,
    canGoPrevious,
    deletingVideo,
    displayError,
    editingVideo,
    filteredUsers,
    filteredVideos,
    firstVideoNumber,
    allVisibleUsersSelected,
    handleCreateUser,
    handleCancelDownloadJob: downloadJob.cancelJob,
    handleCancelFetchJob: syncJob.cancelJob,
    handleDeleteUser,
    handleDeleteVideo,
    handleDownloadActiveVideos,
    handleDownloadSelectedUsers,
    handleFetchActiveUsers,
    handleFetchSelectedUsers,
    handleNextVideoPage,
    handlePreviousVideoPage,
    handleToggleAllVisibleUsers,
    handleToggleUserSelected,
    handleUpdateUser,
    handleUpdateVideo,
    isDownloadJobRunning: downloadJob.isRunning,
    isFetchJobRunning: syncJob.isRunning,
    lastVideoNumber,
    latestJobKind,
    loading,
    setActiveTab,
    setDeletingVideo,
    setEditingVideo,
    setUserSystemFilter,
    setUserStatusFilter,
    setUserSearchQuery,
    setVideoSearchQuery,
    selectedUserCount,
    selectedUserIds,
    someVisibleUsersSelected,
    downloadJobScope: downloadJob.scope,
    downloadEvents: downloadJob.events,
    syncJobScope: syncJob.scope,
    syncEvents: syncJob.events,
    statuses,
    systemNameById,
    systems,
    userNameById,
    users,
    userSystemFilter,
    userStatusFilter,
    videoTotal,
    userSearchQuery,
    videoSearchQuery,
    videos,
    videosLoading,
  };
}
