import { AlertCircle } from "lucide-react";
import { Page } from "@/components/Page";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import DouyinPageHeader from "@/pages/Douyin/components/DouyinPageHeader";
import DouyinTabs from "@/pages/Douyin/components/DouyinTabs";
import DouyinToolbar from "@/pages/Douyin/components/DouyinToolbar";
import DouyinVideosPanel from "@/pages/Douyin/components/DouyinVideosPanel";
import UserTable from "@/pages/Douyin/components/UserTable";
import useDouyinPage from "@/pages/Douyin/hooks/useDouyinPage";

export default function Douyin() {
  const {
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
    handleCancelDownloadJob,
    handleCancelFetchJob,
    handleCreateUser,
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
    isDownloadJobRunning,
    isFetchJobRunning,
    lastVideoNumber,
    loading,
    setActiveTab,
    setDeletingVideo,
    setEditingVideo,
    setUserSearchQuery,
    setUserSystemFilter,
    setUserStatusFilter,
    setVideoSearchQuery,
    selectedUserCount,
    selectedUserIds,
    someVisibleUsersSelected,
    downloadJobScope,
    downloadEvents,
    syncJobScope,
    syncEvents,
    statuses,
    systems,
    userNameById,
    users,
    userSearchQuery,
    userSystemFilter,
    userStatusFilter,
    videoSearchQuery,
    videoTotal,
    videos,
    videosLoading,
  } = useDouyinPage();
  const pageSubtitle = loading
    ? "Loading..."
    : activeTab === "users" &&
        (userSearchQuery || userStatusFilter !== "all" || userSystemFilter !== "all")
      ? `${filteredUsers.length} of ${users.length} users, ${videoTotal} videos`
      : activeTab === "videos" && videoSearchQuery
        ? `${users.length} users, ${filteredVideos.length} of ${videos.length} videos on this page`
        : `${users.length} users, ${videoTotal} videos`;
  const displayedFirstVideoNumber = videoSearchQuery
    ? filteredVideos.length
      ? 1
      : 0
    : firstVideoNumber;
  const displayedLastVideoNumber = videoSearchQuery ? filteredVideos.length : lastVideoNumber;
  const displayedVideoTotal = videoSearchQuery ? filteredVideos.length : videoTotal;

  return (
    <Page>
      <DouyinPageHeader
        activeTab={activeTab}
        subtitle={pageSubtitle}
        systems={systems}
        statuses={statuses}
        onCreateUser={handleCreateUser}
      />

      {displayError && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{displayError}</AlertDescription>
        </Alert>
      )}

      <DouyinTabs activeTab={activeTab} onTabChange={setActiveTab} />

      <DouyinToolbar
        activeTab={activeTab}
        downloadEvents={downloadEvents}
        downloadJobScope={downloadJobScope}
        isDownloadJobRunning={isDownloadJobRunning}
        isFetchJobRunning={isFetchJobRunning}
        loading={loading}
        selectedUserCount={selectedUserCount}
        syncEvents={syncEvents}
        syncJobScope={syncJobScope}
        systems={systems}
        statuses={statuses}
        userSearchQuery={userSearchQuery}
        userSystemFilter={userSystemFilter}
        userStatusFilter={userStatusFilter}
        videoSearchQuery={videoSearchQuery}
        onCancelDownloadJob={handleCancelDownloadJob}
        onCancelFetchJob={handleCancelFetchJob}
        onDownloadActiveVideos={handleDownloadActiveVideos}
        onDownloadSelectedUsers={handleDownloadSelectedUsers}
        onFetchActiveUsers={handleFetchActiveUsers}
        onFetchSelectedUsers={handleFetchSelectedUsers}
        onUserSearchChange={setUserSearchQuery}
        onUserSystemFilterChange={setUserSystemFilter}
        onUserStatusFilterChange={setUserStatusFilter}
        onVideoSearchChange={setVideoSearchQuery}
      />

      {activeTab === "users" ? (
        <UserTable
          users={filteredUsers}
          systems={systems}
          statuses={statuses}
          loading={loading}
          selectedUserIds={selectedUserIds}
          allVisibleUsersSelected={allVisibleUsersSelected}
          someVisibleUsersSelected={someVisibleUsersSelected}
          fetchDisabled={isFetchJobRunning}
          onToggleSelected={handleToggleUserSelected}
          onToggleAllVisible={handleToggleAllVisibleUsers}
          onUpdate={handleUpdateUser}
          onDelete={handleDeleteUser}
        />
      ) : (
        <DouyinVideosPanel
          canGoNext={canGoNext}
          canGoPrevious={canGoPrevious}
          deletingVideo={deletingVideo}
          displayedFirstVideoNumber={displayedFirstVideoNumber}
          displayedLastVideoNumber={displayedLastVideoNumber}
          displayedVideoTotal={displayedVideoTotal}
          editingVideo={editingVideo}
          filteredVideos={filteredVideos}
          loading={loading}
          userNameById={userNameById}
          videoSearchQuery={videoSearchQuery}
          videosLoading={videosLoading}
          onDeleteVideo={handleDeleteVideo}
          onEditVideoChange={setEditingVideo}
          onDeleteVideoChange={setDeletingVideo}
          onNextVideoPage={handleNextVideoPage}
          onPreviousVideoPage={handlePreviousVideoPage}
          onUpdateVideo={handleUpdateVideo}
        />
      )}
    </Page>
  );
}
