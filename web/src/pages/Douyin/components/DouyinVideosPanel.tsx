import DeleteVideoDialog from "@/pages/Douyin/components/DeleteVideoDialog";
import EditVideoDialog from "@/pages/Douyin/components/EditVideoDialog";
import VideoTable from "@/pages/Douyin/components/VideoTable";
import type { DouyinVideo, DouyinVideoUpdate } from "@/pages/Douyin/types";

interface DouyinVideosPanelProps {
  canGoNext: boolean;
  canGoPrevious: boolean;
  deletingVideo: DouyinVideo | null;
  displayedFirstVideoNumber: number;
  displayedLastVideoNumber: number;
  displayedVideoTotal: number;
  editingVideo: DouyinVideo | null;
  filteredVideos: DouyinVideo[];
  loading: boolean;
  userNameById: Map<number, string>;
  videoSearchQuery: string;
  videosLoading: boolean;
  onDeleteVideo: (id: number) => Promise<void>;
  onEditVideoChange: (video: DouyinVideo | null) => void;
  onDeleteVideoChange: (video: DouyinVideo | null) => void;
  onNextVideoPage: () => void;
  onPreviousVideoPage: () => void;
  onUpdateVideo: (id: number, data: DouyinVideoUpdate) => Promise<void>;
}

// Owns video table pagination and the dialogs opened from video rows.
export default function DouyinVideosPanel({
  canGoNext,
  canGoPrevious,
  deletingVideo,
  displayedFirstVideoNumber,
  displayedLastVideoNumber,
  displayedVideoTotal,
  editingVideo,
  filteredVideos,
  loading,
  userNameById,
  videoSearchQuery,
  videosLoading,
  onDeleteVideo,
  onEditVideoChange,
  onDeleteVideoChange,
  onNextVideoPage,
  onPreviousVideoPage,
  onUpdateVideo,
}: DouyinVideosPanelProps) {
  return (
    <>
      <VideoTable
        videos={filteredVideos}
        videoTotal={displayedVideoTotal}
        firstVideoNumber={displayedFirstVideoNumber}
        lastVideoNumber={displayedLastVideoNumber}
        canGoPrevious={canGoPrevious}
        canGoNext={canGoNext}
        loading={loading}
        videosLoading={videosLoading}
        searching={Boolean(videoSearchQuery)}
        userNameById={userNameById}
        onPreviousPage={onPreviousVideoPage}
        onNextPage={onNextVideoPage}
        onEdit={onEditVideoChange}
        onDelete={onDeleteVideoChange}
      />
      {editingVideo && (
        <EditVideoDialog
          video={editingVideo}
          onSubmit={onUpdateVideo}
          open
          onOpenChange={(open) => {
            if (!open) onEditVideoChange(null);
          }}
        />
      )}
      {deletingVideo && (
        <DeleteVideoDialog
          video={deletingVideo}
          onDelete={onDeleteVideo}
          open
          onOpenChange={(open) => {
            if (!open) onDeleteVideoChange(null);
          }}
        />
      )}
    </>
  );
}
