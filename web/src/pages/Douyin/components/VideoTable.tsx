import { Table, TableBody, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import VideoPagination from "@/pages/Douyin/components/VideoPagination";
import VideoTableMessageRow from "@/pages/Douyin/components/VideoTableMessageRow";
import VideoTableRow from "@/pages/Douyin/components/VideoTableRow";
import type { DouyinVideo } from "@/pages/Douyin/types";

const VIDEO_TABLE_COLUMN_COUNT = 9;

const VIDEO_TABLE_HEADERS = [
  { label: "Aweme ID", className: "w-28 whitespace-normal leading-tight" },
  { label: "Title", className: "w-[28%]" },
  { label: "User", className: "w-32" },
  { label: "Created Time", className: "w-36 whitespace-normal leading-tight" },
  { label: "Diggs Count", className: "w-16 whitespace-normal text-right leading-tight" },
  { label: "Duration", className: "w-16 text-right" },
  { label: "Links", className: "w-32 whitespace-normal leading-tight" },
  { label: "Download Status", className: "w-20 whitespace-normal text-center leading-tight" },
  { label: "Actions", className: "w-20 text-right" },
];

interface VideoTableProps {
  videos: DouyinVideo[];
  videoTotal: number;
  firstVideoNumber: number;
  lastVideoNumber: number;
  canGoPrevious: boolean;
  canGoNext: boolean;
  loading: boolean;
  videosLoading: boolean;
  searching: boolean;
  userNameById: Map<number, string>;
  onPreviousPage: () => void;
  onNextPage: () => void;
  onEdit: (video: DouyinVideo) => void;
  onDelete: (video: DouyinVideo) => void;
}

// Renders paginated videos while delegating row details to focused children.
export default function VideoTable({
  videos,
  videoTotal,
  firstVideoNumber,
  lastVideoNumber,
  canGoPrevious,
  canGoNext,
  loading,
  videosLoading,
  searching,
  userNameById,
  onPreviousPage,
  onNextPage,
  onEdit,
  onDelete,
}: VideoTableProps) {
  const showLoading = loading || videosLoading;

  return (
    <>
      <VideoPagination
        canGoNext={canGoNext}
        canGoPrevious={canGoPrevious}
        firstVideoNumber={firstVideoNumber}
        lastVideoNumber={lastVideoNumber}
        searching={searching}
        videoCount={videos.length}
        videosLoading={videosLoading}
        videoTotal={videoTotal}
        onNextPage={onNextPage}
        onPreviousPage={onPreviousPage}
      />
      <div className="overflow-x-auto">
        <Table className="table-fixed">
          <TableHeader>
            <TableRow>
              {VIDEO_TABLE_HEADERS.map((header) => (
                <TableHead key={header.label} className={header.className}>
                  {header.label}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {showLoading ? (
              <VideoTableMessageRow colSpan={VIDEO_TABLE_COLUMN_COUNT} message="Loading..." />
            ) : videos.length ? (
              videos.map((video) => (
                <VideoTableRow
                  key={video.id}
                  userNameById={userNameById}
                  video={video}
                  onDelete={onDelete}
                  onEdit={onEdit}
                />
              ))
            ) : (
              <VideoTableMessageRow colSpan={VIDEO_TABLE_COLUMN_COUNT} message="No videos" />
            )}
          </TableBody>
        </Table>
      </div>
    </>
  );
}
