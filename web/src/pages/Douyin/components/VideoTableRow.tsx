import { Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TableCell, TableRow } from "@/components/ui/table";
import { getVideoDisplayTitle, getVideoSecondaryTitle } from "@/pages/Douyin/display";
import {
  formatCompactNumber,
  formatDateTime,
  formatDuration,
  parseVideoUrls,
} from "@/pages/Douyin/format";
import type { DouyinVideo } from "@/pages/Douyin/types";

interface VideoTableRowProps {
  video: DouyinVideo;
  userNameById: Map<number, string>;
  onEdit: (video: DouyinVideo) => void;
  onDelete: (video: DouyinVideo) => void;
}

export default function VideoTableRow({
  video,
  userNameById,
  onEdit,
  onDelete,
}: VideoTableRowProps) {
  const displayTitle = getVideoDisplayTitle(video);
  const secondaryTitle = getVideoSecondaryTitle(video);
  const videoUrls = parseVideoUrls(video.urls ?? "");
  const userName = userNameById.get(video.user_id) || video.user_id;

  return (
    <TableRow>
      <TableCell className="font-medium">
        <div className="truncate" title={video.aweme_id}>
          {video.aweme_id}
        </div>
      </TableCell>
      <TableCell className="min-w-0">
        <div className="truncate" title={displayTitle}>
          {displayTitle}
        </div>
        <div className="truncate text-muted-foreground" title={secondaryTitle}>
          {secondaryTitle}
        </div>
      </TableCell>
      <TableCell>
        <div className="truncate" title={String(userName)}>
          {userName}
        </div>
      </TableCell>
      <TableCell>{formatDateTime(video.create_time)}</TableCell>
      <TableCell className="text-right" title={String(video.digg_count)}>
        {formatCompactNumber(video.digg_count)}
      </TableCell>
      <TableCell className="text-right">{formatDuration(video.duration)}</TableCell>
      <TableCell>
        {videoUrls.length ? (
          <div className="flex flex-wrap gap-2">
            {videoUrls.map((url, index) => (
              <a
                key={url}
                href={url}
                target="_blank"
                rel="noreferrer"
                className="text-sm font-medium text-primary underline-offset-4 hover:underline"
                title={url}
              >
                link{index + 1}
              </a>
            ))}
          </div>
        ) : (
          <span className="text-muted-foreground">No links</span>
        )}
      </TableCell>
      <TableCell className="text-center">{video.is_downloaded ? "Yes" : "No"}</TableCell>
      <TableCell onClick={(event) => event.stopPropagation()}>
        <div className="flex justify-end gap-1">
          <Button
            type="button"
            size="icon-sm"
            aria-label={`Edit video ${displayTitle}`}
            title="Edit"
            onClick={() => onEdit(video)}
          >
            <Pencil />
          </Button>
          <Button
            type="button"
            size="icon-sm"
            variant="outline"
            aria-label={`Delete video ${displayTitle}`}
            title="Delete"
            onClick={() => onDelete(video)}
          >
            <Trash2 />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}
