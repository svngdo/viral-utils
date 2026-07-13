import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

interface VideoPaginationProps {
  videoCount: number;
  videoTotal: number;
  firstVideoNumber: number;
  lastVideoNumber: number;
  canGoPrevious: boolean;
  canGoNext: boolean;
  videosLoading: boolean;
  searching: boolean;
  onPreviousPage: () => void;
  onNextPage: () => void;
}

export default function VideoPagination({
  videoCount,
  videoTotal,
  firstVideoNumber,
  lastVideoNumber,
  canGoPrevious,
  canGoNext,
  videosLoading,
  searching,
  onPreviousPage,
  onNextPage,
}: VideoPaginationProps) {
  return (
    <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
      <div className="text-sm text-muted-foreground">
        {videosLoading
          ? "Loading videos..."
          : searching
            ? `${videoCount} matching on this page`
            : `Showing ${firstVideoNumber}-${lastVideoNumber} of ${videoTotal}`}
      </div>
      <div className="flex gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!canGoPrevious}
          onClick={onPreviousPage}
        >
          <ChevronLeft />
          Previous
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!canGoNext}
          onClick={onNextPage}
        >
          Next
          <ChevronRight />
        </Button>
      </div>
    </div>
  );
}
