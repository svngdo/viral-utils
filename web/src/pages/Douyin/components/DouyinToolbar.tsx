import { Download, LoaderCircle, RefreshCw } from "lucide-react";
import { PageActions, PageToolbar } from "@/components/Page";
import SearchInput from "@/components/SearchInput";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import DouyinJobToolbarStatus from "@/pages/Douyin/components/DouyinJobToolbarStatus";
import type { DouyinUserStatus } from "@/pages/Douyin/types";
import type { JobEvent, ProgressEvent, StatusEvent } from "@/pages/Video/types";
import type { System } from "@/pages/Systems/types";

interface DouyinToolbarProps {
  activeTab: "users" | "videos";
  downloadEvents: JobEvent[];
  downloadJobScope: "active" | "selected" | null;
  isDownloadJobRunning: boolean;
  isFetchJobRunning: boolean;
  loading: boolean;
  selectedUserCount: number;
  syncEvents: JobEvent[];
  syncJobScope: "active" | "selected" | null;
  systems: System[];
  statuses: DouyinUserStatus[];
  userSearchQuery: string;
  userSystemFilter: number | "all" | "none";
  userStatusFilter: DouyinUserStatus | "all";
  videoSearchQuery: string;
  onCancelDownloadJob: () => void;
  onCancelFetchJob: () => void;
  onDownloadActiveVideos: () => void;
  onDownloadSelectedUsers: () => void;
  onFetchActiveUsers: () => void;
  onFetchSelectedUsers: () => void;
  onUserSearchChange: (value: string) => void;
  onUserSystemFilterChange: (value: number | "all" | "none") => void;
  onUserStatusFilterChange: (value: DouyinUserStatus | "all") => void;
  onVideoSearchChange: (value: string) => void;
}

function latestEvent<T extends JobEvent>(events: JobEvent[], type: T["type"]): T | undefined {
  return [...events].reverse().find((event): event is T => event.type === type);
}

// Builds the temporary button label while a background job is running.
// Status events tell us when cancellation has started; progress events give
// us the optional "done/total" count used in labels like "Syncing 3/10".
function getRunningLabel(label: string, events: JobEvent[]) {
  const status = latestEvent<StatusEvent>(events, "status")?.status;
  const progress = latestEvent<ProgressEvent>(events, "progress");

  if (status === "cancelling") return "Cancelling";
  if (progress?.total) return `${label} ${progress.done}/${progress.total}`;
  return `${label}...`;
}

// Groups search and job controls so the page stays focused on layout.
export default function DouyinToolbar({
  activeTab,
  downloadEvents,
  downloadJobScope,
  isDownloadJobRunning,
  isFetchJobRunning,
  loading,
  selectedUserCount,
  syncEvents,
  syncJobScope,
  systems,
  statuses,
  userSearchQuery,
  userSystemFilter,
  userStatusFilter,
  videoSearchQuery,
  onCancelDownloadJob,
  onCancelFetchJob,
  onDownloadActiveVideos,
  onDownloadSelectedUsers,
  onFetchActiveUsers,
  onFetchSelectedUsers,
  onUserSearchChange,
  onUserSystemFilterChange,
  onUserStatusFilterChange,
  onVideoSearchChange,
}: DouyinToolbarProps) {
  // The toolbar has one search input, but users and videos keep separate query state.
  const activeSearchQuery = activeTab === "users" ? userSearchQuery : videoSearchQuery;

  // Job event arrays contain logs, progress, and status updates. For toolbar
  // state we only need the latest status event from each stream.
  const syncStatus = latestEvent<StatusEvent>(syncEvents, "status")?.status;
  const downloadStatus = latestEvent<StatusEvent>(downloadEvents, "status")?.status;

  // Sync and download jobs can be started for either "active" records or
  // selected records. The scope lets the matching button become the cancel
  // button while the other same-type button stays disabled.
  const syncActiveRunning = isFetchJobRunning && syncJobScope === "active";
  const syncSelectedRunning = isFetchJobRunning && syncJobScope === "selected";
  const downloadActiveRunning = isDownloadJobRunning && downloadJobScope === "active";
  const downloadSelectedRunning = isDownloadJobRunning && downloadJobScope === "selected";

  // A cancelling job is still owned by the backend. Keep related controls
  // disabled until the stream reports a final stopped/completed state upstream.
  const syncCancelling = syncStatus === "cancelling";
  const downloadCancelling = downloadStatus === "cancelling";

  return (
    <PageToolbar>
      <SearchInput
        value={activeSearchQuery}
        onChange={activeTab === "users" ? onUserSearchChange : onVideoSearchChange}
        placeholder={activeTab === "users" ? "Search users" : "Search current video page"}
        className={activeTab === "users" ? "w-full sm:w-56 lg:w-64" : "w-full"}
      />
      {activeTab === "users" && (
        <PageActions className="min-w-0 flex-1 sm:justify-start">
          <Select
            value={userStatusFilter}
            onValueChange={(value) =>
              onUserStatusFilterChange(value as DouyinUserStatus | "all")
            }
          >
            <SelectTrigger className="w-full sm:w-36">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent position="popper">
              <SelectGroup>
                <SelectLabel>Status</SelectLabel>
                <SelectItem value="all">All statuses</SelectItem>
                {statuses.map((status) => (
                  <SelectItem key={status} value={status}>
                    {status}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
          <Select
            value={String(userSystemFilter)}
            onValueChange={(value) =>
              onUserSystemFilterChange(
                value === "all" || value === "none" ? value : Number(value),
              )
            }
          >
            <SelectTrigger className="w-full sm:w-44">
              <SelectValue placeholder="System" />
            </SelectTrigger>
            <SelectContent position="popper">
              <SelectGroup>
                <SelectLabel>System</SelectLabel>
                <SelectItem value="all">All systems</SelectItem>
                <SelectItem value="none">No system</SelectItem>
                {systems.map((system) => (
                  <SelectItem key={system.id} value={String(system.id)}>
                    {system.name}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
          <Button
            type="button"
            variant="outline"
            // Sync-active cannot start during loading, download work, cancellation,
            // or while a different sync scope is already running.
            disabled={
              loading ||
              isDownloadJobRunning ||
              syncCancelling ||
              (isFetchJobRunning && !syncActiveRunning)
            }
            title={syncActiveRunning ? "Cancel sync active" : undefined}
            onClick={syncActiveRunning ? onCancelFetchJob : onFetchActiveUsers}
          >
            {syncActiveRunning ? <LoaderCircle className="animate-spin" /> : <RefreshCw />}
            {syncActiveRunning ? getRunningLabel("Syncing", syncEvents) : "Sync active"}
          </Button>
          <Button
            type="button"
            variant="outline"
            // Sync-selected has the same job guards as sync-active, plus it
            // needs at least one selected user unless it is already running.
            disabled={
              loading ||
              isDownloadJobRunning ||
              syncCancelling ||
              (isFetchJobRunning && !syncSelectedRunning) ||
              (!syncSelectedRunning && selectedUserCount === 0)
            }
            title={syncSelectedRunning ? "Cancel sync" : undefined}
            onClick={syncSelectedRunning ? onCancelFetchJob : onFetchSelectedUsers}
          >
            {syncSelectedRunning ? <LoaderCircle className="animate-spin" /> : <RefreshCw />}
            {syncSelectedRunning ? getRunningLabel("Syncing", syncEvents) : "Sync"}
          </Button>
          <Button
            type="button"
            variant="outline"
            // Download-active is blocked by fetch/sync work because both jobs
            // operate on Douyin data and should not compete for page state.
            disabled={
              loading ||
              isFetchJobRunning ||
              downloadCancelling ||
              (isDownloadJobRunning && !downloadActiveRunning)
            }
            title={downloadActiveRunning ? "Cancel download active" : undefined}
            onClick={downloadActiveRunning ? onCancelDownloadJob : onDownloadActiveVideos}
          >
            {downloadActiveRunning ? <LoaderCircle className="animate-spin" /> : <Download />}
            {downloadActiveRunning
              ? getRunningLabel("Downloading", downloadEvents)
              : "Download active"}
          </Button>
          <Button
            type="button"
            variant="outline"
            // Download-selected follows the download guards and also requires
            // selected users before it can start.
            disabled={
              loading ||
              isFetchJobRunning ||
              downloadCancelling ||
              (isDownloadJobRunning && !downloadSelectedRunning) ||
              (!downloadSelectedRunning && selectedUserCount === 0)
            }
            title={downloadSelectedRunning ? "Cancel download" : undefined}
            onClick={downloadSelectedRunning ? onCancelDownloadJob : onDownloadSelectedUsers}
          >
            {downloadSelectedRunning ? <LoaderCircle className="animate-spin" /> : <Download />}
            {downloadSelectedRunning ? getRunningLabel("Downloading", downloadEvents) : "Download"}
          </Button>
          <DouyinJobToolbarStatus syncEvents={syncEvents} downloadEvents={downloadEvents} />
        </PageActions>
      )}
    </PageToolbar>
  );
}
