import { Page, PageContent, PageHeader, PageSubtitle, PageTitle } from "@/components/Page";
import VideoJobSections from "@/pages/Video/components/VideoJobSections";
import useVideoJob from "./hooks/useVideoJob";

export default function Video() {
  const {
    fetchEvents,
    handleFetchLatestVideos,
    handleCancelFetchLatestVideos,
    processEvents,
    handleProcessVideos,
    handleCancelProcessVideos,
    isFetchRunning,
    isProcessRunning,
  } = useVideoJob();

  return (
    <Page>
      <PageHeader>
        <div>
          <PageTitle>Video</PageTitle>
          <PageSubtitle>Fetch and process video jobs.</PageSubtitle>
        </div>
      </PageHeader>

      <PageContent>
        <VideoJobSections
          fetchEvents={fetchEvents}
          processEvents={processEvents}
          onFetchLatestVideos={handleFetchLatestVideos}
          onCancelFetchLatestVideos={handleCancelFetchLatestVideos}
          onProcessVideos={handleProcessVideos}
          onCancelProcessVideos={handleCancelProcessVideos}
          isFetchRunning={isFetchRunning}
          isProcessRunning={isProcessRunning}
        />
      </PageContent>
    </Page>
  );
}
