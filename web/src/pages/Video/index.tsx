import { Page, PageContent, PageHeader, PageSubtitle, PageTitle } from "@/components/Page";
import VideoJobSections from "@/pages/Video/components/VideoJobSections";
import useVideoJob from "./hooks/useVideoJob";

export default function Video() {
  const {
    fetchEvents,
    handleFetchLatestVideos,
    processEvents,
    handleProcessVideos,
    handleCancelProcessVideos,
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
          onProcessVideos={handleProcessVideos}
          onCancelProcessVideos={handleCancelProcessVideos}
          isProcessRunning={isProcessRunning}
        />
      </PageContent>
    </Page>
  );
}
