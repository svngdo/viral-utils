import { PageHeader, PageSubtitle, PageTitle } from "@/components/Page";
import CreateUserDialog from "@/pages/Douyin/components/CreateUserDialog";
import type { DouyinUserCreate, DouyinUserStatus } from "@/pages/Douyin/types";
import type { System } from "@/pages/Systems/types";

interface DouyinPageHeaderProps {
  activeTab: "users" | "videos";
  subtitle: string;
  systems: System[];
  statuses: DouyinUserStatus[];
  onCreateUser: (data: DouyinUserCreate) => Promise<void>;
}

// Renders the page title and the user-only create action.
export default function DouyinPageHeader({
  activeTab,
  subtitle,
  systems,
  statuses,
  onCreateUser,
}: DouyinPageHeaderProps) {
  return (
    <PageHeader>
      <div>
        <PageTitle>Douyin</PageTitle>
        <PageSubtitle>{subtitle}</PageSubtitle>
      </div>
      {activeTab === "users" && (
        <CreateUserDialog systems={systems} statuses={statuses} onSubmit={onCreateUser} />
      )}
    </PageHeader>
  );
}
