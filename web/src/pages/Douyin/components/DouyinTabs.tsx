import { PageTabs } from "@/components/Page";
import { Button } from "@/components/ui/button";

interface DouyinTabsProps {
  activeTab: "users" | "videos";
  onTabChange: (tab: "users" | "videos") => void;
}

// Keeps the two page modes visually and behaviorally consistent.
export default function DouyinTabs({ activeTab, onTabChange }: DouyinTabsProps) {
  return (
    <PageTabs>
      <Button
        type="button"
        variant={activeTab === "users" ? "secondary" : "ghost"}
        onClick={() => onTabChange("users")}
      >
        Users
      </Button>
      <Button
        type="button"
        variant={activeTab === "videos" ? "secondary" : "ghost"}
        onClick={() => onTabChange("videos")}
      >
        Videos
      </Button>
    </PageTabs>
  );
}
