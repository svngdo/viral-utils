import { useEffect, useState } from "react";
import type { DouyinUser } from "@/pages/Douyin/types";

export default function useDouyinSelection({ users }: { users: DouyinUser[] }) {
  const [selectedUserIds, setSelectedUserIds] = useState<Set<number>>(() => new Set());

  useEffect(() => {
    const availableIds = new Set(users.map((user) => user.id));
    setSelectedUserIds((previous) => {
      const next = new Set([...previous].filter((id) => availableIds.has(id)));
      return next.size === previous.size ? previous : next;
    });
  }, [users]);

  const handleToggleUserSelected = (id: number, selected: boolean) => {
    setSelectedUserIds((previous) => {
      const next = new Set(previous);
      if (selected) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  return {
    selectedUserCount: selectedUserIds.size,
    selectedUserIds,
    setSelectedUserIds,
    handleToggleUserSelected,
  };
}
