import { Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TableCell, TableRow } from "@/components/ui/table";
import { SelectionCheckbox } from "@/pages/Douyin/components/UserTableControls";
import { getUserDisplayName, getUserSecondaryName } from "@/pages/Douyin/display";
import { formatDateTime } from "@/pages/Douyin/format";
import type { DouyinUser, DouyinUserStatus, DouyinUserUpdate } from "@/pages/Douyin/types";
import type { System } from "@/pages/Systems/types";

interface UserTableRowProps {
  user: DouyinUser;
  systems: System[];
  statuses: DouyinUserStatus[];
  selectedUserIds: Set<number>;
  fetchDisabled: boolean;
  onToggleSelected: (id: number, selected: boolean) => void;
  onUpdate: (id: number, data: DouyinUserUpdate) => Promise<void>;
  onEdit: (user: DouyinUser) => void;
  onDelete: (user: DouyinUser) => void;
}

export default function UserTableRow({
  user,
  systems,
  statuses,
  selectedUserIds,
  fetchDisabled,
  onToggleSelected,
  onUpdate,
  onEdit,
  onDelete,
}: UserTableRowProps) {
  const displayName = getUserDisplayName(user);
  const secondaryName = getUserSecondaryName(user);
  const systemName = user.system_id
    ? systems.find((system) => system.id === user.system_id)?.name || user.system_id
    : "No system";

  return (
    <TableRow>
      <TableCell>
        <SelectionCheckbox
          label={`Select ${displayName}`}
          checked={selectedUserIds.has(user.id)}
          disabled={fetchDisabled}
          onCheckedChange={(selected) => onToggleSelected(user.id, selected)}
        />
      </TableCell>
      <TableCell>{user.id}</TableCell>
      <TableCell>
        <div className="max-w-64 truncate font-medium" title={displayName}>
          {displayName}
        </div>
        <div className="max-w-64 truncate text-muted-foreground" title={secondaryName}>
          {secondaryName}
        </div>
      </TableCell>
      <TableCell>
        <select
          aria-label={`Status for ${displayName}`}
          value={user.status}
          disabled={fetchDisabled}
          className="h-7 rounded-md border border-input bg-background px-2 text-sm"
          onChange={(event) => {
            void onUpdate(user.id, { status: event.currentTarget.value as DouyinUserStatus });
          }}
        >
          {statuses.map((status) => (
            <option key={status} value={status}>
              {status}
            </option>
          ))}
        </select>
      </TableCell>
      <TableCell>{user.topic}</TableCell>
      <TableCell>{user.niche}</TableCell>
      <TableCell>{systemName}</TableCell>
      <TableCell>{formatDateTime(user.last_fetched).slice(0, 10)}</TableCell>
      <TableCell onClick={(event) => event.stopPropagation()}>
        <div className="flex justify-end gap-1">
          <Button
            type="button"
            size="icon-sm"
            aria-label={`Edit user ${displayName}`}
            title="Edit"
            onClick={() => onEdit(user)}
          >
            <Pencil />
          </Button>
          <Button
            type="button"
            size="icon-sm"
            variant="outline"
            aria-label={`Delete user ${displayName}`}
            title="Delete"
            onClick={() => onDelete(user)}
          >
            <Trash2 />
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}
