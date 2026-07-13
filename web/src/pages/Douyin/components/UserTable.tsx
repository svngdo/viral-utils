import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import DeleteUserDialog from "@/pages/Douyin/components/DeleteUserDialog";
import EditUserDialog from "@/pages/Douyin/components/EditUserDialog";
import { SelectionCheckbox } from "@/pages/Douyin/components/UserTableControls";
import UserTableRow from "@/pages/Douyin/components/UserTableRow";
import type { DouyinUser, DouyinUserStatus, DouyinUserUpdate } from "@/pages/Douyin/types";
import type { System } from "@/pages/Systems/types";

interface UserTableProps {
  users: DouyinUser[];
  systems: System[];
  statuses: DouyinUserStatus[];
  loading: boolean;
  selectedUserIds: Set<number>;
  allVisibleUsersSelected: boolean;
  someVisibleUsersSelected: boolean;
  fetchDisabled: boolean;
  onToggleSelected: (id: number, selected: boolean) => void;
  onToggleAllVisible: (selected: boolean) => void;
  onUpdate: (id: number, data: DouyinUserUpdate) => Promise<void>;
  onDelete: (id: number) => Promise<void>;
}

// Renders the users table plus edit/delete dialogs owned by row actions.
export default function UserTable({
  users,
  systems,
  statuses,
  loading,
  selectedUserIds,
  allVisibleUsersSelected,
  someVisibleUsersSelected,
  fetchDisabled,
  onToggleSelected,
  onToggleAllVisible,
  onUpdate,
  onDelete,
}: UserTableProps) {
  const [editingUser, setEditingUser] = useState<DouyinUser | null>(null);
  const [deletingUser, setDeletingUser] = useState<DouyinUser | null>(null);

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-10">
              <SelectionCheckbox
                label="Select all visible users"
                checked={allVisibleUsersSelected}
                indeterminate={someVisibleUsersSelected}
                disabled={loading || !users.length}
                onCheckedChange={onToggleAllVisible}
              />
            </TableHead>
            <TableHead>ID</TableHead>
            <TableHead>Name</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Topic</TableHead>
            <TableHead>Niche</TableHead>
            <TableHead>System</TableHead>
            <TableHead>Last Fetched</TableHead>
            <TableHead className="w-10">
              <span className="sr-only">Actions</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={9} className="text-center text-muted-foreground">
                Loading...
              </TableCell>
            </TableRow>
          ) : users.length ? (
            users.map((user) => (
              <UserTableRow
                key={user.id}
                fetchDisabled={fetchDisabled}
                selectedUserIds={selectedUserIds}
                statuses={statuses}
                systems={systems}
                user={user}
                onDelete={setDeletingUser}
                onEdit={setEditingUser}
                onToggleSelected={onToggleSelected}
                onUpdate={onUpdate}
              />
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={9} className="text-center text-muted-foreground">
                No users
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {editingUser && (
        <EditUserDialog
          user={editingUser}
          systems={systems}
          statuses={statuses}
          trigger={null}
          open
          onOpenChange={(open) => {
            if (!open) setEditingUser(null);
          }}
          onSubmit={onUpdate}
        />
      )}
      {deletingUser && (
        <DeleteUserDialog
          user={deletingUser}
          trigger={null}
          open
          onOpenChange={(open) => {
            if (!open) setDeletingUser(null);
          }}
          onDelete={onDelete}
        />
      )}
    </div>
  );
}
