import type { ReactNode } from "react";
import { Button } from "@/components/Button";
import type { useTable } from "@/components/DataTable/useTable";
import { formatDateTime } from "@/lib/formatters";
import type { System } from "@/pages/systems/types";

export type SystemsMeta = {
  onEdit: (system: System) => void;
  onDelete: (system: System) => void;
};

export type ColumnDef<T, M = undefined> = {
  id: string;
  header?:
    | ReactNode
    | ((ctx: {
        table: ReturnType<typeof useTable<T, M>>;
        col: ColumnDef<T, M>;
      }) => ReactNode);
  headerClassName?: string;
  cell?:
    | ReactNode
    | ((ctx: {
        table: ReturnType<typeof useTable<T, M>>;
        row: T;
        col: ColumnDef<T, M>;
      }) => ReactNode);
  cellClassName?: string;
};

export const columns: ColumnDef<System, SystemsMeta>[] = [
  {
    id: "select",
    header: ({ table }) => (
      <input
        type="checkbox"
        className="mx-auto block size-5"
        checked={table.isAllPageRowsSelected}
        onChange={() => table.toggleAllPageRowsSelected()}
      />
    ),
    headerClassName: "w-10",
    cell: ({ table, row }) => (
      <input
        type="checkbox"
        className="mx-auto block size-5"
        checked={!!table.rowSelection[row.id]}
        onChange={() => table.togglePageRowSelected(row.id)}
      />
    ),
  },
  {
    id: "name",
    header: "Name",
    headerClassName: "w-40",
    cell: ({ row }) => row.name,
  },
  {
    id: "description",
    header: "Description",
    headerClassName: "w-64",
    cell: ({ row }) => row.description,
  },
  {
    id: "createdAt",
    header: "Created at",
    headerClassName: "w-28",
    cell: ({ row }) => formatDateTime(row.createdAt),
  },
  {
    id: "updatedAt",
    header: "Updated at",
    headerClassName: "w-28",
    cell: ({ row }) => formatDateTime(row.updatedAt),
  },
  {
    id: "actions",
    headerClassName: "w-36 text-right",
    cell: ({ row, table }) => (
      <div className="space-x-2">
        <Button onClick={() => table.meta.onEdit?.(row)}>Edit</Button>
        <Button variant="secondary" onClick={() => table.meta.onDelete?.(row)}>
          Delete
        </Button>
      </div>
    ),
    cellClassName: "text-right",
  },
];
