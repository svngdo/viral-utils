import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/Button";
import { useTable } from "@/components/DataTable/useTable";
import { useTableState } from "@/components/DataTable/useTableState";
import { fetchSystems } from "@/pages/systems/api";
import { CreateDialog } from "@/pages/systems/CreateDialog";
import { columns, type SystemsMeta } from "@/pages/systems/columns";
import { DeleteDialog } from "@/pages/systems/DeleteDialog";
import { EditDialog } from "@/pages/systems/EditDialog";
import type { System } from "@/pages/systems/types";

export function SystemsPage() {
  const tableState = useTableState<System>({
    defaultSorting: [
      {
        id: "updatedAt",
        desc: true,
      },
    ],
  });

  const { status, data, error } = useQuery({
    queryKey: [
      "systems",
      tableState.pagination,
      tableState.sorting,
      tableState.debouncedSearch,
    ],
    queryFn: () =>
      fetchSystems({
        pagination: tableState.pagination,
        sorting: tableState.sorting,
        search: tableState.debouncedSearch,
      }),
  });

  const systems = data?.items ?? [];
  const totalSystems = data?.total ?? 0;

  const table = useTable<System, SystemsMeta>({
    pageData: systems,
    totalItems: totalSystems,
    getRowId: (row) => row.id,
    meta: {
      onEdit: (system) => setSystemToEdit(system),
      onDelete: (system) => setIdsToDelete([system.id]),
    },
    ...tableState,
  });

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [idsToDelete, setIdsToDelete] = useState<string[] | null>(null);
  const [systemToEdit, setSystemToEdit] = useState<System | null>(null);

  if (status === "error") return <span>Error: {error.message}</span>;

  return (
    <>
      <div className="flex w-full flex-col gap-3 p-3">
        <h1 className="text-2xl font-bold">System Page</h1>
        {/* Toolbar */}
        <div className="flex w-full items-center justify-between">
          {/* Toolbar left */}
          <div className="flex items-center gap-3">
            <input
              type="search"
              placeholder="Enter search..."
              className="rounded-md border px-3 py-2 text-sm font-medium"
              value={table.search}
              onChange={(e) => table.setSearch(e.target.value)}
            />
            <div>
              {table.selectedIds.length > 0 ? (
                <Button
                  variant="default"
                  onClick={() => setIdsToDelete(table.selectedIds)}
                >
                  Delete {table.selectedIds.length} items
                </Button>
              ) : (
                ""
              )}
            </div>
          </div>

          {/* Toolbar right */}
          <div>
            <Button variant="default" onClick={() => setShowCreateDialog(true)}>
              Add System
            </Button>
          </div>
        </div>

        {/* Table wrapper */}
        <div className="w-full overflow-x-auto">
          <table
            className={[
              "w-full table-fixed border-collapse",
              "[&_:is(th)]:text-left",
              "[&_:is(td,th)]:px-3",
              "[&_:is(td,th)]:py-2",
              "[&_:is(td,th)]:truncate",
            ].join(" ")}
          >
            <thead className="border-b border-gray-200">
              <tr>
                {columns.map((col) => (
                  <th key={col.id} className={col.headerClassName}>
                    {typeof col.header === "function"
                      ? col.header({ table, col })
                      : col.header}
                  </th>
                ))}
              </tr>
            </thead>

            <tbody className="divide-y divide-gray-200">
              {status === "pending" && (
                <tr>
                  <td colSpan={columns.length}>
                    <div className="flex items-center justify-center gap-2 py-6">
                      <div className="size-5 animate-spin rounded-full border-4 border-gray-300 border-t-blue-500" />
                      <span>Loading...</span>
                    </div>
                  </td>
                </tr>
              )}

              {systems.length === 0 && status !== "pending" ? (
                <tr>
                  <td colSpan={columns.length}>No results.</td>
                </tr>
              ) : (
                systems.map((row) => (
                  <tr key={row.id}>
                    {columns.map((col) => (
                      <td key={col.id} className={col.cellClassName}>
                        {typeof col.cell === "function"
                          ? // cell is a function
                            col.cell({ table, row, col })
                          : // cell is a literal value
                            col.cell}
                      </td>
                    ))}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {status !== "pending" && (
          <div className="flex items-center justify-between px-4">
            {/* Pagination left */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Rows per page</span>
              <select
                className="rounded border border-gray-400 px-3 py-2 text-sm"
                value={table.pageSize}
                onChange={(e) => table.setPageSize(Number(e.target.value))}
              >
                {[10, 20, 30, 50].map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </div>

            {/* Pagination right */}
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm text-gray-400">
                {table.pageIndex * table.pageSize + 1}-
                {table.pageIndex + 1 === table.totalPages
                  ? totalSystems
                  : table.pageIndex * table.pageSize + table.pageSize}{" "}
                of {totalSystems}
              </span>
              <div className="flex items-center justify-center gap-2">
                {table.canPreviousPage && (
                  <Button onClick={table.previousPage}>Prev</Button>
                )}
                {table.pageNumbers.map((pageItem) =>
                  typeof pageItem === "number" ? (
                    <Button
                      key={pageItem}
                      variant={table.pageIndex === pageItem ? "secondary" : "outline"}
                      onClick={() => table.setPageIndex(pageItem)}
                    >
                      {pageItem + 1}
                    </Button>
                  ) : (
                    <span key={pageItem}>...</span>
                  ),
                )}
                {table.canNextPage && <Button onClick={table.nextPage}>Next</Button>}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Create Dialog */}
      {showCreateDialog && <CreateDialog onClose={() => setShowCreateDialog(false)} />}

      {/* Edit Dialog */}
      {systemToEdit && (
        <EditDialog systemToEdit={systemToEdit} onClose={() => setSystemToEdit(null)} />
      )}

      {/* Delete Confirmation */}
      {idsToDelete && (
        <DeleteDialog idsToDelete={idsToDelete} onClose={() => setIdsToDelete(null)} />
      )}
    </>
  );
}
