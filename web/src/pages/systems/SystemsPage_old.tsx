import { type ChangeEvent, useMemo, useState } from "react";
import { Button } from "@/components/Button";
import { useDebounce } from "@/hooks/useDebounce";
import { formatDateTime } from "@/lib/formatters";
import { generatePageNumbers } from "@/lib/pagination";
import { CreateDialog } from "@/pages/systems/CreateDialog";
import {
  DEFAULT_PAGINATION_STATE,
  DEFAULT_SORTING_STATE,
} from "@/pages/systems/constants";
import { useSystems } from "@/pages/systems/hooks/useSystems";
import type { PaginationState, SortingState } from "@/pages/systems/types";

export function SystemsPage() {
  const [selectedRowIds, setSelectedRowIds] = useState<string[]>([]);
  const [sorting, setSorting] = useState<SortingState>({ ...DEFAULT_SORTING_STATE });
  const [pagination, setPagination] = useState<PaginationState>({
    ...DEFAULT_PAGINATION_STATE,
  });
  const [search, setSearch] = useState("");

  const debouncedSearch = useDebounce(search);

  const [showCreateDialog, setShowCreateDialog] = useState(false);

  const { isPending, error, data, isFetching, isPlaceholderData } = useSystems({
    searchQuery: debouncedSearch,
    sortBy: sorting.sortBy,
    sortOrder: sorting.sortOrder,
    pageSize: pagination.pageSize,
    pageIndex: pagination.pageIndex,
  });

  const isUpdating = isFetching || search !== debouncedSearch;

  const systems = data?.items ?? [];
  const systemCount = data?.total ?? 0;

  const pageCount = Math.ceil(systemCount / pagination.pageSize);
  const pageNumbers = generatePageNumbers(pagination.pageIndex, pageCount);

  const canPreviousPage = pagination.pageIndex > 0;
  const canNextPage = pagination.pageIndex < pageCount - 1;

  const selectedIds = useMemo(() => new Set(selectedRowIds), [selectedRowIds]);
  const rowIds = useMemo(() => systems.map((system) => system.id), [systems]);
  const isAllRowsSelected =
    rowIds.length > 0 && rowIds.every((id) => selectedIds.has(id));

  function resetRowSelection() {
    setSelectedRowIds([]);
  }

  function changePageIndex(pageIndex: PaginationState["pageIndex"]) {
    setPagination((prev) => ({ ...prev, pageIndex }));
  }

  function handleRowSelect(systemId: string) {
    setSelectedRowIds((prev) =>
      prev.includes(systemId)
        ? prev.filter((id) => id !== systemId)
        : [...prev, systemId],
    );
  }

  function handlePageSizeChange(e: ChangeEvent<HTMLSelectElement>) {
    setPagination((prev) => ({
      ...prev,
      pageIndex: 0,
      pageSize: Number(e.target.value),
    }));

    resetRowSelection();
  }

  function handleSelectAll() {
    setSelectedRowIds(isAllRowsSelected ? [] : rowIds);
  }

  function handleSearchChange(value: string) {
    setSearch(value);
    changePageIndex(0);
    resetRowSelection();
  }

  function getNextSorting(
    prev: SortingState,
    sortBy: SortingState["sortBy"],
  ): SortingState {
    /**
     * Toggle order: asc -> desc -> default
     */

    // Handle toggle default sort by column case
    if (
      sortBy === DEFAULT_SORTING_STATE.sortBy &&
      prev.sortOrder === DEFAULT_SORTING_STATE.sortOrder
    ) {
      return { sortBy, sortOrder: "asc" };
    }

    if (prev.sortBy !== sortBy) {
      return { sortBy, sortOrder: "asc" };
    }

    if (prev.sortOrder === "asc") {
      return { sortBy, sortOrder: "desc" };
    }

    return DEFAULT_SORTING_STATE;
  }

  function handleSortChange(sortBy: SortingState["sortBy"]) {
    setSorting((prev) => getNextSorting(prev, sortBy));
    changePageIndex(0);
    resetRowSelection();
  }

  function handlePageChange(pageIndex: number) {
    changePageIndex(pageIndex);
    resetRowSelection();
  }

  function handlePreviousPage() {
    if (pagination.pageIndex > 0) {
      setPagination((prev) => ({ ...prev, pageIndex: prev.pageIndex - 1 }));
      resetRowSelection();
    }
  }

  function handleNextPage() {
    if (pagination.pageIndex < pageCount - 1) {
      setPagination((prev) => ({ ...prev, pageIndex: prev.pageIndex + 1 }));
      resetRowSelection();
    }
  }

  function handleOpenCreateDialog() {
    setShowCreateDialog(true);
  }

  function handleCloseCreateDialog() {
    setShowCreateDialog(false);
  }

  if (isPending) {
    return "Loading...";
  }

  if (error) {
    return `An error has occurred: ${error.message}`;
  }

  return (
    <div className="flex w-full flex-col gap-3 px-4 py-2">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        {/* Left Toolbar */}
        <div className="flex w-full items-center gap-2">
          <input
            type="search"
            placeholder="Enter search"
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full max-w-64 rounded-md border bg-white px-3 py-2 text-sm outline-none"
          />
          <div>{isPlaceholderData || isFetching ? "Updating..." : ""}</div>
        </div>

        {/* Right Toolbar */}
        <div>
          <Button type="button" onClick={handleOpenCreateDialog}>
            Add
          </Button>
        </div>
      </div>
      {/* Table */}
      <table
        className={[
          "w-full table-fixed border-collapse",
          "[&_:is(td,th)]:px-4",
          "[&_:is(td,th)]:py-3",
          "[&_:is(td,th)]:align-middle",
          "[&_input[type=checkbox]]:size-5",
          "[&_input[type=checkbox]]:block",
          "[&_input[type=checkbox]]:mx-auto",
        ].join(" ")}
      >
        <thead className="border-b border-gray-200">
          <tr>
            <th className="w-14">
              <input
                type="checkbox"
                checked={isAllRowsSelected}
                disabled={isUpdating}
                onChange={handleSelectAll}
              />
            </th>
            <th className="w-48 text-left">
              <button
                type="button"
                className="flex items-center gap-2 px-3 py-1"
                onClick={() => handleSortChange("name")}
              >
                Name{" "}
                <span>
                  {sorting.sortBy !== "name"
                    ? "↕"
                    : sorting.sortOrder === "asc"
                      ? "↑"
                      : "↓"}
                </span>
              </button>
            </th>
            <th className="text-left">
              <button
                type="button"
                className="flex items-center gap-2 px-3 py-1"
                onClick={() => handleSortChange("description")}
              >
                Description{" "}
                <span>
                  {sorting.sortBy !== "description"
                    ? "↕"
                    : sorting.sortOrder === "asc"
                      ? "↑"
                      : "↓"}
                </span>
              </button>
            </th>
            <th className="text-left">
              <button
                type="button"
                className="flex items-center gap-2 px-3 py-1"
                onClick={() => handleSortChange("created_at")}
              >
                Created At{" "}
                <span>
                  {sorting.sortBy !== "created_at"
                    ? "↕"
                    : sorting.sortOrder === "asc"
                      ? "↑"
                      : "↓"}
                </span>
              </button>
            </th>
            <th className="text-left">
              <button
                type="button"
                className="flex items-center gap-2 px-3 py-1"
                onClick={() => handleSortChange("updated_at")}
              >
                Updated At{" "}
                <span>
                  {sorting.sortBy !== "updated_at"
                    ? "↕"
                    : sorting.sortOrder === "asc"
                      ? "↑"
                      : "↓"}
                </span>
              </button>
            </th>
            <th className="w-24 text-right">Actions</th>
          </tr>
        </thead>

        <tbody className="divide-y divide-gray-300">
          {systems.map((system) => (
            <tr
              key={system.id}
              className={
                selectedIds.has(system.id) ? "bg-blue-100" : "hover:bg-gray-100"
              }
            >
              <td>
                <input
                  type="checkbox"
                  name="selected"
                  checked={selectedIds.has(system.id)}
                  disabled={isUpdating}
                  onChange={() => handleRowSelect(system.id)}
                />
              </td>
              <td className="truncate">{system.name}</td>
              <td className="truncate">{system.description}</td>
              <td className="truncate">{formatDateTime(system.createdAt)}</td>
              <td className="truncate">{formatDateTime(system.updatedAt)}</td>
              <td className="text-right">{/* Action buttons */}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination */}
      <div className="flex justify-between">
        {/* Page size */}
        <div className="flex items-center gap-3">
          <span>Rows per page</span>
          <select
            name="pageSize"
            value={pagination.pageSize}
            onChange={handlePageSizeChange}
          >
            {[10, 20, 30, 50].map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>

        {/* Right */}
        <div className="flex items-center justify-between gap-4">
          <div>
            {systemCount > 0 ? pagination.pageIndex * pagination.pageSize + 1 : 0}-
            {Math.min((pagination.pageIndex + 1) * pagination.pageSize, systemCount)} of{" "}
            {systemCount}
          </div>

          {/* Page numbers */}
          <div className="flex items-center gap-1">
            {canPreviousPage && (
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded px-3 py-1"
                onClick={handlePreviousPage}
              >
                {"<"} Prev
              </button>
            )}
            {pageNumbers.map((pageIndex) =>
              typeof pageIndex === "number" ? (
                <button
                  key={pageIndex}
                  type="button"
                  className={`size-8 rounded ${pagination.pageIndex === pageIndex ? "bg-gray-300" : ""}`}
                  onClick={() => handlePageChange(pageIndex)}
                >
                  {pageIndex + 1}
                </button>
              ) : (
                <span key={pageIndex}>...</span>
              ),
            )}
            {canNextPage && (
              <button
                type="button"
                className="rounded px-2 py-1"
                onClick={handleNextPage}
              >
                Next {">"}
              </button>
            )}
          </div>
        </div>
      </div>

      <CreateDialog open={showCreateDialog} onClose={handleCloseCreateDialog} />
    </div>
  );
}
