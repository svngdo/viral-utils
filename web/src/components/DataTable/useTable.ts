import type {
  ColumnSort,
  PaginationState,
  TableState,
} from "@/components/DataTable/types";
import { generatePageNumbers } from "@/components/DataTable/utils";

type UseTableProps<T, M> = TableState<T> & {
  pageData: T[];
  totalItems: number;
  getRowId?: (row: T, index: number) => string;
  meta?: M;
};

export function useTable<T, M = undefined>({
  pageData,
  totalItems,
  getRowId,
  meta,

  // Pagination
  pagination,
  setPagination,

  // Sorting
  sorting,
  setSorting,

  // Search
  search,
  setSearch,
  debouncedSearch,

  // Row Selection
  rowSelection,
  setRowSelection,
}: UseTableProps<T, M>) {
  /**
   * Pagination
   * */
  const totalPages = Math.ceil(totalItems / pagination.pageSize);
  const canPreviousPage = pagination.pageIndex > 0;
  const canNextPage = pagination.pageIndex < totalPages - 1;

  function setPageIndex(index: PaginationState["pageIndex"]) {
    if (!Number.isFinite(index)) return;

    // Keep the `lastPageIndex` >= 0
    const lastPageIndex = Math.max(0, totalPages - 1);

    // Math.min keep the `nextIndex` <= lastPageIndex
    // Math.max keep the `nextIndex` >= 0
    const nextIndex = Math.max(0, Math.min(index, lastPageIndex));

    if (nextIndex === pagination.pageIndex) return;

    setPagination((prev) => ({
      ...prev,
      pageIndex: nextIndex,
    }));

    clearRowSelection();
  }

  function setPageSize(size: PaginationState["pageSize"]) {
    if (!Number.isInteger(size) || size <= 0) return;
    if (size === pagination.pageSize) return;

    setPagination({
      pageIndex: 0,
      pageSize: size,
    });

    clearRowSelection();
  }

  function previousPage() {
    const currentPageIndex = pagination.pageIndex;

    if (currentPageIndex <= 0) return;

    setPageIndex(currentPageIndex - 1);
  }

  function nextPage() {
    const currentPageIndex = pagination.pageIndex;

    if (currentPageIndex >= totalPages - 1) return;

    setPageIndex(currentPageIndex + 1);
  }

  /**
   * Sorting
   * */
  function toggleSorting(sortId: ColumnSort<T>["id"]) {
    // Check if sorting key exists
    setSorting((prev) => {
      const existing = prev.find((sort) => sort.id === sortId);

      // Unsorted -> Ascending
      if (!existing) {
        return [
          ...prev,
          {
            id: sortId,
            desc: false,
          },
        ];
      }

      // Ascending -> Descending
      if (!existing.desc) {
        return prev.map((sort) =>
          sort.id === sortId
            ? {
                ...sort,
                desc: true,
              }
            : sort,
        );
      }

      // Descending -> Unsorted
      return prev.filter((sort) => sort.id !== sortId);
    });

    setPageIndex(0);
    clearRowSelection();
  }

  function clearSorting() {
    setSorting([]);
    setPageIndex(0);
    clearRowSelection();
  }

  /**
   * Filtering
   */

  /**
   * Row Selection
   */
  function resolveRowId(row: T, index: number) {
    return getRowId ? getRowId(row, index) : String(index);
  }

  const pageRowIds = pageData.map(resolveRowId);
  const selectedIds = pageRowIds.filter((id) => rowSelection[id]);
  const isAllPageRowsSelected =
    pageRowIds.length > 0 && pageRowIds.every((id) => rowSelection[id]);
  const isSomePageRowsSelected =
    pageRowIds.some((id) => rowSelection[id]) && !isAllPageRowsSelected;

  function togglePageRowSelected(id: string) {
    if (!pageRowIds.includes(id)) return;

    setRowSelection((prev) => {
      const next = { ...prev };

      if (next[id]) {
        delete next[id]; // removes the property, not just its value
      } else {
        next[id] = true;
      }

      return next;
    });
  }

  function toggleAllPageRowsSelected() {
    setRowSelection((prev) => {
      const isAllPageRowsSelected =
        pageRowIds.length > 0 && pageRowIds.every((id) => prev[id]);
      if (isAllPageRowsSelected) {
        return {};
      }
      return Object.fromEntries(pageRowIds.map((id) => [id, true]));
    });
  }

  function clearRowSelection() {
    setRowSelection({});
  }

  return {
    // Pagination
    pageIndex: pagination.pageIndex,
    pageSize: pagination.pageSize,
    totalPages,
    pageNumbers: generatePageNumbers(pagination.pageIndex, totalPages),
    canPreviousPage,
    canNextPage,
    setPageIndex,
    setPageSize,
    previousPage,
    nextPage,

    // Sorting
    sorting,
    toggleSorting,
    clearSorting,

    // Search
    search,
    setSearch,
    debouncedSearch,

    // Row Selection
    rowSelection,
    setRowSelection,
    clearSelection: clearRowSelection,
    selectedIds,
    pageRowIds,
    isAllPageRowsSelected,
    isSomePageRowsSelected,
    togglePageRowSelected,
    toggleAllPageRowsSelected,

    meta: meta as M,
  };
}
