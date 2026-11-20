import { useState } from "react";
import type {
  PaginationState,
  RowSelectionState,
  SortingState,
} from "@/components/DataTable/types";
import { useDebounce } from "@/hooks/useDebounce";

type UseTableStateProps<T> = {
  defaultPagination?: PaginationState;
  defaultSorting?: SortingState<T>;
};

export function useTableState<T>({
  defaultPagination = {
    pageIndex: 0,
    pageSize: 10,
  },
  defaultSorting = [],
}: UseTableStateProps<T> = {}) {
  // Pagination
  const [pagination, setPagination] = useState<PaginationState>(defaultPagination);

  // Sorting
  const [sorting, setSorting] = useState<SortingState<T>>(defaultSorting);

  // Filtering
  const [search, setSearch] = useState("");
  const debouncedSearch = useDebounce<string>(search);

  // Row Selection
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  return {
    // Pagination
    pagination,
    setPagination,

    // Sorting
    sorting,
    setSorting,

    // Filter
    search,
    setSearch,
    debouncedSearch,

    // Row Selection
    rowSelection,
    setRowSelection,
  };
}
