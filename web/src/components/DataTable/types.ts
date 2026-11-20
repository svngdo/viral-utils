import type { Dispatch, ReactNode, SetStateAction } from "react";

export type ColumnDef<T> = {
  /** Unique column ID. Also used as the API sort key. */
  id: string;

  header: ReactNode | ((row: T) => ReactNode);
  headerClassName?: string;

  /** Access a property or derive a value from a row. */
  accessorKey?: keyof T;
  accessorFn?: (row: T) => unknown;

  /** Custom cell renderer. Takes precedence over accessor. */
  cell?: (row: T) => ReactNode;
  cellClassName?: string;

  /** Column functions */
  sortable?: boolean;
  width?: string;
};

export type PaginationState = {
  pageIndex: number;
  pageSize: number;
};

export type ColumnSort<T> = {
  id: keyof T;
  desc: boolean;
};

export type SortingState<T> = ColumnSort<T>[];

export type RowSelectionState = Record<string, boolean>;

export type TableState<T> = {
  // Pagination
  pagination: PaginationState;
  setPagination: Dispatch<SetStateAction<PaginationState>>;

  // Sorting
  sorting: SortingState<T>;
  setSorting: Dispatch<SetStateAction<SortingState<T>>>;

  // Filter
  search: string;
  setSearch: Dispatch<SetStateAction<string>>;
  debouncedSearch: string;

  // Row Selection
  rowSelection: RowSelectionState;
  setRowSelection: Dispatch<SetStateAction<RowSelectionState>>;
};
