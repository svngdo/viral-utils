import type { PageNumber, PaginationState } from "@/components/DataTable/types";

interface DataTablePaginationProps {
  total: number;
  pagination: PaginationState;
  onChange: (next: PaginationState) => void;
  pageSizeOptions?: number[];
  disabled?: boolean;
}

export function generatePageNumbers(
  currentPage: number,
  totalPages: number,
): PageNumber[] {
  if (totalPages <= 0) return [];

  const startIndex = currentPage <= 4 ? 0 : currentPage - 2;
  const endIndex = currentPage >= totalPages - 5 ? totalPages - 1 : currentPage + 2;

  const candidates = [
    0,
    ...Array.from(
      {
        length: endIndex - startIndex + 1,
      },
      (_, i) => startIndex + i,
    ),
    totalPages - 1,
  ];

  const validIndexes = candidates.filter((index) => index >= 0 && index < totalPages);
  const indexes = [...new Set(validIndexes)].sort((a, b) => a - b);
  const result: PageNumber[] = [];

  indexes.forEach((currIndex, i) => {
    const prevIndex = indexes[i - 1];

    if (prevIndex !== undefined && currIndex - prevIndex > 1) {
      result.push("...");
    }

    result.push(currIndex);
  });

  return result;
}

export function DataTablePagination({
  total,
  pagination,
  onChange,
  pageSizeOptions = [10, 20, 30, 50],
  disabled = false,
}: DataTablePaginationProps) {
  const { pageIndex, pageSize } = pagination;

  const pageCount = Math.ceil(total / pageSize);

  const pageNumbers = generatePageNumbers(pageIndex, pageCount);

  const canPreviousPage = pageIndex > 0;
  const canNextPage = pageIndex < pageCount - 1;

  const firstRow = total > 0 ? pageIndex * pageSize + 1 : 0;
  const lastRow = Math.min((pageIndex + 1) * pageSize, total);

  function changePage(pageIndex: number) {
    onChange({
      ...pagination,
      pageIndex,
    });
  }

  function changePageSize(pageSize: number) {
    onChange({
      pageIndex: 0,
      pageSize,
    });
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-4">
      {/* Page size */}
      <div className="flex items-center gap-3 text-sm">
        <label htmlFor="page-size">Rows per page</label>

        <select
          id="page-size"
          value={pageSize}
          disabled={disabled}
          onChange={(e) => changePageSize(Number(e.target.value))}
          className="rounded-md border bg-white px-2 py-1"
        >
          {pageSizeOptions.map((size) => (
            <option key={size} value={size}></option>
          ))}
        </select>
      </div>

      {/* Pagination controls */}
      <div className="flex flex-wrap items-center gap-4 text-sm">
        <span>
          {firstRow}-{lastRow} of {total}
        </span>

        <nav className="flex items-center gap-1">
          {canPreviousPage && (
            <button
              type="button"
              disabled={disabled}
              className="rounded px-3 py-1 hover:bg-gray-100 disabled:opacity-50"
              onClick={() => changePage(pageIndex - 1)}
            >
              &lt; Prev
            </button>
          )}

          {pageNumbers.map((page) =>
            typeof page === "number" ? (
              <button
                key={page}
                type="button"
                className={[
                  "size-8 rounded disabled:opacity-50",
                  pageIndex === page
                    ? "bg-gray-200 font-semibold"
                    : "hover:bg-gray-100",
                ].join(" ")}
              >
                {page + 1}
              </button>
            ) : (
              <span key={`${page}-${pageNumbers.indexOf(page)}`} className="px-1">
                ...
              </span>
            ),
          )}

          {canNextPage && (
            <button
              type="button"
              disabled={disabled}
              className="rounded px-3 py-1 hover:bg-gray-100 disabled:opacity-50"
              onClick={() => changePage(pageIndex + 1)}
            >
              Next &gt;
            </button>
          )}
        </nav>
      </div>
    </div>
  );
}
