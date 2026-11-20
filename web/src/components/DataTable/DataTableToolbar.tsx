import type { ReactNode } from "react";

interface DataTableToolbarProps {
  search: string;
  onSearchChange: (value: string) => void;
  searchPlaceHolder?: string;
  isUpdating?: boolean;
  actions?: ReactNode;
}

export function DataTableToolbar({
  search,
  onSearchChange,
  searchPlaceHolder = "Search...",
  isUpdating = false,
  actions,
}: DataTableToolbarProps) {
  return (
    <div className="flex w-full items-center justify-center gap-3">
      {/* Left side */}
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <input
          type="search"
          value={search}
          placeholder={searchPlaceHolder}
          onChange={(e) => onSearchChange(e.target.value)}
          className={[
            "w-full max-w-64 rounded-md border bg-white px-3 py-2 text-sm outline-none",
            "focus:ring-2 focus:ring-blue-500",
          ].join(" ")}
        />

        <span className="shrink-0 text-sm text-gray-500">
          {isUpdating ? "Updating..." : ""}
        </span>
      </div>

      {/* Right side */}
      {actions && <div className="shrink-0">{actions}</div>}
    </div>
  );
}
