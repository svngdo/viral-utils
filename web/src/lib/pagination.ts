
export function getValidPageIndex(pageIndex: number, pageCount: number) {
  if (pageCount === 0) return 0;

  return Math.min(pageIndex, pageCount - 1);
}

