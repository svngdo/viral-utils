export function generatePageNumbers(
  currentPage: number,
  totalPages: number,
): (number | string)[] {
  if (totalPages <= 0) return [];

  const lastIndex = totalPages - 1;

  const start = currentPage < 3 ? 0 : currentPage - 1;
  const end = currentPage > lastIndex - 3 ? lastIndex : currentPage + 1;

  const candidates = [
    0,
    ...Array.from({ length: end - start + 1 }, (_, i) => start + i),
    totalPages - 1,
  ];

  // Remove out-of-range and duplicate indexes, then sort ascending.
  const pageIndexes = [
    ...new Set(candidates.filter((i) => i >= 0 && i <= lastIndex)),
  ].sort((a, b) => a - b);

  const result: (number | string)[] = [];

  pageIndexes.forEach((current, i) => {
    const prev = pageIndexes[i - 1];

    if (prev !== undefined && current - prev > 1) {
      result.push(`ellipsis-${i}`);
    }

    result.push(current);
  });

  return result;
}
