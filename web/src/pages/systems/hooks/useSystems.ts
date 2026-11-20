import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { fetchSystems } from "@/pages/systems/api";
import type { SystemListParams } from "@/pages/systems/types";

export function useSystems(params: SystemListParams) {
  const { isPending, isError, error, data, isFetching, isPlaceholderData } = useQuery({
    queryKey: ["systems", params],
    queryFn: () => fetchSystems(params),
    placeholderData: keepPreviousData,
  });

  return {
    isPending,
    isError,
    error,
    data,
    isFetching,
    isPlaceholderData,
  };
}
