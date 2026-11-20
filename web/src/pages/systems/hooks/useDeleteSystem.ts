import { useMutation, useQueryClient } from "@tanstack/react-query";
import { deleteSystems } from "@/pages/systems/api";

export function useDeleteSystems() {
  const queryClient = useQueryClient();

  const { mutateAsync, isPending, error } = useMutation({
    mutationFn: deleteSystems,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["systems"] }),
  });

  return {
    mutateAsync,
    isPending,
    error,
  };
}
