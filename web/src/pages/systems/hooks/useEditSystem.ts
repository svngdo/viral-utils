import { useMutation, useQueryClient } from "@tanstack/react-query";
import { updateSystem } from "@/pages/systems/api";

export function useEditSystem() {
  const queryClient = useQueryClient();

  const { mutateAsync, isPending, error, reset } = useMutation({
    mutationFn: updateSystem,
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["systems"],
      }),
  });

  return {
    mutateAsync,
    isPending,
    error,
    reset,
  };
}
