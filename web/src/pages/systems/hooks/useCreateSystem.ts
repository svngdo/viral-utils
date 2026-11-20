import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createSystem } from "@/pages/systems/api";

export function useCreateSystem() {
  const queryClient = useQueryClient();

  const { mutateAsync, isPending, error, reset } = useMutation({
    mutationFn: createSystem,
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["systems"],
      });
    },
  });

  return {
    mutateAsync,
    isPending,
    error,
    reset,
  };
}
