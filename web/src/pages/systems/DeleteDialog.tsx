import { Button } from "@/components/Button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/Dialog";
import { ErrorText } from "@/components/ErrorText";
import { LoadingText } from "@/components/LoadingText";
import { useDeleteSystems } from "@/pages/systems/hooks/useDeleteSystem";

type DeleteDialogProps = {
  idsToDelete: Array<string>;
  onClose: () => void;
};

export function DeleteDialog({ idsToDelete, onClose }: DeleteDialogProps) {
  const { mutateAsync, isPending, error } = useDeleteSystems();

  async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();
    try {
      await mutateAsync({ ids: idsToDelete });
      onClose();
    } catch {
      // Keep the dialog open on error
    }
  }

  return (
    <Dialog onClose={onClose} isBusy={isPending}>
      <form noValidate onSubmit={handleSubmit}>
        {/* Header */}
        <DialogHeader>
          <DialogTitle>Delete Confirmation</DialogTitle>
        </DialogHeader>

        {/* Content */}
        <DialogContent>
          <div className="space-y-4">
            <p>Do you want to delete</p>
            <p>{idsToDelete.join("\n")}</p>
            {isPending && <LoadingText text="Deleting systems..." />}
            {error && <ErrorText message={error.message} />}
          </div>
        </DialogContent>

        {/* Footer */}
        <DialogFooter>
          <Button type="button" onClick={onClose} disabled={isPending}>
            Cancel
          </Button>
          <Button type="submit" disabled={isPending}>
            Ok
          </Button>
        </DialogFooter>
      </form>
    </Dialog>
  );
}
