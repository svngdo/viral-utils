import { type ChangeEvent, useState } from "react";
import { Button } from "@/components/Button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/Dialog";
import { ErrorText } from "@/components/ErrorText";
import { Input } from "@/components/Input";
import { LoadingText } from "@/components/LoadingText";
import { Textarea } from "@/components/Textarea";
import {
  DEFAULT_SYSTEM_FORM_DATA,
  DEFAULT_SYSTEM_FORM_ERRORS,
} from "@/pages/systems/constants";
import { useCreateSystem } from "@/pages/systems/hooks/useCreateSystem";
import type { CreateSystemInput, SystemFormError } from "@/pages/systems/types";

type CreateDialogProps = {
  onClose: () => void;
};

export function CreateDialog({ onClose }: CreateDialogProps) {
  const [formData, setFormData] = useState<CreateSystemInput>({
    ...DEFAULT_SYSTEM_FORM_DATA,
  });
  const [formErrors, setFormErrors] = useState<SystemFormError>({
    ...DEFAULT_SYSTEM_FORM_ERRORS,
  });

  const { mutateAsync, isPending, error, reset } = useCreateSystem();

  function handleChange(e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    const name = e.target.name;
    const value = e.target.value;

    if (error) reset();
    setFormData((prev) => ({ ...prev, [name]: value }));

    setFormErrors((prev) => ({ ...prev, [name]: "" }));
  }

  function validateFormData(data: CreateSystemInput): SystemFormError {
    return {
      name: data.name.trim().length < 2 ? "Name must be at least 2 characters" : "",
      description: "",
    };
  }

  async function handleSubmit(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault();

    if (isPending) return;

    const errors = validateFormData(formData);
    setFormErrors(errors);

    const hasError = Object.values(errors).some((value) => value !== "");
    if (hasError) return;

    try {
      await mutateAsync({
        name: formData.name.trim(),
        description: formData.description.trim(),
      });
      onClose();
    } catch {
      // The error is show via `error` from useCreateSystem; dialog stays open
    }
  }

  return (
    <Dialog onClose={onClose} isBusy={isPending}>
      <form noValidate onSubmit={handleSubmit}>
        {/* Header */}
        <DialogHeader>
          <DialogTitle>Create System</DialogTitle>
        </DialogHeader>

        {/* Content */}
        <DialogContent>
          <fieldset className="space-y-4" disabled={isPending}>
            <Input
              label="Name"
              name="name"
              placeholder="Enter system name"
              required
              error={formErrors.name}
              value={formData.name}
              onChange={handleChange}
            />
            <Textarea
              label="Description"
              name="description"
              placeholder="Enter system description"
              rows={3}
              value={formData.description}
              onChange={handleChange}
            />

            {isPending && <LoadingText text="Saving system..." />}
            {error && <ErrorText message={error.message} />}
          </fieldset>
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
