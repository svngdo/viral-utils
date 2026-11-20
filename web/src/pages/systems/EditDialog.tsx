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
import { Textarea } from "@/components/Textarea";
import { DEFAULT_SYSTEM_FORM_ERRORS } from "@/pages/systems/constants";
import { useEditSystem } from "@/pages/systems/hooks/useEditSystem";
import type { System, SystemFormData, SystemFormError } from "@/pages/systems/types";

type EditDialogProps = {
  onClose: () => void;
  systemToEdit: System;
};

export function EditDialog({ onClose, systemToEdit }: EditDialogProps) {
  const [formData, setFormData] = useState<SystemFormData>({
    ...(systemToEdit as SystemFormData),
  });
  const [formErrors, setFormErrors] = useState<SystemFormError>({
    ...DEFAULT_SYSTEM_FORM_ERRORS,
  });

  const { mutateAsync, isPending, error, reset } = useEditSystem();

  function handleChange(e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    const { name, value } = e.target;
    if (error) reset();
    setFormErrors((prev) => ({ ...prev, [name]: "" }));
    setFormData((prev) => ({ ...prev, [name]: value }));
  }

  async function handleSubmit() {
    try {
      await mutateAsync({
        id: systemToEdit.id,
        name: formData.name.trim(),
        description: formData.description.trim(),
      });
      onClose();
    } catch {
      // Keep dialog open on error
    }
  }

  return (
    <Dialog onClose={onClose}>
      <form
        noValidate
        onSubmit={async (e) => {
          e.preventDefault();
          await handleSubmit();
        }}
      >
        <DialogHeader>
          <DialogTitle>Edit System</DialogTitle>
        </DialogHeader>

        <DialogContent>
          <fieldset className="space-y-4" disabled={isPending}>
            <Input
              label="Name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              error={formErrors.name}
            />
            <Textarea
              label="Description"
              name="description"
              value={formData.description}
              onChange={handleChange}
              error={formErrors.description}
            />

            {error && <ErrorText message={error.message} />}
          </fieldset>
        </DialogContent>

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
