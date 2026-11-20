import { type ReactNode, useEffect, useRef } from "react";
import { Button } from "@/components/Button";

type DialogProps = {
  onClose: () => void;
  children?: ReactNode;
  isBusy?: boolean;
};

export function Dialog({ onClose, children, isBusy = false }: DialogProps) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (dialog && !dialog.open) dialog.showModal();
    dialog?.querySelector<HTMLElement>("input, textarea,select")?.focus();
    // No cleanup needed
    // Unmounting removes the dialog
  }, []);

  return (
    <dialog
      ref={ref}
      onClose={() => {
        if (isBusy) {
          ref.current?.showModal(); // closed mid-request, reopen it
          return;
        }
        onClose();
      }}
      onCancel={(e) => {
        // `cancel` fires on Esc, before the dialog closes
        if (isBusy) e.preventDefault();
      }}
      className={[
        "m-auto w-full max-w-md rounded-lg border border-gray-200 bg-white p-4 shadow-lg",
        "backdrop:bg-black/80",
      ].join(" ")}
    >
      <Button
        variant="ghost"
        type="button"
        disabled={isBusy}
        onClick={() => {
          if (isBusy) return;
          // fires native `close` -> onClose
          ref.current?.close();
        }}
        className="absolute top-3 right-3 p-2 text-gray-500"
        aria-label="Close dialog"
      >
        ✕
      </Button>
      {children}
    </dialog>
  );
}

export function DialogHeader({ children }: { children: ReactNode }) {
  return <div className="mb-4 border-b border-gray-200 pb-2">{children}</div>;
}

export function DialogTitle({ children }: { children: ReactNode }) {
  return <h2 className="text-lg font-semibold">{children}</h2>;
}

export function DialogContent({ children }: { children: ReactNode }) {
  return <div>{children}</div>;
}

export function DialogFooter({ children }: { children: ReactNode }) {
  return <div className="mt-4 flex justify-end gap-2">{children}</div>;
}
