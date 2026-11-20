import { type Ref, type TextareaHTMLAttributes, useId } from "react";

type TextareaProps = TextareaHTMLAttributes<HTMLTextAreaElement> & {
  ref?: Ref<HTMLTextAreaElement>;
  label: string;
  error?: string;
};

export function Textarea({ ref, id, label, rows, error, ...props }: TextareaProps) {
  const generatedId = useId();
  const textareaId = id ?? props.name ?? generatedId;

  return (
    <div className="space-y-1.5">
      <label htmlFor={textareaId} className="block text-sm font-medium">
        {label}
      </label>

      <textarea
        {...props}
        ref={ref}
        id={textareaId}
        rows={rows}
        className={`w-full rounded-md border border-gray-300 px-3 py-2 text-sm`}
      />
    </div>
  );
}
