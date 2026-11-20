import { type InputHTMLAttributes, type Ref, useId } from "react";

type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  ref?: Ref<HTMLInputElement>;
  label: string;
  error?: string;
};

export function Input({ ref, id, label, error, ...props }: InputProps) {
  const generatedId = useId();
  const inputId = id ?? props.name ?? generatedId;

  return (
    <div className="space-y-1.5">
      <label htmlFor={inputId} className="block text-sm font-medium text-gray-700">
        {label}
      </label>

      <input
        {...props}
        ref={ref}
        id={inputId}
        className={[
          "w-full rounded-md border bg-white px-3 py-2 text-sm outline-none",
          "placeholder:text-gray-400 focus:ring-2",
          error
            ? "border-red-500 focus:border-red-500 focus:ring-red-100"
            : "border-gray-300 focus:border-gray-500 focus:ring-gray-200",
        ].join(" ")}
      />

      {error && (
        <p id={`${inputId}-error`} className="text-sm text-red-600">
          {error}
        </p>
      )}
    </div>
  );
}

Input.displayName = "Input";
