import { useEffect, useRef } from "react";

interface SelectionCheckboxProps {
  label: string;
  checked: boolean;
  indeterminate?: boolean;
  disabled?: boolean;
  onCheckedChange: (checked: boolean) => void;
}

export function SelectionCheckbox({
  label,
  checked,
  indeterminate = false,
  disabled = false,
  onCheckedChange,
}: SelectionCheckboxProps) {
  const checkboxRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (checkboxRef.current) checkboxRef.current.indeterminate = indeterminate;
  }, [indeterminate]);

  return (
    <input
      ref={checkboxRef}
      type="checkbox"
      aria-label={label}
      checked={checked}
      disabled={disabled}
      className="size-4 accent-primary"
      onChange={(event) => onCheckedChange(event.currentTarget.checked)}
    />
  );
}
