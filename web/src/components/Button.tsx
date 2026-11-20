import type { ButtonHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

type ButtonVariant = "default" | "secondary" | "outline" | "ghost" | "destructive";

const variants: Record<ButtonVariant, string> = {
  default: "bg-gray-900 text-white hover:bg-gray-700",
  secondary: "bg-gray-200 text-gray-900 hover:bg-gray-300",
  outline: "border border-gray-300 bg-transparent text-gray-900 hover:bg-gray-100",
  ghost: "bg-transparent text-gray-900 hover:bg-gray-100",
  destructive: "bg-red-600 text-white hover:bg-red-700",
};

export function Button({
  variant = "default",
  type = "button",
  className = "",
  ...props
}: ButtonProps) {
  return (
    <button
      type={type}
      {...props}
      className={[
        "rounded-md px-3 py-2 text-sm font-medium transition-colors",
        "cursor-pointer",
        "disabled:cursor-not-allowed disabled:opacity-50",
        variants[variant],
        className,
      ].join(" ")}
    />
  );
}
