type SpinnerProps = {
  size?: "sm" | "md" | "lg";
};

export function Spinner({ size = "md" }: SpinnerProps) {
  const sizes = {
    sm: "h-4 w-4 border-2",
    md: "h-6 w-6 border-2",
    lg: "h-8 w-8 border-[3px]",
  };

  return (
    <span
      role="status"
      aria-label="Loading"
      className={`inline-block animate-spin rounded-full border-solid border-current border-r-transparent ${sizes[size]}`}
    />
  );
}
