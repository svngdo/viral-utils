import { Spinner } from "@/components/Spinner";

type LoadingTextProps = {
  text: string;
};

export function LoadingText({ text = "" }: LoadingTextProps) {
  return (
    <p className="flex items-center gap-2 text-sm">
      <Spinner size="sm" />
      {text}
    </p>
  );
}
