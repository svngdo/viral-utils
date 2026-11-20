type ErrorTextProps = {
  message: string;
};
export function ErrorText({ message = "" }: ErrorTextProps) {
  return <p className="text-sm text-red-600">{message}</p>;
}
