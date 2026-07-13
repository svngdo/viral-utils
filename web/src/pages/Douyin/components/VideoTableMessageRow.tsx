import { TableCell, TableRow } from "@/components/ui/table";

interface VideoTableMessageRowProps {
  colSpan: number;
  message: string;
}

export default function VideoTableMessageRow({ colSpan, message }: VideoTableMessageRowProps) {
  return (
    <TableRow>
      <TableCell colSpan={colSpan} className="text-center text-muted-foreground">
        {message}
      </TableCell>
    </TableRow>
  );
}
