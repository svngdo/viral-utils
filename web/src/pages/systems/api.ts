import type { PaginationState, SortingState } from "@/components/DataTable/types";
import { api } from "@/lib/api";
import type {
  System,
  SystemFormData,
  SystemList,
  UpdateSystemInput,
} from "@/pages/systems/types";

interface SystemDto {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

// Convert SystemDto -> System
function toSystem(dto: SystemDto): System {
  return {
    id: dto.id,
    name: dto.name,
    description: dto.description,
    createdAt: new Date(dto.created_at),
    updatedAt: new Date(dto.updated_at),
  };
}

export async function createSystem(data: SystemFormData) {
  const response = await api<SystemDto>("/systems", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  return toSystem(response);
}

export type FetchSystemsParams = {
  pagination: PaginationState;
  sorting: SortingState<System>;
  search: string;
};

interface FetchSystemsResponse {
  items: SystemDto[];
  total: number;
}

export async function fetchSystems({
  pagination,
  sorting,
  search,
}: FetchSystemsParams): Promise<SystemList> {
  const SORT_COLUMNS = {
    id: "id",
    name: "name",
    description: "description",
    createdAt: "created_at",
    updatedAt: "updated_at",
  };

  const response = await api<FetchSystemsResponse>("/systems", {
    params: {
      q: search,
      order_by: sorting
        .map(({ id, desc }) => `${SORT_COLUMNS[id]}${desc ? " desc" : ""}`)
        .join(","),
      limit: pagination.pageSize,
      offset: pagination.pageIndex * pagination.pageSize,
    },
  });

  return {
    items: response.items.map(toSystem),
    total: response.total,
  };
}

export async function updateSystem({ id, ...data }: UpdateSystemInput) {
  const response = await api<SystemDto>(`/systems/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });

  return toSystem(response);
}

export async function deleteSystems({ ids }: { ids: Array<string> }) {
  await api<void>("/systems", {
    method: "DELETE",
    body: JSON.stringify({ ids }),
  });
}
