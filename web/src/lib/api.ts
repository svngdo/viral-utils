type QueryValue = string | number | boolean | undefined;

interface ApiOptions extends RequestInit {
  params?: Record<string, QueryValue>;
}

type ErrorBody = {
  detail?: string | { message?: string } | { msg?: string }[];
};

const baseUrl = import.meta.env.VITE_API_URL;

function getErrorMessage(result: ErrorBody | null, status: number): string {
  const detail = result?.detail;
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail) && typeof detail?.message === "string")
    return detail.message;
  if (Array.isArray(detail) && typeof detail[0]?.msg === "string") return detail[0].msg;
  return `Request failed (${status})`;
}

export async function api<T>(path: string, options: ApiOptions = {}): Promise<T> {
  if (!baseUrl?.trim()) {
    throw new Error("Set VITE_API_URL in web/.env before making API requests");
  }

  const { params, ...requestOptions } = options;

  const url = new URL(`${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined) {
      url.searchParams.set(key, String(value));
    }
  }

  const headers = new Headers(requestOptions.headers);
  if (requestOptions.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(url, { ...requestOptions, headers });

  if (response.status === 204) {
    return undefined as T;
  }

  const result = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(getErrorMessage(result, response.status));
  }

  return result as T;
}
