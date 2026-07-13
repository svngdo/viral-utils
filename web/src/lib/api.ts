export const apiUrl = import.meta.env.VITE_API_URL;

export function apiAbsoluteUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith("http")) return pathOrUrl;
  if (!apiUrl) throw new Error("Missing VITE_API_URL");
  return `${apiUrl}${pathOrUrl}`;
}

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(apiAbsoluteUrl(path), options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      detail = await res.text();
    }
    throw new Error(`Request failed: ${res.status} ${path}${detail ? ` - ${detail}` : ""}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}
