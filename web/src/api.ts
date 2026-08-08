export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = sessionStorage.getItem("rag-admin-token");
  const response = await fetch(`/api${path}`, { ...options, headers: {"Content-Type":"application/json", ...(token ? {Authorization:`Bearer ${token}`} : {}), ...options.headers} });
  if (response.status === 401) { sessionStorage.removeItem("rag-admin-token"); throw new Error("Your admin session was rejected. Sign in again."); }
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail ?? `Request failed (${response.status})`);
  return response.status === 204 ? (undefined as T) : response.json();
}
