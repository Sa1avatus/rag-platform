import {beforeEach, describe, expect, it, vi} from "vitest";

import {api} from "./api";

const values = new Map<string, string>();

beforeEach(() => {
  values.clear();
  vi.stubGlobal("sessionStorage", {
    clear: () => values.clear(),
    getItem: (key: string) => values.get(key) ?? null,
    removeItem: (key: string) => values.delete(key),
    setItem: (key: string, value: string) => values.set(key, value),
  });
});

describe("api", () => {
  it("adds the API prefix and bearer token", async () => {
    values.set("rag-admin-token", "test-token");
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({status: "ok"}), {
        headers: {"Content-Type": "application/json"},
        status: 200,
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(api<{status: string}>("/v1/health")).resolves.toEqual({status: "ok"});
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/health",
      expect.objectContaining({
        headers: expect.objectContaining({Authorization: "Bearer test-token"}),
      }),
    );
  });

  it("clears a rejected admin session", async () => {
    values.set("rag-admin-token", "expired-token");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, {status: 401})));

    await expect(api("/v1/admin/projects")).rejects.toThrow("Sign in again");
    expect(values.has("rag-admin-token")).toBe(false);
  });

  it("uses an API error detail when available", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({detail: "Project is disabled"}), {
          headers: {"Content-Type": "application/json"},
          status: 409,
        }),
      ),
    );

    await expect(api("/v1/admin/projects")).rejects.toThrow("Project is disabled");
  });
});
