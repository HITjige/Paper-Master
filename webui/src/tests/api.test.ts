import { beforeEach, describe, expect, it, vi } from "vitest";

import { deleteSession, fetchSessionMessages } from "@/lib/api";
import { fetchKBStats } from "@/lib/kb-api";

describe("webui API helpers", () => {
  beforeEach(() => {
    delete (window as unknown as Record<string, unknown>).__NANOBOT_API_TOKEN__;
    delete (window as unknown as Record<string, unknown>).__NANOBOT_API_URL__;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ deleted: true, key: "websocket:chat-1", messages: [] }),
      }),
    );
  });

  it("percent-encodes websocket keys when fetching session history", async () => {
    await fetchSessionMessages("tok", "websocket:chat-1");

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("/api/sessions/websocket%3Achat-1/messages");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer tok");
  });

  it("percent-encodes websocket keys when deleting a session", async () => {
    await deleteSession("tok", "websocket:chat-1");

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("/api/sessions/websocket%3Achat-1/delete");
    expect(new Headers(init?.headers).get("Authorization")).toBe("Bearer tok");
  });

  it("refreshes an expired token and retries session history once", async () => {
    const history = {
      key: "websocket:chat-1",
      created_at: null,
      updated_at: null,
      messages: [{ role: "assistant", content: "restored" }],
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 401, json: async () => ({}) })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          token: "fresh-token",
          ws_path: "/",
          expires_in: 300,
          api_url: "http://127.0.0.1:18790",
        }),
      })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => history });
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      fetchSessionMessages("expired-token", "websocket:chat-1"),
    ).resolves.toEqual(history);

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/api/sessions/websocket%3Achat-1/messages",
      "/webui/bootstrap",
      "/api/sessions/websocket%3Achat-1/messages",
    ]);
    expect(
      new Headers(fetchMock.mock.calls[0][1]?.headers).get("Authorization"),
    ).toBe("Bearer expired-token");
    expect(
      new Headers(fetchMock.mock.calls[2][1]?.headers).get("Authorization"),
    ).toBe("Bearer fresh-token");
  });

  it("shares a token refreshed by the KB API with session history", async () => {
    const globals = window as unknown as Record<string, unknown>;
    globals.__NANOBOT_API_TOKEN__ = "expired-token";
    globals.__NANOBOT_API_URL__ = "http://127.0.0.1:18790";
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 401, json: async () => ({}) })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          token: "fresh-token",
          ws_path: "/",
          expires_in: 300,
          api_url: "http://127.0.0.1:19999",
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ paper_count: 1, chunk_count: 2 }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          key: "websocket:chat-1",
          created_at: null,
          updated_at: null,
          messages: [{ role: "assistant", content: "still here" }],
        }),
      });
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchKBStats()).resolves.toMatchObject({ paper_count: 1 });
    await expect(
      fetchSessionMessages("stale-provider-token", "websocket:chat-1"),
    ).resolves.toMatchObject({
      messages: [{ role: "assistant", content: "still here" }],
    });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:18790/api/kb/stats",
      "/webui/bootstrap",
      "http://127.0.0.1:19999/api/kb/stats",
      "/api/sessions/websocket%3Achat-1/messages",
    ]);
    expect(
      new Headers(fetchMock.mock.calls[3][1]?.headers).get("Authorization"),
    ).toBe("Bearer fresh-token");
  });
});
