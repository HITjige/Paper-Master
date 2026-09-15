import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { useNanobotStream } from "@/hooks/useNanobotStream";
import type { InboundEvent } from "@/lib/types";
import { ClientProvider } from "@/providers/ClientProvider";

function fakeClient() {
  const handlers = new Map<string, Set<(ev: InboundEvent) => void>>();
  return {
    client: {
      status: "open" as const,
      defaultChatId: null as string | null,
      onStatus: () => () => {},
      onChat(chatId: string, h: (ev: InboundEvent) => void) {
        let set = handlers.get(chatId);
        if (!set) {
          set = new Set();
          handlers.set(chatId, set);
        }
        set.add(h);
        return () => set!.delete(h);
      },
      sendMessage: vi.fn(),
      newChat: vi.fn(),
      attach: vi.fn(),
      connect: vi.fn(),
      close: vi.fn(),
      updateUrl: vi.fn(),
    },
    emit(chatId: string, ev: InboundEvent) {
      const set = handlers.get(chatId);
      set?.forEach((h) => h(ev));
    },
  };
}

function wrap(client: ReturnType<typeof fakeClient>["client"]) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <ClientProvider
        client={client as unknown as import("@/lib/nanobot-client").NanobotClient}
        token="tok"
      >
        {children}
      </ClientProvider>
    );
  };
}

describe("useNanobotStream", () => {
  it("drops whitespace-only tool segments while keeping the run active", () => {
    const fake = fakeClient();
    const { result } = renderHook(() => useNanobotStream("chat-t", []), {
      wrapper: wrap(fake.client),
    });

    act(() => {
      fake.emit("chat-t", {
        event: "delta",
        chat_id: "chat-t",
        text: "\n\n",
        stream_id: "segment-0",
      });
    });
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.isStreaming).toBe(true);

    act(() => {
      fake.emit("chat-t", {
        event: "stream_end",
        chat_id: "chat-t",
        stream_id: "segment-0",
        resuming: true,
      });
    });
    expect(result.current.messages).toHaveLength(0);
    expect(result.current.isStreaming).toBe(true);

    act(() => {
      fake.emit("chat-t", {
        event: "delta",
        chat_id: "chat-t",
        text: "Final answer",
        stream_id: "segment-1",
      });
      fake.emit("chat-t", {
        event: "stream_end",
        chat_id: "chat-t",
        stream_id: "segment-1",
        resuming: false,
      });
    });
    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].content).toBe("Final answer");
    expect(result.current.messages[0].isStreaming).toBe(false);
    expect(result.current.isStreaming).toBe(false);
  });

  it("collapses consecutive tool_hint frames into one trace row", () => {
    const fake = fakeClient();
    const { result } = renderHook(() => useNanobotStream("chat-t", []), {
      wrapper: wrap(fake.client),
    });

    act(() => {
      fake.emit("chat-t", {
        event: "message",
        chat_id: "chat-t",
        text: 'weather("get")',
        kind: "tool_hint",
      });
      fake.emit("chat-t", {
        event: "message",
        chat_id: "chat-t",
        text: 'search "hk weather"',
        kind: "tool_hint",
      });
    });

    expect(result.current.messages).toHaveLength(1);
    expect(result.current.messages[0].kind).toBe("trace");
    expect(result.current.messages[0].role).toBe("tool");
    expect(result.current.messages[0].traces).toEqual([
      'weather("get")',
      'search "hk weather"',
    ]);

    act(() => {
      fake.emit("chat-t", {
        event: "message",
        chat_id: "chat-t",
        text: "## Summary",
      });
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].role).toBe("assistant");
    expect(result.current.messages[1].kind).toBeUndefined();
  });
});
