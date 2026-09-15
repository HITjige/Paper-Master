import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { runtimeToken, subscribeRuntimeToken } from "@/lib/auth";
import type { NanobotClient } from "@/lib/nanobot-client";

interface ClientContextValue {
  client: NanobotClient;
  token: string;
  modelName: string | null;
}

const ClientContext = createContext<ClientContextValue | null>(null);

export function ClientProvider({
  client,
  token,
  modelName = null,
  children,
}: {
  client: NanobotClient;
  token: string;
  modelName?: string | null;
  children: ReactNode;
}) {
  const [activeToken, setActiveToken] = useState(() => runtimeToken(token));

  useEffect(() => {
    setActiveToken(runtimeToken(token));
    return subscribeRuntimeToken(setActiveToken);
  }, [token]);

  const value = useMemo(
    () => ({ client, token: activeToken, modelName }),
    [activeToken, client, modelName],
  );

  return (
    <ClientContext.Provider value={value}>
      {children}
    </ClientContext.Provider>
  );
}

export function useClient(): ClientContextValue {
  const ctx = useContext(ClientContext);
  if (!ctx) {
    throw new Error("useClient must be used within a ClientProvider");
  }
  return ctx;
}
