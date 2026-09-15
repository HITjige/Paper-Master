import { fetchBootstrap } from "./bootstrap";
import type { BootstrapResponse } from "./types";

type TokenListener = (token: string) => void;

const tokenListeners = new Set<TokenListener>();
let refreshInFlight: Promise<BootstrapResponse> | null = null;

type RuntimeGlobals = typeof globalThis & {
  __NANOBOT_API_URL__?: unknown;
  __NANOBOT_API_TOKEN__?: unknown;
};

function runtimeGlobals(): RuntimeGlobals {
  return globalThis as RuntimeGlobals;
}

/** Publish one bootstrap response to every HTTP consumer in the WebUI. */
export function applyRuntimeAuth(boot: BootstrapResponse): void {
  const globals = runtimeGlobals();
  if (boot.api_url) globals.__NANOBOT_API_URL__ = boot.api_url;
  globals.__NANOBOT_API_TOKEN__ = boot.token;
  for (const listener of tokenListeners) listener(boot.token);
}

/** Return the latest runtime token, falling back to the initial React value. */
export function runtimeToken(fallback = ""): string {
  const token = runtimeGlobals().__NANOBOT_API_TOKEN__;
  return typeof token === "string" && token ? token : fallback;
}

/** Return the API origin advertised by the gateway. */
export function runtimeApiBase(
  fallback = "http://127.0.0.1:18790",
): string {
  const base = runtimeGlobals().__NANOBOT_API_URL__;
  return (typeof base === "string" && base ? base : fallback).replace(/\/$/, "");
}

/** Keep React context consumers synchronized when any API refreshes auth. */
export function subscribeRuntimeToken(listener: TokenListener): () => void {
  tokenListeners.add(listener);
  return () => tokenListeners.delete(listener);
}

/** Mint one fresh token, coalescing concurrent 401 recovery attempts. */
export async function refreshRuntimeAuth(): Promise<BootstrapResponse> {
  if (refreshInFlight) return refreshInFlight;

  const request = fetchBootstrap().then((boot) => {
    applyRuntimeAuth(boot);
    return boot;
  });
  refreshInFlight = request;
  try {
    return await request;
  } finally {
    if (refreshInFlight === request) refreshInFlight = null;
  }
}

/** Perform an authenticated request and retry once with a fresh token on 401. */
export async function authenticatedFetch(
  resolveUrl: string | (() => string),
  init: RequestInit = {},
  fallbackToken = "",
): Promise<Response> {
  let token = runtimeToken(fallbackToken);
  const request = () => {
    const headers = new Headers(init.headers);
    if (token) headers.set("Authorization", `Bearer ${token}`);
    const url = typeof resolveUrl === "function" ? resolveUrl() : resolveUrl;
    return fetch(url, {
      ...init,
      headers,
      credentials: init.credentials ?? "same-origin",
    });
  };

  let response = await request();
  if (response.status === 401) {
    const boot = await refreshRuntimeAuth();
    token = boot.token;
    response = await request();
  }
  return response;
}
