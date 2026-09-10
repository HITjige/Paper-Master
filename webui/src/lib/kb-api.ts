import type { DeletePaperResult, KBStats, UploadResult } from "./types";
import { fetchBootstrap } from "./bootstrap";

/** Resolve the KB API discovered during bootstrap, with a dev-server fallback. */
function kbApiConfig(): { base: string; headers: Record<string, string> } {
  const globals = window as unknown as Record<string, unknown>;
  const base =
    typeof globals.__NANOBOT_API_URL__ === "string"
      ? globals.__NANOBOT_API_URL__.replace(/\/$/, "")
      : "http://127.0.0.1:18790";
  const token = globals.__NANOBOT_API_TOKEN__;
  return {
    base,
    headers:
      typeof token === "string" && token
        ? { Authorization: `Bearer ${token}` }
        : {},
  };
}

async function refreshKbApiConfig(): Promise<ReturnType<typeof kbApiConfig>> {
  const boot = await fetchBootstrap();
  const globals = window as unknown as Record<string, unknown>;
  if (boot.api_url) globals.__NANOBOT_API_URL__ = boot.api_url;
  globals.__NANOBOT_API_TOKEN__ = boot.token;
  return kbApiConfig();
}

async function kbFetch(path: string, init: RequestInit = {}): Promise<Response> {
  let api = kbApiConfig();
  const request = () => {
    const headers = new Headers(init.headers);
    for (const [name, value] of Object.entries(api.headers)) {
      headers.set(name, value);
    }
    return fetch(`${api.base}${path}`, { ...init, headers });
  };
  let response = await request();
  if (response.status === 401) {
    api = await refreshKbApiConfig();
    response = await request();
  }
  return response;
}

/**
 * Upload one or more PDF files to the knowledge base.
 *
 * @param files - Array of File objects (only .pdf files will be processed)
 * @returns Per-file ingestion results
 */
export async function uploadPapers(files: File[]): Promise<UploadResult[]> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  const res = await kbFetch("/api/papers/upload", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message = body?.error?.message ?? `Upload failed (HTTP ${res.status})`;
    throw new Error(message);
  }
  const data = await res.json();
  if (res.status === 202 && typeof data.job_id === "string") {
    return pollIngestJob(data.job_id);
  }
  return data.results ?? [];
}

async function pollIngestJob(jobId: string): Promise<UploadResult[]> {
  const deadline = Date.now() + 15 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 750));
    const response = await kbFetch(
      `/api/papers/jobs/${encodeURIComponent(jobId)}`,
    );
    if (!response.ok) {
      throw new Error(`Unable to read ingestion job (HTTP ${response.status})`);
    }
    const job = await response.json();
    if (job.status === "completed") return job.results ?? [];
    if (job.status === "cancelled") throw new Error("PDF ingestion was cancelled");
  }
  throw new Error("PDF ingestion is still running; refresh the knowledge base later");
}

/**
 * Fetch knowledge base statistics.
 *
 * @returns KBStats or null if the endpoint is unavailable
 */
export async function fetchKBStats(): Promise<KBStats | null> {
  try {
    const res = await kbFetch("/api/kb/stats");
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Delete one paper and all of its knowledge-base index entries. */
export async function deletePaper(paperId: string): Promise<DeletePaperResult> {
  const res = await kbFetch(`/api/papers/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message = body?.error?.message ?? `Delete failed (HTTP ${res.status})`;
    throw new Error(message);
  }
  return await res.json();
}
