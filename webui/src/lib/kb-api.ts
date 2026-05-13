import type { KBStats, UploadResult } from "./types";

/** KB API base URL — aiohttp server on the gateway health/KB port. */
const KB_BASE = "http://127.0.0.1:18790";

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
  const res = await fetch(`${KB_BASE}/api/papers/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const message = body?.error?.message ?? `Upload failed (HTTP ${res.status})`;
    throw new Error(message);
  }
  const data = await res.json();
  return data.results ?? [];
}

/**
 * Fetch knowledge base statistics.
 *
 * @returns KBStats or null if the endpoint is unavailable
 */
export async function fetchKBStats(): Promise<KBStats | null> {
  try {
    const res = await fetch(`${KB_BASE}/api/kb/stats`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}
