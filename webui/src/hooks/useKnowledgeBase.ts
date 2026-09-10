import { useCallback, useRef, useState } from "react";
import {
  deletePaper as requestPaperDeletion,
  fetchKBStats,
  uploadPapers,
} from "@/lib/kb-api";
import type { KBStats, UploadResult } from "@/lib/types";

export interface UploadState {
  results: UploadResult[]; // accumulated per-file results across uploads (incl. pending)
  error: string | null;
}

export function useKnowledgeBase() {
  const [uploadState, setUploadState] = useState<UploadState>({
    results: [],
    error: null,
  });
  const [stats, setStats] = useState<KBStats | null>(null);
  const [deletingPaperId, setDeletingPaperId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const resultsRef = useRef<UploadResult[]>([]);
  // Monotonically increasing batch counter; each doUpload call gets a unique
  // batchId so we can locate its pending entries even if another batch starts
  // before the current one finishes.
  const batchCounter = useRef(0);

  const refreshStats = useCallback(async () => {
    const s = await fetchKBStats();
    if (s) setStats(s);
  }, []);

  const doUpload = useCallback(async (files: File[]) => {
    if (files.length === 0) return;

    // Increment batch counter to track this batch's position
    batchCounter.current += 1;
    const batchStart = resultsRef.current.length;

    // Insert pending entries immediately
    const pendingEntries: UploadResult[] = files.map((f) => ({
      filename: f.name,
      status: "pending" as const,
    }));
    resultsRef.current = [...resultsRef.current, ...pendingEntries];
    setUploadState({ results: resultsRef.current, error: null });

    try {
      const newResults = await uploadPapers(files);
      // Replace pending entries for THIS batch only, by position
      const updated = [...resultsRef.current];
      for (let i = 0; i < newResults.length; i++) {
        const idx = batchStart + i;
        if (idx < updated.length && updated[idx].status === "pending") {
          updated[idx] = { ...newResults[i], filename: updated[idx].filename };
        }
      }
      resultsRef.current = updated;
      setUploadState({ results: resultsRef.current, error: null });
      await refreshStats();
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : "Unknown upload error";
      // Only mark THIS batch's pending entries as error
      const updated = [...resultsRef.current];
      for (let i = 0; i < files.length; i++) {
        const idx = batchStart + i;
        if (idx < updated.length && updated[idx].status === "pending") {
          updated[idx] = { ...updated[idx], status: "error" as const, error: message };
        }
      }
      resultsRef.current = updated;
      setUploadState({ results: resultsRef.current, error: message });
    }
  }, [refreshStats]);

  const clearResults = useCallback(() => {
    resultsRef.current = [];
    setUploadState({ results: [], error: null });
  }, []);

  const deletePaperById = useCallback(async (paperId: string) => {
    setDeletingPaperId(paperId);
    setDeleteError(null);
    try {
      await requestPaperDeletion(paperId);
      await refreshStats();
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : "Unknown delete error";
      setDeleteError(message);
      throw error;
    } finally {
      setDeletingPaperId(null);
    }
  }, [refreshStats]);

  const clearDeleteError = useCallback(() => {
    setDeleteError(null);
  }, []);

  return {
    uploadState,
    stats,
    deletingPaperId,
    deleteError,
    doUpload,
    deletePaperById,
    clearDeleteError,
    clearResults,
    refreshStats,
  } as const;
}
