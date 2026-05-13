import { useCallback, useRef, useState } from "react";
import {
  CheckCircle,
  FileText,
  FileUp,
  Loader2,
  RefreshCcw,
  Trash2,
  XCircle,
} from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { KBStats, UploadResult } from "@/lib/types";

interface KnowledgeBaseProps {
  onUpload: (files: File[]) => void;
  uploadState: {
    results: UploadResult[];
    error: string | null;
  };
  stats: KBStats | null;
  onRefresh: () => void;
  onClearResults: () => void;
}

export function KnowledgeBase({
  onUpload,
  uploadState,
  stats,
  onRefresh,
  onClearResults,
}: KnowledgeBaseProps) {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | File[]) => {
      const pdfs = Array.from(files).filter(
        (f) => f.type === "application/pdf" || f.name.endsWith(".pdf"),
      );
      if (pdfs.length > 0) onUpload(pdfs);
    },
    [onUpload],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      {/* Title */}
      <h2 className="text-lg font-semibold tracking-tight">
        {t("kb.title", "Knowledge Base")}
      </h2>

      {/* Drop zone / Upload area */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors",
          dragOver
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/40",
        )}
      >
        <FileUp className="h-8 w-8 text-muted-foreground/60" />
        <div>
          <p className="text-sm font-medium">
            {t("kb.dropHint", "Drop PDF files here")}
          </p>
          <p className="text-xs text-muted-foreground">
            {t(
              "kb.dropSubHint",
              "or click to browse — PDF only, up to 50 MB per file",
            )}
          </p>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
      </div>

      {/* Upload results (including pending) */}
      {uploadState.results.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">
              {t("kb.uploadResults", "Upload Results")}
            </p>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={onClearResults}
              aria-label={t("kb.clearHistory", "Clear history")}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </div>
          <div className="space-y-1">
            {uploadState.results.map((r, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm"
              >
                {r.status === "pending" ? (
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
                ) : r.status === "ok" ? (
                  <CheckCircle className="h-4 w-4 shrink-0 text-green-500" />
                ) : (
                  <XCircle className="h-4 w-4 shrink-0 text-red-500" />
                )}
                <span className="flex-1 truncate" title={r.filename}>
                  {r.filename}
                </span>
                {r.status === "pending" ? (
                  <span className="shrink-0 text-xs text-muted-foreground animate-pulse">
                    uploading…
                  </span>
                ) : r.status === "ok" ? (
                  <span className="shrink-0 text-xs text-muted-foreground">
                    {r.chunk_count} chunks
                  </span>
                ) : (
                  <span className="shrink-0 text-xs text-red-500">
                    {r.error ?? "failed"}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* KB Stats */}
      {stats && (
        <div className="mt-auto rounded-lg border p-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">
              {t("kb.stats", "Knowledge Base Stats")}
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={onRefresh}
              aria-label={t("kb.refresh", "Refresh stats")}
            >
              <RefreshCcw className="h-3.5 w-3.5" />
            </Button>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-md bg-muted p-2">
              <span className="text-muted-foreground">
                {t("kb.papers", "Papers")}
              </span>
              <p className="text-lg font-semibold">{stats.paper_count}</p>
            </div>
            <div className="rounded-md bg-muted p-2">
              <span className="text-muted-foreground">
                {t("kb.chunks", "Chunks")}
              </span>
              <p className="text-lg font-semibold">{stats.chunk_count}</p>
            </div>
          </div>
          {stats.recent_papers.length > 0 && (
            <div className="mt-3 space-y-1">
              <p className="text-[11px] font-medium text-muted-foreground">
                {t("kb.recent", "Recent Papers")}
              </p>
              {stats.recent_papers.slice(0, 5).map((p) => (
                <div
                  key={p.paper_id}
                  className="flex items-center gap-1.5 truncate text-xs"
                >
                  <FileText className="h-3 w-3 shrink-0 text-muted-foreground/60" />
                  <span className="truncate">{p.title || p.paper_id}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
