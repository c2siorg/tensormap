/**
 * ReproducibilityDetails Component
 * Displays deterministic dataset fingerprint, training configuration,
 * runtime environment, and random seed for reproducible training runs.
 * @module
 */

import { useState, useEffect } from "react";
import {
  Database,
  Cpu,
  Sliders,
  Calendar,
  Fingerprint,
  Check,
  Copy,
  Info,
  AlertCircle,
  FileCode,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import axios from "@/shared/Axios";
import { BACKEND_TRAINING_JOB } from "@/constants/Urls";
import logger from "@/shared/logger";
import { ReproducibilitySnapshot, ReproducibilityDetailsProps } from "@/types/training";

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || isNaN(bytes)) return "Unavailable";
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

export default function ReproducibilityDetails({ jobId }: ReproducibilityDetailsProps) {
  const [snapshot, setSnapshot] = useState<ReproducibilitySnapshot | null>(null);
  const [hasSnapshot, setHasSnapshot] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedFingerprint, setCopiedFingerprint] = useState<boolean>(false);
  const [copiedJson, setCopiedJson] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;

    async function fetchReproducibility() {
      try {
        setLoading(true);
        setError(null);
        const response = await axios.get(`${BACKEND_TRAINING_JOB}/${jobId}/reproducibility`);
        if (!isMounted) return;

        const data = response.data?.data;
        if (data && data.has_snapshot && data.reproducibility) {
          setSnapshot(data.reproducibility);
          setHasSnapshot(true);
        } else {
          setSnapshot(null);
          setHasSnapshot(false);
        }
      } catch (err: any) {
        if (!isMounted) return;
        logger.error("Failed to load reproducibility snapshot:", err);
        setError(err.response?.data?.message || err.message || "Failed to load reproducibility metadata");
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    if (jobId) {
      fetchReproducibility();
    }

    return () => {
      isMounted = false;
    };
  }, [jobId]);

  const copyToClipboard = async (text: string, type: "fingerprint" | "json") => {
    try {
      await navigator.clipboard.writeText(text);
      if (type === "fingerprint") {
        setCopiedFingerprint(true);
        setTimeout(() => setCopiedFingerprint(false), 2000);
      } else {
        setCopiedJson(true);
        setTimeout(() => setCopiedJson(false), 2000);
      }
    } catch {
      // Fallback or ignore
    }
  };

  if (loading) {
    return (
      <Card className="mt-4">
        <CardContent className="py-6 text-center text-sm text-muted-foreground">
          Loading reproducibility metadata...
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="mt-4">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!hasSnapshot || !snapshot) {
    return (
      <Card className="mt-4 border-dashed">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <Info className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base font-medium">Reproducibility & Run Context</CardTitle>
          </div>
          <CardDescription>
            No reproducibility snapshot recorded for this training run.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground">
            Legacy jobs executed prior to snapshot tracking do not include content fingerprints or runtime environment archives.
          </p>
        </CardContent>
      </Card>
    );
  }

  const { dataset, training_config, runtime_environment, random_seed, captured_at } = snapshot;

  return (
    <Card className="mt-4 border" data-testid="reproducibility-details-card">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <Fingerprint className="h-5 w-5 text-primary" />
              <CardTitle className="text-base font-semibold">Reproducibility & Execution Context</CardTitle>
              <Badge variant="outline" className="text-xs font-mono">
                Job {jobId.slice(0, 8)}
              </Badge>
            </div>
            <CardDescription className="flex items-center gap-1.5 mt-1 text-xs">
              <Calendar className="h-3.5 w-3.5" />
              Captured: {captured_at ? new Date(captured_at).toLocaleString() : "Unavailable"}
            </CardDescription>
          </div>

          <Button
            variant="outline"
            size="sm"
            className="h-8 gap-1.5 text-xs"
            onClick={() => copyToClipboard(JSON.stringify(snapshot, null, 2), "json")}
          >
            {copiedJson ? (
              <>
                <Check className="h-3.5 w-3.5 text-green-600" /> Copied JSON
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" /> Export Snapshot JSON
              </>
            )}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 pt-1">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* 1. Dataset Snapshot */}
          <div className="rounded-lg border bg-card p-3 shadow-sm space-y-2">
            <div className="flex items-center gap-1.5 font-medium text-xs text-muted-foreground uppercase tracking-wider">
              <Database className="h-3.5 w-3.5" />
              Dataset Provenance
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">File:</span>
                <span className="font-medium truncate max-w-[150px]" title={dataset?.file_name || ""}>
                  {dataset?.file_name || "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Format:</span>
                <span className="font-medium uppercase">{dataset?.file_type || "Unavailable"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Size:</span>
                <span className="font-medium">{formatBytes(dataset?.file_size_bytes)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Rows:</span>
                <span className="font-medium">
                  {dataset?.row_count != null ? dataset.row_count.toLocaleString() : "Unavailable"}
                </span>
              </div>

              <div className="pt-1">
                <span className="text-muted-foreground block mb-1">Content Fingerprint (SHA-256):</span>
                {dataset?.fingerprint ? (
                  <div className="flex items-center justify-between gap-1 bg-muted px-2 py-1 rounded font-mono text-[11px]">
                    <span className="truncate" title={dataset.fingerprint}>
                      {dataset.fingerprint.slice(0, 20)}...
                    </span>
                    <button
                      type="button"
                      aria-label="Copy fingerprint"
                      className="text-muted-foreground hover:text-foreground shrink-0"
                      onClick={() => copyToClipboard(dataset.fingerprint!, "fingerprint")}
                    >
                      {copiedFingerprint ? (
                        <Check className="h-3 w-3 text-green-600" />
                      ) : (
                        <Copy className="h-3 w-3" />
                      )}
                    </button>
                  </div>
                ) : (
                  <span className="text-muted-foreground italic text-[11px]">Unavailable</span>
                )}
              </div>
            </div>
          </div>

          {/* 2. Training Configuration */}
          <div className="rounded-lg border bg-card p-3 shadow-sm space-y-2">
            <div className="flex items-center gap-1.5 font-medium text-xs text-muted-foreground uppercase tracking-wider">
              <Sliders className="h-3.5 w-3.5" />
              Training Configuration
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Problem Type:</span>
                <span className="font-medium capitalize">
                  {training_config?.problem_type || "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Target Field:</span>
                <span className="font-medium">
                  {training_config?.target_field || "Unconfigured (unsupervised/image)"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Optimizer:</span>
                <span className="font-medium capitalize">{training_config?.optimizer || "Unavailable"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Learning Rate:</span>
                <span className="font-medium">
                  {training_config?.learning_rate != null
                    ? training_config.learning_rate
                    : "Default (unconfigured)"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Loss / Metric:</span>
                <span className="font-medium truncate max-w-[140px]" title={`${training_config?.loss || ""} / ${training_config?.metric || ""}`}>
                  {training_config?.loss || "Unavailable"} / {training_config?.metric || "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Epochs / Batch:</span>
                <span className="font-medium">
                  {training_config?.epochs != null ? training_config.epochs : "Unavailable"} /{" "}
                  {training_config?.batch_size != null ? training_config.batch_size : "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Train Split:</span>
                <span className="font-medium">
                  {training_config?.training_split != null
                    ? `${training_config.training_split}%`
                    : "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Random Seed:</span>
                <span className="font-mono font-medium">
                  {random_seed != null ? random_seed : "Unconfigured"}
                </span>
              </div>
            </div>
          </div>

          {/* 3. Runtime Environment */}
          <div className="rounded-lg border bg-card p-3 shadow-sm space-y-2">
            <div className="flex items-center gap-1.5 font-medium text-xs text-muted-foreground uppercase tracking-wider">
              <Cpu className="h-3.5 w-3.5" />
              Runtime Environment
            </div>

            <div className="space-y-1.5 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Python:</span>
                <span className="font-mono font-medium">
                  {runtime_environment?.python_version || "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Platform:</span>
                <span
                  className="font-mono text-[11px] truncate max-w-[150px]"
                  title={runtime_environment?.platform || ""}
                >
                  {runtime_environment?.platform || "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">TensorFlow:</span>
                <span className="font-mono font-medium">
                  {runtime_environment?.libraries?.tensorflow || "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">NumPy:</span>
                <span className="font-mono font-medium">
                  {runtime_environment?.libraries?.numpy || "Unavailable"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Pandas:</span>
                <span className="font-mono font-medium">
                  {runtime_environment?.libraries?.pandas || "Unavailable"}
                </span>
              </div>
              <div className="pt-2 text-[11px] text-muted-foreground border-t flex items-center gap-1">
                <FileCode className="h-3 w-3 shrink-0" />
                <span>Audited environment; credentials excluded.</span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
