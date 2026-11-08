/**
 * ReproducibilityPanel - displays training run reproducibility metadata.
 * Shows dataset fingerprints, configuration, randomness, and runtime versions.
 * @module
 */

import { useState, useEffect } from "react";
import { ChevronDown, ChevronUp, AlertCircle, Package } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import CopyableText from "@/components/shared/CopyableText";
import { getTrainingJobDetails } from "@/services/ModelServices";
import type { TrainingJobDetails, ReproducibilitySnapshot } from "@/types/reproducibility";
import logger from "@/shared/logger";

interface ReproducibilityPanelProps {
  jobId: string;
  jobStatus?: string;
}

export default function ReproducibilityPanel({ jobId, jobStatus }: ReproducibilityPanelProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobDetails, setJobDetails] = useState<TrainingJobDetails | null>(null);
  const [lastJobId, setLastJobId] = useState<string | null>(null);

  useEffect(() => {
    // Fetch when expanded or when job ID changes
    if (jobId && (open || jobId !== lastJobId)) {
      fetchJobDetails();
      setLastJobId(jobId);
    }
  }, [jobId, open]);

  const fetchJobDetails = async () => {
    if (!jobId) return;

    try {
      setLoading(true);
      setError(null);
      const data = await getTrainingJobDetails(jobId);
      setJobDetails(data);
    } catch (err: any) {
      logger.error("Failed to fetch job details:", err);
      setError(err.message || "Failed to load training run details");
    } finally {
      setLoading(false);
    }
  };

  const metadata = jobDetails?.reproducibility_metadata;
  const isActiveJob = jobStatus === "pending" || jobStatus === "running";
  const isLegacyJob = !isActiveJob && !metadata;

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const formatPercentage = (value: number | null): string => {
    if (value === null) return "Not available";
    return `${value.toFixed(1)}%`;
  };

  return (
    <Card>
      <CardHeader>
        <button
          onClick={() => setOpen(!open)}
          className="flex w-full items-center justify-between text-left hover:opacity-80 transition-opacity"
          aria-expanded={open}
        >
          <CardTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            Run Details
          </CardTitle>
          {open ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
        </button>
      </CardHeader>

      {open && (
        <CardContent className="space-y-6">
          {loading && (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              Loading run details...
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="flex items-center justify-between">
                <span>{error}</span>
                <Button variant="outline" size="sm" onClick={fetchJobDetails}>
                  Retry
                </Button>
              </AlertDescription>
            </Alert>
          )}

          {!loading && !error && isLegacyJob && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Reproducibility metadata is unavailable for this training run. This feature was
                added after this job was created.
              </AlertDescription>
            </Alert>
          )}

          {!loading && !error && isActiveJob && !metadata && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Run details will be available once training configuration is captured.
              </AlertDescription>
            </Alert>
          )}

          {!loading && !error && metadata && (
            <div className="space-y-6">
              {/* Dataset Section */}
              <Section title="Dataset">
                <KeyValue label="File" value={metadata.dataset.filename || "Unknown"} />
                <KeyValue label="Type" value={metadata.dataset.dataset_type.toUpperCase()} />
                <KeyValue
                  label="Fingerprint Algorithm"
                  value={metadata.dataset.fingerprint_algorithm.toUpperCase()}
                />
                <KeyValue
                  label="Fingerprint Scope"
                  value={
                    metadata.dataset.fingerprint_scope === "file_bytes"
                      ? "File Bytes"
                      : "Image Tree Manifest"
                  }
                />
                <div className="space-y-1">
                  <dt className="text-sm font-medium text-muted-foreground">SHA-256 Fingerprint</dt>
                  <dd>
                    <CopyableText
                      value={metadata.dataset.fingerprint}
                      label="fingerprint"
                      truncate={false}
                    />
                  </dd>
                </div>
                <KeyValue
                  label="Dataset Size"
                  value={formatBytes(metadata.dataset.size_bytes)}
                />
              </Section>

              <Separator />

              {/* Training Configuration Section */}
              <Section title="Training Configuration">
                <KeyValue
                  label="Problem Type"
                  value={
                    metadata.training_configuration.problem_type
                      ?.replace(/_/g, " ")
                      .replace(/\b\w/g, (l) => l.toUpperCase()) || "Not available"
                  }
                />
                <KeyValue
                  label="Target Column"
                  value={metadata.training_configuration.target_column || "Not available"}
                />
                {metadata.training_configuration.feature_columns &&
                  metadata.training_configuration.feature_columns.length > 0 && (
                    <KeyValue
                      label="Feature Columns"
                      value={`${metadata.training_configuration.feature_columns.length} columns`}
                      detail={metadata.training_configuration.feature_columns.join(", ")}
                    />
                  )}
                {metadata.training_configuration.class_names &&
                  metadata.training_configuration.class_names.length > 0 && (
                    <KeyValue
                      label="Classes"
                      value={`${metadata.training_configuration.class_names.length} classes`}
                      detail={metadata.training_configuration.class_names.join(", ")}
                    />
                  )}
                <KeyValue
                  label="Epochs"
                  value={metadata.training_configuration.epochs?.toString() || "Not available"}
                />
                <KeyValue
                  label="Batch Size"
                  value={metadata.training_configuration.batch_size?.toString() || "Not available"}
                />
                <KeyValue
                  label="Optimizer"
                  value={
                    metadata.training_configuration.optimizer?.class_name ||
                    metadata.training_configuration.optimizer?.config?.name ||
                    "Not available"
                  }
                />
                <KeyValue
                  label="Loss Function"
                  value={
                    metadata.training_configuration.loss?.class_name ||
                    metadata.training_configuration.loss?.config?.name ||
                    "Not available"
                  }
                />
                {metadata.training_configuration.metrics.length > 0 && (
                  <KeyValue
                    label="Metrics"
                    value={metadata.training_configuration.metrics.join(", ")}
                  />
                )}
              </Section>

              <Separator />

              {/* Data Split Section */}
              <Section title="Data Split">
                <KeyValue
                  label="Requested Training Split"
                  value={formatPercentage(
                    metadata.training_configuration.requested_training_split_percent,
                  )}
                />
                <KeyValue
                  label="Requested Holdout Split"
                  value={formatPercentage(
                    metadata.training_configuration.requested_holdout_split_percent,
                  )}
                />
                {metadata.training_configuration.actual_training_rows !== null && (
                  <KeyValue
                    label="Actual Training Data"
                    value={`${metadata.training_configuration.actual_training_rows} rows (${formatPercentage(
                      metadata.training_configuration.actual_training_split_percent,
                    )})`}
                  />
                )}
                {metadata.training_configuration.actual_holdout_rows !== null && (
                  <KeyValue
                    label="Actual Holdout Data"
                    value={`${metadata.training_configuration.actual_holdout_rows} rows (${formatPercentage(
                      metadata.training_configuration.actual_holdout_split_percent,
                    )})`}
                  />
                )}
                <KeyValue
                  label="Holdout Role"
                  value={
                    metadata.training_configuration.holdout_role
                      ?.replace(/\b\w/g, (l) => l.toUpperCase()) || "Not available"
                  }
                />
              </Section>

              <Separator />

              {/* Randomness Section */}
              <Section title="Randomness">
                <KeyValue
                  label="Configured Seed"
                  value={metadata.randomness.configured_seed?.toString() || "None"}
                />
                {Object.keys(metadata.randomness.effective_seeds).length > 0 && (
                  <div className="space-y-1">
                    <dt className="text-sm font-medium text-muted-foreground">Effective Seeds</dt>
                    <dd className="text-sm">
                      {Object.entries(metadata.randomness.effective_seeds).map(([key, value]) => (
                        <div key={key} className="ml-4">
                          <span className="font-mono text-xs text-muted-foreground">{key}:</span>{" "}
                          {value !== null ? value : "Not set"}
                        </div>
                      ))}
                    </dd>
                  </div>
                )}
                {metadata.randomness.applied_seeds.length > 0 && (
                  <KeyValue
                    label="Applied Seeds"
                    value={metadata.randomness.applied_seeds.join(", ")}
                  />
                )}
                <KeyValue
                  label="Global Framework Seed"
                  value={metadata.randomness.global_framework_seed_applied ? "Applied" : "Not applied"}
                />
                {!metadata.randomness.deterministic_execution_guaranteed && (
                  <Alert className="mt-2">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription className="text-xs">
                      Deterministic execution is not guaranteed. TensorFlow GPU operations may
                      produce non-deterministic results even with fixed seeds due to parallel
                      execution optimizations.
                    </AlertDescription>
                  </Alert>
                )}
              </Section>

              <Separator />

              {/* Runtime Versions Section */}
              <Section title="Runtime Versions">
                <KeyValue label="Python" value={metadata.runtime_versions.python} />
                <KeyValue
                  label="TensorFlow"
                  value={metadata.runtime_versions.tensorflow || "Not available"}
                />
                <KeyValue
                  label="Keras"
                  value={metadata.runtime_versions.keras || "Not available"}
                />
                <KeyValue
                  label="NumPy"
                  value={metadata.runtime_versions.numpy || "Not available"}
                />
                <KeyValue
                  label="Pandas"
                  value={metadata.runtime_versions.pandas || "Not available"}
                />
              </Section>

              <Separator />

              {/* Metadata Information */}
              <div className="text-xs text-muted-foreground space-y-1">
                <div>Captured: {formatTimestamp(metadata.captured_at)}</div>
                <div>Schema Version: {metadata.schema_version}</div>
                <div className="font-mono">Job ID: {metadata.job_id}</div>
              </div>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}

// Helper Components

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold">{title}</h3>
      <dl className="space-y-2">{children}</dl>
    </div>
  );
}

function KeyValue({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="space-y-1">
      <dt className="text-sm font-medium text-muted-foreground">{label}</dt>
      <dd className="text-sm">
        {value}
        {detail && <div className="text-xs text-muted-foreground mt-1">{detail}</div>}
      </dd>
    </div>
  );
}
