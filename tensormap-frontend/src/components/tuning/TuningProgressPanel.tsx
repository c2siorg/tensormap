/**
 * Tuning Progress Panel Component
 * Displays real-time progress of hyperparameter tuning session.
 * @module
 */

import { useEffect, useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  subscribeToTuning,
  unsubscribeFromTuning,
  TuningProgress,
  TuningTrialSummary,
} from "@/services/tuningService";
import { CheckCircle2, Loader2, XCircle } from "lucide-react";

interface TuningProgressPanelProps {
  tuningId: string;
  totalTrials: number;
  metric: string;
  direction: "maximize" | "minimize";
  onCancel: () => void;
  onComplete?: () => void;
}

export default function TuningProgressPanel({
  tuningId,
  totalTrials,
  metric,
  direction,
  onCancel,
  onComplete,
}: TuningProgressPanelProps) {
  const [status, setStatus] = useState<"pending" | "running" | "completed" | "cancelled">(
    "pending",
  );
  const [completedTrials, setCompletedTrials] = useState(0);
  const [trials, setTrials] = useState<TuningTrialSummary[]>([]);
  const [bestJobId, setBestJobId] = useState<string | null>(null);
  const [bestMetric, setBestMetric] = useState<number | null>(null);
  const [earlyStopped, setEarlyStopped] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);

  useEffect(() => {
    const cleanup = subscribeToTuning(tuningId, (data: TuningProgress) => {
      if (data.type === "tuning_catchup") {
        // Handle catch-up: restore current state
        setCompletedTrials(data.completed_trials || 0);
        setStatus(
          (data.status as "pending" | "running" | "completed" | "cancelled") || "pending",
        );
        setEarlyStopped(data.early_stopped || false);
        if (data.trials) {
          setTrials(data.trials);
        }
      } else if (data.type === "tuning_progress") {
        // New trial completed
        setStatus("running");
        setCompletedTrials(data.trial || 0);

        // Update best metric
        if (data.best_metric !== undefined && data.best_metric !== null) {
          setBestMetric(data.best_metric);
        }
        if (data.best_job_id) {
          setBestJobId(data.best_job_id);
        }

        // Add trial to results (assuming backend sends hyperparams and metric)
        if (data.hyperparams && data.trial) {
          setTrials((prev) => {
            // Check if trial already exists (avoid duplicates)
            const existingIndex = prev.findIndex((t) => t.job_id === `trial-${data.trial}`);
            const newTrial: TuningTrialSummary = {
              job_id: `trial-${data.trial}`,
              status: "completed",
              hyperparams: data.hyperparams!,
              metric_value: data.metric || null,
              started_at: null,
              completed_at: new Date().toISOString(),
            };

            if (existingIndex >= 0) {
              const updated = [...prev];
              updated[existingIndex] = newTrial;
              return updated;
            }
            return [...prev, newTrial];
          });
        }
      } else if (data.type === "tuning_complete") {
        // Tuning finished
        setStatus((data.status as "completed" | "cancelled") || "completed");
        setCompletedTrials(totalTrials);

        if (data.best_metric !== undefined && data.best_metric !== null) {
          setBestMetric(data.best_metric);
        }
        if (data.best_job_id) {
          setBestJobId(data.best_job_id);
        }

        if (onComplete) {
          onComplete();
        }
      }
    });

    return () => {
      cleanup();
      unsubscribeFromTuning(tuningId);
    };
  }, [tuningId, totalTrials, onComplete]);

  const progress = totalTrials > 0 ? (completedTrials / totalTrials) * 100 : 0;

  const handleCancel = () => {
    setCancelRequested(true);
    onCancel();
  };

  // Calculate ETA
  const eta = useMemo(() => {
    if (completedTrials === 0 || status !== "running") return 0;
    // Simple estimate: assume remaining trials take same time as completed trials
    const remainingTrials = totalTrials - completedTrials;
    // Assume ~2 minutes per trial (120 seconds) - adjust based on actual data if available
    return remainingTrials * 120;
  }, [completedTrials, totalTrials, status]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    return `${mins} min`;
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Tuning Progress</CardTitle>
            <CardDescription>
              Session: <span className="font-mono">{tuningId.slice(0, 8)}</span>
            </CardDescription>
          </div>
          {status === "running" && (
            <Button
              variant="destructive"
              size="sm"
              onClick={handleCancel}
              disabled={cancelRequested}
            >
              {cancelRequested ? "Cancelling..." : "Cancel"}
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Progress Bar */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">
              Trial {completedTrials} / {totalTrials}
            </span>
            <span className="text-sm text-gray-600">{progress.toFixed(0)}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Status Badges */}
        <div className="flex items-center gap-3">
          {status === "running" && (
            <>
              <Badge variant="secondary" className="flex items-center gap-1">
                <Loader2 className="w-3 h-3 animate-spin" />
                Running
              </Badge>
              {eta > 0 && (
                <span className="text-sm text-gray-600">
                  {totalTrials - completedTrials} trials left · ~{formatTime(eta)} remaining
                </span>
              )}
            </>
          )}
          {status === "completed" && (
            <Badge variant="default" className="flex items-center gap-1 bg-green-600">
              <CheckCircle2 className="w-3 h-3" />
              Completed {completedTrials} trials
            </Badge>
          )}
          {status === "cancelled" && (
            <Badge variant="destructive" className="flex items-center gap-1">
              <XCircle className="w-3 h-3" />
              Cancelled
            </Badge>
          )}
          {earlyStopped && (
            <Badge variant="outline" className="border-blue-500 text-blue-700">
              Early Stopped — target metric reached
            </Badge>
          )}
        </div>

        {/* Results Table */}
        {trials.length > 0 && (
          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">#</TableHead>
                  <TableHead>Optimizer</TableHead>
                  <TableHead>Learning Rate</TableHead>
                  <TableHead>Batch Size</TableHead>
                  <TableHead>Epochs</TableHead>
                  <TableHead className="text-right">{metric}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trials.map((trial, index) => {
                  const isBest = trial.job_id === bestJobId;
                  const params = trial.hyperparams || {};

                  return (
                    <TableRow
                      key={trial.job_id}
                      className={isBest ? "bg-green-50 font-medium" : ""}
                    >
                      <TableCell>{index + 1}</TableCell>
                      <TableCell className="capitalize">
                        {(params.optimizer as string) || "—"}
                      </TableCell>
                      <TableCell>
                        {params.learning_rate
                          ? (params.learning_rate as number).toExponential(2)
                          : "—"}
                      </TableCell>
                      <TableCell>{(params.batch_size as number) || "—"}</TableCell>
                      <TableCell>{(params.epochs as number) || "—"}</TableCell>
                      <TableCell className="text-right">
                        {trial.metric_value !== null && trial.metric_value !== undefined
                          ? trial.metric_value.toFixed(4)
                          : "—"}
                        {isBest && <Badge className="ml-2 bg-green-600 text-xs">BEST</Badge>}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}

        {/* No Data Message */}
        {trials.length === 0 && status === "running" && (
          <div className="p-8 text-center text-gray-500">
            <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2 text-gray-400" />
            <p>Waiting for first trial to complete...</p>
          </div>
        )}

        {/* Best Metric Summary */}
        {bestMetric !== null && status === "completed" && (
          <div className="p-4 bg-green-50 border border-green-200 rounded-md">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-green-600" />
              <span className="font-medium text-green-900">
                Best {metric}: {bestMetric.toFixed(4)}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
