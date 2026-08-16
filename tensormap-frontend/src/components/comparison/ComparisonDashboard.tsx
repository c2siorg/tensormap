/**
 * Comparison Dashboard Component
 * Compares training metrics across multiple training runs.
 * @module
 */

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Download, AlertCircle, Loader2 } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { compareJobs, getTrainingJobs, ComparisonJob } from "@/services/comparisonService";

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"];

interface ComparisonDashboardProps {
  modelName: string;
}

export default function ComparisonDashboard({ modelName }: ComparisonDashboardProps) {
  const [availableJobs, setAvailableJobs] = useState<any[]>([]);
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<ComparisonJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeMetric, setActiveMetric] = useState<"loss" | "accuracy">("loss");

  // Fetch available jobs
  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const response = await getTrainingJobs(modelName);
        setAvailableJobs(response.data.data);
      } catch (err: any) {
        setError(err.response?.data?.message || "Failed to fetch training jobs");
      }
    };

    if (modelName) {
      fetchJobs();
    }
  }, [modelName]);

  // Fetch comparison data when selection changes
  useEffect(() => {
    const fetchComparison = async () => {
      if (selectedJobIds.length === 0) {
        setComparisonData([]);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const response = await compareJobs(selectedJobIds, activeMetric === "loss" ? "val_loss" : "val_accuracy");
        setComparisonData(response.data.data.jobs);
      } catch (err: any) {
        setError(err.response?.data?.message || "Failed to fetch comparison data");
      } finally {
        setLoading(false);
      }
    };

    fetchComparison();
  }, [selectedJobIds, activeMetric]);

  const toggleJobSelection = (jobId: string) => {
    setSelectedJobIds((prev) => {
      if (prev.includes(jobId)) {
        return prev.filter((id) => id !== jobId);
      }
      if (prev.length >= 5) {
        setError("Maximum 5 jobs can be compared at once");
        return prev;
      }
      return [...prev, jobId];
    });
  };

  // Prepare chart data
  const chartData = comparisonData.length > 0
    ? (() => {
        const allEpochs = new Set<number>();
        comparisonData.forEach((job) => {
          job.metrics.forEach((m) => allEpochs.add(m.epoch));
        });

        return Array.from(allEpochs)
          .sort((a, b) => a - b)
          .map((epoch) => {
            const point: any = { epoch };
            comparisonData.forEach((job, idx) => {
              const metric = job.metrics.find((m) => m.epoch === epoch);
              if (metric) {
                if (activeMetric === "loss") {
                  point[`job${idx}`] = metric.val_loss || metric.loss;
                } else {
                  point[`job${idx}`] = metric.val_accuracy || metric.accuracy;
                }
              }
            });
            return point;
          });
      })()
    : [];

  // Export to CSV
  const exportCSV = () => {
    if (comparisonData.length === 0) return;

    const headers = ["Job ID", "Optimizer", "Learning Rate", "Batch Size", "Epochs", "Final Loss", "Final Accuracy"];
    const rows = comparisonData.map((job) => {
      const finalMetrics = job.metrics[job.metrics.length - 1] || {};
      return [
        job.job_id,
        job.hyperparams.optimizer || "",
        job.hyperparams.lr || "",
        job.hyperparams.batch_size || "",
        job.hyperparams.epochs || "",
        finalMetrics.val_loss || finalMetrics.loss || "",
        finalMetrics.val_accuracy || finalMetrics.accuracy || "",
      ];
    });

    const csv = [headers, ...rows].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `comparison-${modelName}-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Find best job
  const bestJobId = comparisonData.length > 0
    ? (() => {
        const jobsWithFinalMetric = comparisonData
          .map((job) => ({
            job_id: job.job_id,
            finalMetric: job.metrics[job.metrics.length - 1]?.[activeMetric === "loss" ? "val_loss" : "val_accuracy"],
          }))
          .filter((j) => j.finalMetric !== undefined);

        if (jobsWithFinalMetric.length === 0) return null;

        return activeMetric === "loss"
          ? jobsWithFinalMetric.reduce((best, curr) =>
              (curr.finalMetric! < best.finalMetric! ? curr : best)
            ).job_id
          : jobsWithFinalMetric.reduce((best, curr) =>
              (curr.finalMetric! > best.finalMetric! ? curr : best)
            ).job_id;
      })()
    : null;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Training Run Comparison</CardTitle>
          <CardDescription>
            Compare metrics across multiple training runs for{" "}
            <strong>{modelName}</strong>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Run Selector */}
          <div>
            <h3 className="font-semibold mb-3">Select Runs to Compare (max 5)</h3>
            <div className="space-y-2">
              {availableJobs.length === 0 ? (
                <p className="text-sm text-gray-500">No training runs available</p>
              ) : (
                availableJobs.slice(0, 10).map((job, idx) => {
                  const colorIdx = selectedJobIds.indexOf(job.job_id);
                  const color = colorIdx >= 0 ? COLORS[colorIdx] : "#gray";

                  return (
                    <label
                      key={job.job_id}
                      className={`flex items-center gap-3 p-3 border rounded cursor-pointer hover:bg-gray-50 ${
                        selectedJobIds.includes(job.job_id) ? "border-blue-500 bg-blue-50" : ""
                      }`}
                    >
                      <Checkbox
                        checked={selectedJobIds.includes(job.job_id)}
                        onCheckedChange={() => toggleJobSelection(job.job_id)}
                      />
                      {selectedJobIds.includes(job.job_id) && (
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: color }}
                        />
                      )}
                      <div className="flex-1">
                        <span className="font-medium">Run {idx + 1}</span>
                        <span className="text-sm text-gray-600 ml-2">
                          {job.job_id.slice(0, 8)}
                        </span>
                        <Badge className="ml-2" variant="outline">
                          {job.status}
                        </Badge>
                      </div>
                      {job.started_at && (
                        <span className="text-xs text-gray-500">
                          {new Date(job.started_at).toLocaleDateString()}
                        </span>
                      )}
                    </label>
                  );
                })
              )}
            </div>
          </div>

          {/* Comparison Content */}
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-gray-400" />
              <span className="ml-2 text-gray-600">Loading comparison data...</span>
            </div>
          )}

          {!loading && comparisonData.length > 0 && (
            <>
              {/* Metric Tabs */}
              <Tabs value={activeMetric} onValueChange={(v) => setActiveMetric(v as "loss" | "accuracy")}>
                <TabsList>
                  <TabsTrigger value="loss">Loss</TabsTrigger>
                  <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
                </TabsList>

                <TabsContent value={activeMetric} className="space-y-6">
                  {/* Comparison Chart */}
                  <Card>
                    <CardHeader>
                      <CardTitle>{activeMetric === "loss" ? "Loss" : "Accuracy"} Comparison</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={chartData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" label={{ value: "Epoch", position: "insideBottom", offset: -5 }} />
                          <YAxis label={{ value: activeMetric === "loss" ? "Loss" : "Accuracy", angle: -90, position: "insideLeft" }} />
                          <Tooltip />
                          <Legend />
                          {comparisonData.map((job, idx) => (
                            <Line
                              key={job.job_id}
                              type="monotone"
                              dataKey={`job${idx}`}
                              stroke={COLORS[idx]}
                              strokeWidth={2}
                              name={`Run ${idx + 1}`}
                              dot={false}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Results Table */}
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle>Comparison Summary</CardTitle>
                        <Button onClick={exportCSV} variant="outline" size="sm">
                          <Download className="w-4 h-4 mr-2" />
                          Export CSV
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Run</TableHead>
                            <TableHead>Optimizer</TableHead>
                            <TableHead>Learning Rate</TableHead>
                            <TableHead>Batch Size</TableHead>
                            <TableHead>Epochs</TableHead>
                            <TableHead className="text-right">Final Loss</TableHead>
                            <TableHead className="text-right">Final Accuracy</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {comparisonData.map((job, idx) => {
                            const finalMetrics = job.metrics[job.metrics.length - 1] || {};
                            const isBest = job.job_id === bestJobId;

                            return (
                              <TableRow key={job.job_id} className={isBest ? "bg-green-50 font-medium" : ""}>
                                <TableCell>
                                  <div className="flex items-center gap-2">
                                    <div
                                      className="w-3 h-3 rounded-full"
                                      style={{ backgroundColor: COLORS[idx] }}
                                    />
                                    Run {idx + 1}
                                    {isBest && (
                                      <Badge className="bg-green-600">BEST</Badge>
                                    )}
                                  </div>
                                </TableCell>
                                <TableCell className="capitalize">
                                  {job.hyperparams.optimizer || "—"}
                                </TableCell>
                                <TableCell>
                                  {job.hyperparams.lr ? job.hyperparams.lr.toExponential(2) : "—"}
                                </TableCell>
                                <TableCell>{job.hyperparams.batch_size || "—"}</TableCell>
                                <TableCell>{job.hyperparams.epochs || "—"}</TableCell>
                                <TableCell className="text-right">
                                  {finalMetrics.val_loss !== undefined
                                    ? finalMetrics.val_loss.toFixed(4)
                                    : finalMetrics.loss !== undefined
                                    ? finalMetrics.loss.toFixed(4)
                                    : "—"}
                                </TableCell>
                                <TableCell className="text-right">
                                  {finalMetrics.val_accuracy !== undefined
                                    ? (finalMetrics.val_accuracy * 100).toFixed(2) + "%"
                                    : finalMetrics.accuracy !== undefined
                                    ? (finalMetrics.accuracy * 100).toFixed(2) + "%"
                                    : "—"}
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </>
          )}

          {!loading && selectedJobIds.length === 0 && (
            <div className="p-8 text-center text-gray-500">
              <p>Select training runs above to compare their metrics</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
