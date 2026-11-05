/**
 * Probability Calibration Curve and Reliability Analysis Panel Component
 * Displays reliability diagrams, confidence distribution histograms,
 * and key calibration metrics (Brier Score, ECE, MCE).
 * @module
 */

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { ChevronDown, ChevronUp, RefreshCw, AlertCircle, Info } from "lucide-react";
import axios from "@/shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "@/constants/Urls";
import { CalibrationCurveData, ClassCalibrationData } from "@/types/analysis";
import logger from "@/shared/logger";

interface CalibrationCurvePanelProps {
  jobId: string;
}

export default function CalibrationCurvePanel({ jobId }: CalibrationCurvePanelProps) {
  const [data, setData] = useState<CalibrationCurveData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedClassIndex, setSelectedClassIndex] = useState(0);
  const [nBins, setNBins] = useState(10);
  const [showTable, setShowTable] = useState(false);

  const fetchCalibrationData = async (bins: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${BACKEND_MODEL_ANALYSIS}/${jobId}/calibration-curve`, {
        params: { n_bins: bins },
      });
      setData(response.data as CalibrationCurveData);
    } catch (err: any) {
      logger.error("Failed to fetch calibration curve:", err);
      setError(
        err.response?.data?.detail || err.message || "Failed to load calibration curve data",
      );
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchCalibrationData(nBins);
  }, [jobId, nBins]);

  const selectedClassData: ClassCalibrationData | undefined =
    data?.classes && data.classes.length > selectedClassIndex
      ? data.classes[selectedClassIndex]
      : data?.classes?.[0];

  const validScatterPoints = (selectedClassData?.points || [])
    .filter((pt) => pt.count > 0 && pt.fraction_positives !== null)
    .map((pt) => ({
      mean_predicted: pt.mean_predicted,
      fraction_positives: pt.fraction_positives,
      count: pt.count,
      error: pt.error,
      bin_label: pt.bin_label,
    }));

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <CardTitle className="text-base font-semibold">
                Probability Calibration & Reliability Curve
              </CardTitle>
              <span
                className="text-gray-400 hover:text-gray-600 cursor-help"
                title="Evaluates how well predicted probabilities match empirical outcomes. Perfect calibration aligns with the diagonal dashed line."
              >
                <Info className="h-4 w-4" />
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              One-vs-Rest reliability diagram and confidence distribution histogram
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Class selector for multi-class */}
            {data && data.class_names && data.class_names.length > 1 && (
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-gray-600">Class:</span>
                <Select
                  value={String(selectedClassIndex)}
                  onValueChange={(val) => setSelectedClassIndex(Number(val))}
                >
                  <SelectTrigger className="w-[140px] h-8 text-xs">
                    <SelectValue placeholder="Select class" />
                  </SelectTrigger>
                  <SelectContent>
                    {data.class_names.map((name, idx) => (
                      <SelectItem key={idx} value={String(idx)}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Bin count selector */}
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-medium text-gray-600">Bins:</span>
              <Select value={String(nBins)} onValueChange={(val) => setNBins(Number(val))}>
                <SelectTrigger className="w-[80px] h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="5">5</SelectItem>
                  <SelectItem value="10">10</SelectItem>
                  <SelectItem value="15">15</SelectItem>
                  <SelectItem value="20">20</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              variant="outline"
              size="sm"
              className="h-8 px-2"
              onClick={() => fetchCalibrationData(nBins)}
              disabled={isLoading}
              title="Refresh calibration data"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        </div>

        {/* Metric Badges */}
        {selectedClassData && !selectedClassData.insufficient_data && (
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t mt-3">
            <Badge
              variant="outline"
              className="bg-blue-50 text-blue-700 border-blue-200 text-xs py-0.5"
            >
              Brier Score:{" "}
              {selectedClassData.brier_score !== null
                ? selectedClassData.brier_score.toFixed(4)
                : "N/A"}
            </Badge>
            <Badge
              variant="outline"
              className="bg-purple-50 text-purple-700 border-purple-200 text-xs py-0.5"
            >
              ECE:{" "}
              {selectedClassData.ece !== null
                ? `${(selectedClassData.ece * 100).toFixed(2)}%`
                : "N/A"}
            </Badge>
            <Badge
              variant="outline"
              className="bg-indigo-50 text-indigo-700 border-indigo-200 text-xs py-0.5"
            >
              MCE:{" "}
              {selectedClassData.mce !== null
                ? `${(selectedClassData.mce * 100).toFixed(2)}%`
                : "N/A"}
            </Badge>

            {data && data.macro_ece !== null && data.classes.length > 1 && (
              <Badge
                variant="outline"
                className="bg-gray-50 text-gray-700 border-gray-200 text-xs py-0.5"
              >
                Macro ECE: {(data.macro_ece * 100).toFixed(2)}%
              </Badge>
            )}
            {data && data.macro_brier_score !== null && data.classes.length > 1 && (
              <Badge
                variant="outline"
                className="bg-gray-50 text-gray-700 border-gray-200 text-xs py-0.5"
              >
                Macro Brier: {data.macro_brier_score.toFixed(4)}
              </Badge>
            )}
          </div>
        )}
      </CardHeader>

      <CardContent>
        {isLoading ? (
          <div className="flex justify-center items-center py-16">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <p className="text-xs text-gray-500">Computing probability calibration...</p>
            </div>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-50 border border-red-200 rounded-md flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-red-800 font-medium">Failed to load calibration data</p>
              <p className="text-xs text-red-600 mt-1">{error}</p>
            </div>
          </div>
        ) : selectedClassData?.insufficient_data ? (
          <div className="p-6 text-center bg-gray-50 rounded-md border border-dashed border-gray-200">
            <p className="text-sm font-medium text-gray-700">Insufficient Class Samples</p>
            <p className="text-xs text-gray-500 mt-1">
              The test dataset does not contain sufficient positive and negative examples to compute
              probability calibration for class &ldquo;{selectedClassData.class_name}&rdquo;.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <Tabs defaultValue="diagram" className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-4">
                <TabsTrigger value="diagram">Reliability Diagram</TabsTrigger>
                <TabsTrigger value="histogram">Confidence Histogram</TabsTrigger>
              </TabsList>

              {/* Reliability Diagram Tab */}
              <TabsContent value="diagram" className="space-y-2">
                <div className="h-[320px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 25, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        type="number"
                        dataKey="mean_predicted"
                        name="Mean Predicted"
                        domain={[0, 1]}
                        ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                        label={{
                          value: "Mean Predicted Probability",
                          position: "insideBottom",
                          offset: -12,
                          style: { fontSize: 12, fill: "#4b5563" },
                        }}
                      />
                      <YAxis
                        type="number"
                        dataKey="fraction_positives"
                        name="Fraction Positives"
                        domain={[0, 1]}
                        ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
                        label={{
                          value: "Fraction of Positives",
                          angle: -90,
                          position: "insideLeft",
                          style: { fontSize: 12, fill: "#4b5563" },
                        }}
                      />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const pt = payload[0].payload;
                            return (
                              <div className="bg-white p-2.5 border rounded-md shadow-md text-xs space-y-1">
                                <p className="font-semibold text-gray-800">Bin: {pt.bin_label}</p>
                                <p className="text-gray-600">
                                  Mean Predicted:{" "}
                                  <span className="font-medium text-gray-900">
                                    {pt.mean_predicted?.toFixed(4)}
                                  </span>
                                </p>
                                <p className="text-gray-600">
                                  Empirical Fraction:{" "}
                                  <span className="font-medium text-gray-900">
                                    {pt.fraction_positives?.toFixed(4)}
                                  </span>
                                </p>
                                <p className="text-gray-600">
                                  Sample Count:{" "}
                                  <span className="font-medium text-gray-900">{pt.count}</span>
                                </p>
                                {pt.error !== undefined && pt.error !== null && (
                                  <p className="text-gray-600">
                                    Calibration Error:{" "}
                                    <span className="font-medium text-red-600">
                                      {pt.error?.toFixed(4)}
                                    </span>
                                  </p>
                                )}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <ReferenceLine
                        segment={[
                          { x: 0, y: 0 },
                          { x: 1, y: 1 },
                        ]}
                        stroke="#9ca3af"
                        strokeWidth={2}
                        strokeDasharray="4 4"
                      />
                      <Scatter
                        name="Model Calibration"
                        data={validScatterPoints}
                        fill="#2563eb"
                        line={{ stroke: "#2563eb", strokeWidth: 2 }}
                        shape="circle"
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-xs text-gray-500 flex items-center justify-between px-2">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-gray-400 border-dashed inline-block"></span>
                      Perfect Calibration (y = x)
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block"></span>
                      Model Reliability Curve
                    </span>
                  </div>
                  <span>Points closer to the diagonal indicate higher calibration accuracy</span>
                </div>
              </TabsContent>

              {/* Confidence Histogram Tab */}
              <TabsContent value="histogram" className="space-y-2">
                <div className="h-[320px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={selectedClassData?.histogram || []}
                      margin={{ top: 20, right: 20, bottom: 25, left: 20 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="bin_label"
                        label={{
                          value: "Confidence Interval",
                          position: "insideBottom",
                          offset: -12,
                          style: { fontSize: 12, fill: "#4b5563" },
                        }}
                      />
                      <YAxis
                        label={{
                          value: "Sample Count",
                          angle: -90,
                          position: "insideLeft",
                          style: { fontSize: 12, fill: "#4b5563" },
                        }}
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const b = payload[0].payload;
                            return (
                              <div className="bg-white p-2.5 border rounded-md shadow-md text-xs space-y-1">
                                <p className="font-semibold text-gray-800">
                                  Interval: {b.bin_label}
                                </p>
                                <p className="text-gray-600">
                                  Count:{" "}
                                  <span className="font-medium text-gray-900">{b.count}</span>
                                </p>
                                <p className="text-gray-600">
                                  Frequency:{" "}
                                  <span className="font-medium text-gray-900">
                                    {(b.frequency * 100).toFixed(1)}%
                                  </span>
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar
                        dataKey="count"
                        fill="#3b82f6"
                        radius={[4, 4, 0, 0]}
                        name="Sample Count"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-xs text-gray-500 px-2">
                  Distribution of predicted probabilities across confidence intervals.
                </div>
              </TabsContent>
            </Tabs>

            {/* Collapsible Bin Table */}
            <div className="border rounded-md mt-4">
              <button
                type="button"
                className="w-full flex items-center justify-between p-3 text-xs font-medium text-gray-700 bg-gray-50 hover:bg-gray-100 transition-colors"
                onClick={() => setShowTable((prev) => !prev)}
              >
                <span>Bin Details & Calibration Error Breakdown</span>
                {showTable ? (
                  <ChevronUp className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                )}
              </button>

              {showTable && selectedClassData?.points && (
                <div className="overflow-x-auto p-2">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-gray-50 border-b text-gray-600">
                      <tr>
                        <th className="p-2">Bin Interval</th>
                        <th className="p-2">Samples</th>
                        <th className="p-2">Mean Confidence</th>
                        <th className="p-2">Empirical Accuracy</th>
                        <th className="p-2">Calibration Error</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y">
                      {selectedClassData.points.map((pt, idx) => (
                        <tr
                          key={idx}
                          className={pt.count === 0 ? "text-gray-400" : "text-gray-800"}
                        >
                          <td className="p-2 font-mono">{pt.bin_label}</td>
                          <td className="p-2 font-medium">{pt.count}</td>
                          <td className="p-2">
                            {pt.mean_predicted !== null ? pt.mean_predicted.toFixed(4) : "—"}
                          </td>
                          <td className="p-2">
                            {pt.fraction_positives !== null
                              ? pt.fraction_positives.toFixed(4)
                              : "—"}
                          </td>
                          <td className="p-2">
                            {pt.error !== null ? (
                              <span
                                className={
                                  pt.error > 0.1
                                    ? "text-red-600 font-medium"
                                    : "text-green-600 font-medium"
                                }
                              >
                                {pt.error.toFixed(4)}
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
