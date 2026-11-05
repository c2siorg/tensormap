import { useState, useEffect, useCallback } from "react";
import PropTypes from "prop-types";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Sliders,
  ChevronDown,
  ChevronUp,
  Info,
} from "lucide-react";
import { getDatasetDrift } from "../../services/FileServices";
import logger from "../../shared/logger";

const DatasetDriftPanel = ({ fileId, files = [] }) => {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("split"); // "split" | "compare"
  const [baselineFileId, setBaselineFileId] = useState("");
  const [trainingSplit, setTrainingSplit] = useState(0.8);
  const [selectedFeatureName, setSelectedFeatureName] = useState(null);
  const [isOpen, setIsOpen] = useState(true);

  // Other CSV files available for comparison
  const otherCsvFiles = files.filter(
    (f) => String(f.file_id) !== String(fileId) && f.file_type === "csv",
  );

  const fetchDrift = useCallback(async () => {
    if (!fileId) return;
    setLoading(true);
    setError(null);

    const options = {
      training_split: trainingSplit,
      seed: 42,
      alpha: 0.05,
    };

    if (mode === "compare" && baselineFileId) {
      options.baseline_file_id = baselineFileId;
    }

    try {
      const data = await getDatasetDrift(fileId, options);
      setReport(data);
      if (data && data.features && data.features.length > 0) {
        // Select first drifted feature by default, or first feature
        const firstDrifted = data.features.find((f) => f.drift_detected);
        setSelectedFeatureName(
          firstDrifted ? firstDrifted.feature_name : data.features[0].feature_name,
        );
      }
    } catch (err) {
      logger.error("Failed to load dataset drift:", err);
      const detail = err.response?.data?.message || err.message || "Failed to load drift profile";
      setError(detail);
      setReport(null);
    } finally {
      setLoading(false);
    }
  }, [fileId, mode, baselineFileId, trainingSplit]);

  useEffect(() => {
    fetchDrift();
  }, [fetchDrift]);

  const selectedFeature = report?.features?.find((f) => f.feature_name === selectedFeatureName);

  return (
    <Card className="border shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsOpen((prev) => !prev)}
              className="flex items-center gap-2 text-left hover:text-blue-600 transition-colors"
            >
              <CardTitle className="text-base font-semibold">
                Dataset Drift &amp; Covariate Shift Profiler
              </CardTitle>
              {isOpen ? (
                <ChevronUp className="h-4 w-4 text-gray-500" />
              ) : (
                <ChevronDown className="h-4 w-4 text-gray-500" />
              )}
            </button>
            <span
              className="text-gray-400 hover:text-gray-600 cursor-help"
              title="Detects divergence between training/testing splits or dataset versions using KS tests, PSI, and Chi-square statistics."
            >
              <Info className="h-4 w-4" />
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Mode Selector */}
            <div className="flex items-center rounded-lg border bg-gray-50 p-0.5 text-xs">
              <button
                type="button"
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  mode === "split"
                    ? "bg-white text-blue-700 shadow-sm"
                    : "text-gray-600 hover:text-gray-900"
                }`}
                onClick={() => {
                  setMode("split");
                  setBaselineFileId("");
                }}
              >
                Train/Test Split
              </button>
              <button
                type="button"
                className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                  mode === "compare"
                    ? "bg-white text-blue-700 shadow-sm"
                    : "text-gray-600 hover:text-gray-900"
                }`}
                onClick={() => setMode("compare")}
              >
                Compare Datasets
              </button>
            </div>

            {mode === "split" ? (
              <div className="flex items-center gap-1.5 text-xs text-gray-600">
                <Sliders className="h-3.5 w-3.5" />
                <span>Split:</span>
                <select
                  value={trainingSplit}
                  onChange={(e) => setTrainingSplit(parseFloat(e.target.value))}
                  className="px-2 py-1 border rounded text-xs bg-white"
                >
                  <option value={0.7}>70% / 30%</option>
                  <option value={0.8}>80% / 20%</option>
                  <option value={0.85}>85% / 15%</option>
                  <option value={0.9}>90% / 10%</option>
                </select>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-xs text-gray-600">
                <span>Baseline:</span>
                <select
                  value={baselineFileId}
                  onChange={(e) => setBaselineFileId(e.target.value)}
                  className="px-2 py-1 border rounded text-xs bg-white max-w-[150px] truncate"
                >
                  <option value="">Select baseline dataset...</option>
                  {otherCsvFiles.map((f) => (
                    <option key={f.file_id} value={String(f.file_id)}>
                      {f.file_name}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <Button
              variant="outline"
              size="sm"
              className="h-8 px-2"
              onClick={fetchDrift}
              disabled={loading || (mode === "compare" && !baselineFileId)}
              title="Re-run drift analysis"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </div>
        </div>

        {isOpen && report && (
          <div className="mt-3 pt-3 border-t flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              {report.dataset_drift_detected ? (
                <div className="flex items-center gap-1.5 text-amber-700 bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-md font-medium">
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                  <span>Dataset Drift Detected</span>
                </div>
              ) : (
                <div className="flex items-center gap-1.5 text-green-700 bg-green-50 border border-green-200 px-2.5 py-1 rounded-md font-medium">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <span>Distributions are Stable</span>
                </div>
              )}

              <Badge variant="outline" className="text-gray-700">
                {report.drifted_features_count} of {report.total_features_count} features drifted (
                {(report.drift_share * 100).toFixed(1)}%)
              </Badge>
            </div>

            <div className="text-gray-500 space-x-2">
              <span>
                Baseline: <strong>{report.baseline?.name}</strong> ({report.baseline?.rows} rows)
              </span>
              <span>·</span>
              <span>
                Target: <strong>{report.target?.name}</strong> ({report.target?.rows} rows)
              </span>
            </div>
          </div>
        )}
      </CardHeader>

      {isOpen && (
        <CardContent>
          {loading ? (
            <div className="flex justify-center items-center py-12">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                <p className="text-xs text-gray-500">Computing distribution drift...</p>
              </div>
            </div>
          ) : error ? (
            <div className="p-4 bg-red-50 border border-red-200 rounded-md text-xs text-red-700 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Could not profile dataset drift</p>
                <p className="mt-0.5">{error}</p>
              </div>
            </div>
          ) : mode === "compare" && !baselineFileId ? (
            <div className="p-6 text-center bg-gray-50 rounded-md border border-dashed text-gray-500 text-xs">
              <p className="font-medium text-gray-700 mb-1">Select a Baseline Dataset</p>
              <p>
                Choose another CSV file from the dropdown above to compare distributions against
                this dataset.
              </p>
            </div>
          ) : report && report.features ? (
            <div className="space-y-6">
              {/* Features summary table */}
              <div className="overflow-x-auto border rounded-md">
                <table className="w-full text-xs text-left">
                  <thead className="bg-gray-50 border-b text-gray-600">
                    <tr>
                      <th className="p-2.5">Feature Name</th>
                      <th className="p-2.5">Type</th>
                      <th className="p-2.5">Drift Status</th>
                      <th className="p-2.5">Statistical Metric</th>
                      <th className="p-2.5">PSI</th>
                      <th className="p-2.5 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {report.features.map((feat) => {
                      const isSelected = feat.feature_name === selectedFeatureName;
                      return (
                        <tr
                          key={feat.feature_name}
                          className={`hover:bg-blue-50/50 cursor-pointer transition-colors ${
                            isSelected ? "bg-blue-50 font-medium" : ""
                          }`}
                          onClick={() => setSelectedFeatureName(feat.feature_name)}
                        >
                          <td className="p-2.5 font-mono text-gray-900">{feat.feature_name}</td>
                          <td className="p-2.5 capitalize text-gray-600">{feat.data_type}</td>
                          <td className="p-2.5">
                            {feat.drift_severity === "severe" ? (
                              <Badge className="bg-red-100 text-red-800 border-red-200 text-[10px] px-1.5 py-0.5">
                                Severe Drift
                              </Badge>
                            ) : feat.drift_severity === "moderate" ? (
                              <Badge className="bg-amber-100 text-amber-800 border-amber-200 text-[10px] px-1.5 py-0.5">
                                Moderate Drift
                              </Badge>
                            ) : (
                              <Badge className="bg-green-100 text-green-800 border-green-200 text-[10px] px-1.5 py-0.5">
                                Stable
                              </Badge>
                            )}
                          </td>
                          <td className="p-2.5 text-gray-700">
                            {feat.data_type === "numeric" ? (
                              feat.ks_statistic !== null ? (
                                <span>
                                  KS D={feat.ks_statistic.toFixed(3)} (p={feat.ks_pvalue.toFixed(3)}
                                  )
                                </span>
                              ) : (
                                "—"
                              )
                            ) : feat.chi2_statistic !== null ? (
                              <span>
                                Chi²={feat.chi2_statistic.toFixed(2)} (p=
                                {feat.chi2_pvalue.toFixed(3)})
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="p-2.5">
                            {feat.psi !== null ? (
                              <span
                                className={
                                  feat.psi >= 0.25
                                    ? "text-red-600 font-bold"
                                    : feat.psi >= 0.1
                                      ? "text-amber-600 font-medium"
                                      : "text-gray-600"
                                }
                              >
                                {feat.psi.toFixed(3)}
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="p-2.5 text-right">
                            <button
                              type="button"
                              className={`text-xs px-2 py-0.5 rounded border ${
                                isSelected
                                  ? "bg-blue-600 text-white border-blue-600"
                                  : "text-blue-600 border-blue-200 hover:bg-blue-50"
                              }`}
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedFeatureName(feat.feature_name);
                              }}
                            >
                              {isSelected ? "Viewing" : "View"}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Selected Feature Distribution Visualizer */}
              {selectedFeature && (
                <div className="border rounded-md p-4 bg-gray-50/50 space-y-4">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b pb-2">
                    <div>
                      <h4 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
                        <span>Distribution Comparison:</span>
                        <code className="text-blue-700 bg-blue-50 px-1.5 py-0.5 rounded">
                          {selectedFeature.feature_name}
                        </code>
                      </h4>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Overlaid frequency distribution of baseline vs target dataset.
                      </p>
                    </div>

                    {selectedFeature.unseen_categories &&
                      selectedFeature.unseen_categories.length > 0 && (
                        <div className="text-xs bg-red-50 text-red-700 border border-red-200 px-2 py-1 rounded">
                          Unseen in baseline:{" "}
                          <strong>{selectedFeature.unseen_categories.join(", ")}</strong>
                        </div>
                      )}
                  </div>

                  {/* Histogram Chart */}
                  {selectedFeature.distribution && selectedFeature.distribution.length > 0 ? (
                    <div className="h-[280px] w-full bg-white p-2 border rounded">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={selectedFeature.distribution}
                          margin={{ top: 15, right: 20, bottom: 25, left: 10 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                          <XAxis
                            dataKey={
                              selectedFeature.data_type === "numeric" ? "bin_label" : "category"
                            }
                            label={{
                              value:
                                selectedFeature.data_type === "numeric"
                                  ? "Value Interval"
                                  : "Category",
                              position: "insideBottom",
                              offset: -12,
                              style: { fontSize: 11, fill: "#6b7280" },
                            }}
                          />
                          <YAxis
                            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                            label={{
                              value: "Relative Share",
                              angle: -90,
                              position: "insideLeft",
                              style: { fontSize: 11, fill: "#6b7280" },
                            }}
                          />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (active && payload && payload.length) {
                                const pt = payload[0].payload;
                                const label = pt.bin_label || pt.category;
                                return (
                                  <div className="bg-white p-2.5 border rounded shadow-md text-xs space-y-1">
                                    <p className="font-semibold text-gray-800">{label}</p>
                                    <p className="text-blue-600">
                                      Baseline: {(pt.baseline_pct * 100).toFixed(1)}% (
                                      {pt.baseline_count} samples)
                                    </p>
                                    <p className="text-amber-600">
                                      Target: {(pt.target_pct * 100).toFixed(1)}% ({pt.target_count}{" "}
                                      samples)
                                    </p>
                                  </div>
                                );
                              }
                              return null;
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar
                            dataKey="baseline_pct"
                            name={report.baseline?.name || "Baseline"}
                            fill="#3b82f6"
                            radius={[3, 3, 0, 0]}
                          />
                          <Bar
                            dataKey="target_pct"
                            name={report.target?.name || "Target"}
                            fill="#f59e0b"
                            radius={[3, 3, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="text-center py-6 text-xs text-gray-500">
                      No distribution data available for this feature.
                    </div>
                  )}

                  {/* Per-feature Stats Comparison */}
                  {selectedFeature.stats && (
                    <div className="grid grid-cols-2 gap-4 text-xs pt-1">
                      <div className="bg-white p-2.5 rounded border">
                        <span className="font-semibold text-blue-700 block mb-1">
                          {report.baseline?.name || "Baseline"} Summary
                        </span>
                        <div className="space-y-0.5 text-gray-600">
                          <p>Samples: {selectedFeature.stats.baseline?.count}</p>
                          {selectedFeature.stats.baseline?.mean !== undefined && (
                            <>
                              <p>Mean: {selectedFeature.stats.baseline?.mean}</p>
                              <p>Std: {selectedFeature.stats.baseline?.std}</p>
                              <p>
                                Range: [{selectedFeature.stats.baseline?.min} ,{" "}
                                {selectedFeature.stats.baseline?.max}]
                              </p>
                            </>
                          )}
                          <p>
                            Missing:{" "}
                            {(selectedFeature.stats.baseline?.missing_pct * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>

                      <div className="bg-white p-2.5 rounded border">
                        <span className="font-semibold text-amber-700 block mb-1">
                          {report.target?.name || "Target"} Summary
                        </span>
                        <div className="space-y-0.5 text-gray-600">
                          <p>Samples: {selectedFeature.stats.target?.count}</p>
                          {selectedFeature.stats.target?.mean !== undefined && (
                            <>
                              <p>Mean: {selectedFeature.stats.target?.mean}</p>
                              <p>Std: {selectedFeature.stats.target?.std}</p>
                              <p>
                                Range: [{selectedFeature.stats.target?.min} ,{" "}
                                {selectedFeature.stats.target?.max}]
                              </p>
                            </>
                          )}
                          <p>
                            Missing: {(selectedFeature.stats.target?.missing_pct * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : null}
        </CardContent>
      )}
    </Card>
  );
};

DatasetDriftPanel.propTypes = {
  fileId: PropTypes.string.isRequired,
  files: PropTypes.array,
};

export default DatasetDriftPanel;
