import { useState, useEffect, useCallback, useMemo } from "react";
import PropTypes from "prop-types";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { getFeatureDistribution } from "../../services/FileServices";
import logger from "../../shared/logger";

const fmt = (val) => {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") return Number.isInteger(val) ? val : val.toFixed(4);
  return String(val);
};

const FeatureDistributionPanel = ({ fileId }) => {
  const [profileData, setProfileData] = useState(null);
  const [selectedColName, setSelectedColName] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(true);

  const fetchDistribution = useCallback(async (id) => {
    try {
      setLoading(true);
      setError(null);
      const data = await getFeatureDistribution(id);
      setProfileData(data);
      if (data && Array.isArray(data.columns) && data.columns.length > 0) {
        setSelectedColName(data.columns[0].column);
      } else {
        setSelectedColName("");
      }
    } catch (e) {
      logger.error("Error fetching feature distributions:", e);
      setError("Failed to load feature distribution profiles");
      setProfileData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (fileId) {
      fetchDistribution(fileId);
    } else {
      setProfileData(null);
      setSelectedColName("");
    }
  }, [fileId, fetchDistribution]);

  const activeColumn = useMemo(() => {
    if (!profileData || !Array.isArray(profileData.columns)) return null;
    return (
      profileData.columns.find((c) => c.column === selectedColName) ||
      profileData.columns[0] ||
      null
    );
  }, [profileData, selectedColName]);

  const totalOutliers = useMemo(() => {
    if (!profileData || !Array.isArray(profileData.columns)) return 0;
    return profileData.columns.reduce((acc, c) => acc + (c.outlier_count || 0), 0);
  }, [profileData]);

  const summary =
    profileData?.columns?.length != null
      ? `${profileData.columns.length} columns · ${totalOutliers} outliers detected`
      : "";

  return (
    <div className="space-y-4">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between py-2 text-sm font-medium text-muted-foreground hover:text-foreground"
      >
        <span className="flex items-center gap-2">
          <span className="font-semibold text-base text-foreground">
            Feature Distributions & Outlier Profiler
          </span>
          {summary && (
            <span className="font-normal text-xs text-muted-foreground">({summary})</span>
          )}
        </span>
        <span>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="pt-2">
          {loading && (
            <div
              data-testid="distribution-loading"
              className="flex h-24 items-center justify-center text-muted-foreground text-sm"
            >
              Loading feature distributions...
            </div>
          )}

          {error && !loading && (
            <div
              data-testid="distribution-error"
              className="flex h-20 items-center justify-center text-destructive text-sm font-medium"
            >
              {error}
            </div>
          )}

          {!loading &&
            !error &&
            (!profileData || !profileData.columns || profileData.columns.length === 0) && (
              <div className="flex h-20 items-center justify-center text-muted-foreground text-sm">
                No distribution profiles available
              </div>
            )}

          {!loading && !error && profileData && activeColumn && (
            <div className="space-y-5">
              {/* Feature selector */}
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-lg bg-muted/40 p-3">
                <div className="flex items-center gap-3">
                  <label htmlFor="feature-select" className="text-sm font-medium">
                    Select Feature:
                  </label>
                  <select
                    id="feature-select"
                    aria-label="Select feature"
                    value={activeColumn.column}
                    onChange={(e) => setSelectedColName(e.target.value)}
                    className="rounded-md border bg-background px-3 py-1.5 text-sm font-medium shadow-sm focus:ring-2 focus:ring-primary"
                  >
                    {profileData.columns.map((col) => (
                      <option key={col.column} value={col.column}>
                        {col.column}{" "}
                        {col.is_numeric
                          ? `(numeric${col.outlier_count > 0 ? ` · ${col.outlier_count} outliers` : ""})`
                          : "(categorical)"}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <span className="rounded bg-secondary px-2 py-0.5 text-secondary-foreground font-mono">
                    {activeColumn.dtype}
                  </span>
                  <span className="rounded bg-blue-100 dark:bg-blue-950/50 text-blue-800 dark:text-blue-300 px-2 py-0.5 font-medium">
                    {activeColumn.count} non-null
                  </span>
                  {activeColumn.null_count > 0 && (
                    <span className="rounded bg-amber-100 dark:bg-amber-950/50 text-amber-800 dark:text-amber-300 px-2 py-0.5 font-medium">
                      {activeColumn.null_count} nulls
                    </span>
                  )}
                </div>
              </div>

              {/* Numeric Feature Profiling */}
              {activeColumn.is_numeric ? (
                <div className="space-y-4">
                  {/* Outlier Alert Banner */}
                  {activeColumn.outlier_count > 0 ? (
                    <div className="rounded-lg border border-amber-300 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/40 p-3 text-amber-900 dark:text-amber-200 text-sm flex items-start gap-2">
                      <span className="text-base leading-none">⚠️</span>
                      <div>
                        <div className="font-semibold">
                          {activeColumn.outlier_count} Outlier
                          {activeColumn.outlier_count > 1 ? "s" : ""} Detected (
                          {activeColumn.outlier_percentage}% of column)
                        </div>
                        <div className="text-xs text-amber-800/90 dark:text-amber-300/90 mt-0.5">
                          Values outside Tukey&apos;s 1.5×IQR boundary: [
                          {fmt(activeColumn.lower_bound)}, {fmt(activeColumn.upper_bound)}
                          ]. Consider scaling or clipping during preprocessing.
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-lg border border-green-200 bg-green-50/50 dark:border-green-900 dark:bg-green-950/20 p-2.5 text-xs text-green-800 dark:text-green-300 flex items-center gap-2">
                      <span>✓</span>
                      <span>
                        No extreme outliers detected within Tukey&apos;s 1.5×IQR boundaries.
                      </span>
                    </div>
                  )}

                  {/* Summary Metric Tiles */}
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <div className="rounded-lg border bg-card p-3 shadow-sm">
                      <div className="text-xs text-muted-foreground font-medium">Mean ± Std</div>
                      <div className="mt-1 font-semibold text-sm">
                        {fmt(activeColumn.mean)}
                        <span className="text-xs text-muted-foreground font-normal ml-1">
                          ± {fmt(activeColumn.std)}
                        </span>
                      </div>
                    </div>

                    <div className="rounded-lg border bg-card p-3 shadow-sm">
                      <div className="text-xs text-muted-foreground font-medium">Median & IQR</div>
                      <div className="mt-1 font-semibold text-sm">
                        {fmt(activeColumn.median)}
                        <span className="text-xs text-muted-foreground font-normal ml-1">
                          (IQR: {fmt(activeColumn.iqr)})
                        </span>
                      </div>
                    </div>

                    <div className="rounded-lg border bg-card p-3 shadow-sm">
                      <div className="text-xs text-muted-foreground font-medium">
                        Min / Max Range
                      </div>
                      <div className="mt-1 font-semibold text-sm">
                        [{fmt(activeColumn.min)}, {fmt(activeColumn.max)}]
                      </div>
                    </div>

                    <div className="rounded-lg border bg-card p-3 shadow-sm">
                      <div className="text-xs text-muted-foreground font-medium">
                        Quartiles (Q1 / Q3)
                      </div>
                      <div className="mt-1 font-semibold text-sm">
                        [{fmt(activeColumn.q25)}, {fmt(activeColumn.q75)}]
                      </div>
                    </div>
                  </div>

                  {/* Histogram Chart */}
                  <div className="rounded-lg border bg-card p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-semibold text-foreground">
                        Value Distribution (Histogram)
                      </h4>
                      <div className="text-xs text-muted-foreground">
                        Skewness:{" "}
                        <span className="font-mono font-medium text-foreground">
                          {fmt(activeColumn.skewness)}
                        </span>
                      </div>
                    </div>

                    {activeColumn.histogram && activeColumn.histogram.length > 0 ? (
                      <div className="h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={activeColumn.histogram}
                            margin={{ top: 10, right: 15, left: -10, bottom: 25 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              vertical={false}
                              stroke="#e5e7eb"
                            />
                            <XAxis
                              dataKey="bin_label"
                              angle={-25}
                              textAnchor="end"
                              height={45}
                              tick={{ fontSize: 11 }}
                            />
                            <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                            <Tooltip
                              formatter={(value) => [`${value} samples`, "Frequency"]}
                              labelFormatter={(label) => `Range: ${label}`}
                            />
                            <Bar
                              dataKey="count"
                              name="Count"
                              fill="#3b82f6"
                              radius={[4, 4, 0, 0]}
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <div className="flex h-40 items-center justify-center text-xs text-muted-foreground">
                        No finite values available to render histogram.
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                /* Categorical Feature Profiling */
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    <div className="rounded-lg border bg-card p-3 shadow-sm">
                      <div className="text-xs text-muted-foreground font-medium">
                        Unique Categories
                      </div>
                      <div className="mt-1 font-semibold text-base">
                        {fmt(activeColumn.unique_count)}
                      </div>
                    </div>
                    <div className="rounded-lg border bg-card p-3 shadow-sm">
                      <div className="text-xs text-muted-foreground font-medium">Total Samples</div>
                      <div className="mt-1 font-semibold text-base">{fmt(activeColumn.count)}</div>
                    </div>
                    <div className="rounded-lg border bg-card p-3 shadow-sm">
                      <div className="text-xs text-muted-foreground font-medium">
                        Missing / Null
                      </div>
                      <div className="mt-1 font-semibold text-base">
                        {fmt(activeColumn.null_count)}
                      </div>
                    </div>
                  </div>

                  {activeColumn.top_categories && activeColumn.top_categories.length > 0 ? (
                    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                      <div className="rounded-lg border bg-card p-4 shadow-sm">
                        <h4 className="text-sm font-semibold text-foreground mb-3">
                          Top Categories Frequency
                        </h4>
                        <div className="h-64 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={activeColumn.top_categories}
                              layout="vertical"
                              margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
                            >
                              <CartesianGrid
                                strokeDasharray="3 3"
                                horizontal={false}
                                stroke="#e5e7eb"
                              />
                              <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                              <YAxis
                                dataKey="category"
                                type="category"
                                tick={{ fontSize: 11 }}
                                width={80}
                              />
                              <Tooltip
                                formatter={(value, name, item) => [
                                  `${value} (${item.payload.percentage}%)`,
                                  "Count",
                                ]}
                              />
                              <Bar
                                dataKey="count"
                                name="Count"
                                fill="#10b981"
                                radius={[0, 4, 4, 0]}
                              />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div className="rounded-lg border bg-card p-4 shadow-sm overflow-auto max-h-[320px]">
                        <h4 className="text-sm font-semibold text-foreground mb-3">
                          Category Breakdown
                        </h4>
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b bg-muted/40 text-muted-foreground">
                              <th className="px-3 py-2 text-left font-semibold">Category</th>
                              <th className="px-3 py-2 text-right font-semibold">Count</th>
                              <th className="px-3 py-2 text-right font-semibold">Frequency</th>
                            </tr>
                          </thead>
                          <tbody>
                            {activeColumn.top_categories.map((cat) => (
                              <tr key={cat.category} className="border-b hover:bg-muted/30">
                                <td className="px-3 py-2 font-medium">{cat.category}</td>
                                <td className="px-3 py-2 text-right">{cat.count}</td>
                                <td className="px-3 py-2 text-right font-mono">
                                  {cat.percentage}%
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
                      No categories found.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

FeatureDistributionPanel.propTypes = {
  fileId: PropTypes.string.isRequired,
};

export default FeatureDistributionPanel;
