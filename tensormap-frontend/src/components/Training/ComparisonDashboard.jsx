import { useState } from "react";
import PropTypes from "prop-types";
import { X, Plus } from "lucide-react";
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
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";

// Distinct colors for up to 8 runs
const RUN_COLORS = [
  "#3b82f6",
  "#ef4444",
  "#10b981",
  "#f59e0b",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
];

/**
 * Merges multiple runs' epoch arrays into a single array keyed by epoch,
 * so recharts can render overlaid lines.
 *
 * Input:  runs = [{ name, metrics: [{epoch, loss, metric}] }, ...]
 * Output: [{ epoch: 1, "Run A loss": 0.5, "Run B loss": 0.6, ... }, ...]
 */
function mergeRunsForChart(runs, field) {
  if (runs.length === 0) return [];
  const maxEpochs = Math.max(...runs.map((r) => r.metrics.length));
  return Array.from({ length: maxEpochs }, (_, i) => {
    const row = { epoch: i + 1 };
    runs.forEach((run) => {
      const pt = run.metrics[i];
      if (pt) row[`${run.name} ${field}`] = pt[field] ?? null;
    });
    return row;
  });
}

export default function ComparisonDashboard({ availableRuns }) {
  const [selected, setSelected] = useState([]); // array of run names
  const [addingRun, setAddingRun] = useState(false);

  if (availableRuns.length < 2) return null;

  const activeRuns = availableRuns.filter((r) => selected.includes(r.name));
  const lossData = mergeRunsForChart(activeRuns, "loss");
  const metricName = activeRuns[0]?.metrics[0]?.metric_name ?? "metric";
  const metricData = mergeRunsForChart(activeRuns, "metric");

  const unselected = availableRuns.filter((r) => !selected.includes(r.name));

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Run Comparison</CardTitle>
          {unselected.length > 0 && (
            <div className="flex items-center gap-2">
              {addingRun ? (
                <>
                  <Select
                    onValueChange={(v) => {
                      setSelected((prev) => [...prev, v]);
                      setAddingRun(false);
                    }}
                  >
                    <SelectTrigger className="h-7 w-40 text-xs">
                      <SelectValue placeholder="Pick a run" />
                    </SelectTrigger>
                    <SelectContent>
                      {unselected.map((r) => (
                        <SelectItem key={r.name} value={r.name} className="text-xs">
                          {r.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7"
                    onClick={() => setAddingRun(false)}
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs"
                  onClick={() => setAddingRun(true)}
                >
                  <Plus className="mr-1 h-3.5 w-3.5" />
                  Add run
                </Button>
              )}
            </div>
          )}
        </div>

        {/* Active run chips */}
        {selected.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {selected.map((name, i) => (
              <span
                key={name}
                className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs"
                style={{ borderColor: RUN_COLORS[i % RUN_COLORS.length] }}
              >
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: RUN_COLORS[i % RUN_COLORS.length] }}
                />
                {name}
                <button
                  onClick={() => setSelected((prev) => prev.filter((n) => n !== name))}
                  className="ml-0.5 opacity-60 hover:opacity-100"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        )}
      </CardHeader>

      {selected.length === 0 ? (
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Add runs above to compare their training curves side by side.
          </p>
        </CardContent>
      ) : (
        <CardContent className="space-y-4">
          {/* Loss overlay */}
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">Loss</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={lossData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} width={52} />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {activeRuns.map((run, i) => (
                  <Line
                    key={run.name}
                    type="monotone"
                    dataKey={`${run.name} loss`}
                    stroke={RUN_COLORS[i % RUN_COLORS.length]}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Metric overlay */}
          {activeRuns.some((r) => r.metrics.some((m) => m.metric != null)) && (
            <div>
              <p className="mb-1 text-xs font-medium text-muted-foreground">
                {metricName.charAt(0).toUpperCase() + metricName.slice(1)}
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={metricData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} width={52} />
                  <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {activeRuns.map((run, i) => (
                    <Line
                      key={run.name}
                      type="monotone"
                      dataKey={`${run.name} metric`}
                      stroke={RUN_COLORS[i % RUN_COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}

ComparisonDashboard.propTypes = {
  availableRuns: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      metrics: PropTypes.arrayOf(PropTypes.object).isRequired,
    }),
  ).isRequired,
};
