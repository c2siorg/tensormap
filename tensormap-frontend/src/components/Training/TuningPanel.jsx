import { useState, useCallback } from "react";
import PropTypes from "prop-types";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";

const API = import.meta.env.VITE_BACKEND_URL;

const DEFAULT_SPACE = {
  optimizer: ["adam", "sgd"],
  learning_rate: [0.001, 0.01],
  batch_size: [16, 32],
  epochs: [10, 20],
};

function parseSpaceField(raw) {
  try {
    return raw
      .split(",")
      .map((s) => {
        const v = s.trim();
        const n = Number(v);
        return isNaN(n) ? v : n;
      })
      .filter((v) => v !== "");
  } catch {
    return [];
  }
}

export default function TuningPanel({ modelName, configSaved, onApplyBest }) {
  const [strategy, setStrategy] = useState("random");
  const [nTrials, setNTrials] = useState(6);
  const [space, setSpace] = useState({
    optimizer: "adam,sgd",
    learning_rate: "0.001,0.01",
    batch_size: "16,32",
    epochs: "10,20",
  });
  const [results, setResults] = useState([]);
  const [best, setBest] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [error, setError] = useState(null);

  // Listen for Socket.IO tuning events from parent (passed via prop would be
  // cleaner but Training.jsx already owns the socket — we poll via fetch here
  // for simplicity and use the REST response for final results)

  const handleRun = useCallback(async () => {
    if (!modelName || !configSaved) return;
    setLoading(true);
    setError(null);
    setResults([]);
    setBest(null);
    setProgress({ current: 0, total: 0 });

    const parsedSpace = {};
    for (const [k, v] of Object.entries(space)) {
      const arr = parseSpaceField(v);
      if (arr.length > 0) parsedSpace[k] = arr;
    }

    try {
      const resp = await fetch(`${API}/api/model/tune`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_name: modelName,
          strategy,
          n_trials: Number(nTrials),
          space: parsedSpace,
        }),
      });
      const json = await resp.json();
      if (!json.success) throw new Error(json.message);
      setResults(json.data.results);
      setBest(json.data.best);
      setProgress({ current: json.data.total_trials, total: json.data.total_trials });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [modelName, configSaved, strategy, nTrials, space]);

  if (!modelName || !configSaved) return null;

  const chartData = results.map((r) => ({
    name: `T${r.trial}`,
    val_loss: r.val_loss,
    val_metric: r.val_metric,
    isBest: best && r.trial === best.trial,
  }));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Hyperparameter Tuning</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Config row */}
        <div className="flex flex-wrap gap-4 items-end">
          <div className="space-y-1">
            <Label>Strategy</Label>
            <Select value={strategy} onValueChange={setStrategy}>
              <SelectTrigger className="w-36">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="random">Random Search</SelectItem>
                <SelectItem value="grid">Grid Search</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {strategy === "random" && (
            <div className="space-y-1">
              <Label>Trials</Label>
              <Input
                type="number"
                min="2"
                max="50"
                className="w-20"
                value={nTrials}
                onChange={(e) => setNTrials(e.target.value)}
              />
            </div>
          )}
          <Button onClick={handleRun} disabled={loading}>
            {loading ? "Searching..." : "Run Search"}
          </Button>
        </div>

        {/* Search space */}
        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Search Space (comma-separated values)
          </p>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {Object.entries(space).map(([key, val]) => (
              <div key={key} className="space-y-1">
                <Label className="text-xs">{key}</Label>
                <Input
                  className="text-xs h-8"
                  value={val}
                  onChange={(e) => setSpace((prev) => ({ ...prev, [key]: e.target.value }))}
                  placeholder={DEFAULT_SPACE[key]?.join(",") ?? ""}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Progress bar */}
        {loading && (
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">
              Running trials... {progress.current}/{progress.total || "?"}
            </p>
            <div className="h-1.5 w-full rounded-full bg-muted">
              <div
                className="h-1.5 rounded-full bg-primary transition-all"
                style={{
                  width: progress.total ? `${(progress.current / progress.total) * 100}%` : "5%",
                }}
              />
            </div>
          </div>
        )}

        {error && <p className="text-sm text-destructive">{error}</p>}

        {/* Best result banner */}
        {best && (
          <div className="rounded-md border border-green-200 bg-green-50 p-3 text-sm">
            <p className="font-semibold text-green-800 mb-1">
              Best trial #{best.trial} — val_loss: {best.val_loss}
            </p>
            <div className="flex flex-wrap gap-3 text-xs text-green-700">
              {Object.entries(best.params).map(([k, v]) => (
                <span key={k}>
                  <strong>{k}:</strong> {v}
                </span>
              ))}
            </div>
            <Button
              size="sm"
              variant="outline"
              className="mt-2 border-green-400 text-green-800 hover:bg-green-100"
              onClick={() => onApplyBest && onApplyBest(best.params)}
            >
              Apply Best Params
            </Button>
          </div>
        )}

        {/* Results chart */}
        {results.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">Val Loss by Trial</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} width={52} />
                <Tooltip
                  formatter={(v) => v.toFixed(4)}
                  labelFormatter={(l) => {
                    const r = results.find((x) => `T${x.trial}` === l);
                    if (!r) return l;
                    return `Trial ${r.trial}: ${JSON.stringify(r.params)}`;
                  }}
                />
                <Bar dataKey="val_loss" isAnimationActive={false}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={entry.isBest ? "#22c55e" : "#3b82f6"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Results table */}
        {results.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b">
                  <th className="py-1.5 text-left">#</th>
                  <th className="py-1.5 text-left">Params</th>
                  <th className="py-1.5 text-right">Val Loss</th>
                  <th className="py-1.5 text-right">Val Metric</th>
                </tr>
              </thead>
              <tbody>
                {results
                  .slice()
                  .sort((a, b) => a.val_loss - b.val_loss)
                  .map((r) => (
                    <tr
                      key={r.trial}
                      className={`border-b last:border-0 ${
                        best && r.trial === best.trial ? "bg-green-50 font-semibold" : ""
                      }`}
                    >
                      <td className="py-1 tabular-nums">{r.trial}</td>
                      <td className="py-1 font-mono text-muted-foreground">
                        {Object.entries(r.params)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(", ")}
                      </td>
                      <td className="py-1 text-right tabular-nums">{r.val_loss.toFixed(4)}</td>
                      <td className="py-1 text-right tabular-nums">{r.val_metric.toFixed(4)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

TuningPanel.propTypes = {
  modelName: PropTypes.string,
  configSaved: PropTypes.bool.isRequired,
  onApplyBest: PropTypes.func,
};
