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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";

const API = import.meta.env.VITE_BACKEND_URL;

async function fetchJSON(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!j.success) throw new Error(j.message);
  return j.data;
}

// ── Confusion Matrix ──────────────────────────────────────────────────────────
function ConfusionMatrix({ modelName }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await fetchJSON(`${API}/api/model/${modelName}/interpretability/confusion-matrix`);
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [modelName]);

  const max = data ? Math.max(...data.matrix.flat()) : 1;

  return (
    <div className="space-y-3">
      <Button size="sm" onClick={load} disabled={loading}>
        {loading ? "Computing..." : "Compute Confusion Matrix"}
      </Button>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {data && (
        <div className="overflow-x-auto">
          <table className="text-xs border-collapse">
            <thead>
              <tr>
                <th className="p-1 text-muted-foreground">Act\Pred</th>
                {data.labels.map((l) => (
                  <th key={l} className="p-1 font-medium">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.matrix.map((row, i) => (
                <tr key={i}>
                  <td className="p-1 font-medium text-muted-foreground">{data.labels[i]}</td>
                  {row.map((val, j) => (
                    <td
                      key={j}
                      className="p-2 text-center font-mono border"
                      style={{
                        backgroundColor: `rgba(59,130,246,${val / max})`,
                        color: val / max > 0.5 ? "white" : "inherit",
                      }}
                    >
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Classification Report ─────────────────────────────────────────────────────
function ClassificationReport({ modelName }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await fetchJSON(
        `${API}/api/model/${modelName}/interpretability/classification-report`,
      );
      setData(d.report);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [modelName]);

  const rows = data
    ? Object.entries(data).filter(([k]) => !["accuracy", "macro avg", "weighted avg"].includes(k))
    : [];
  const summary = data
    ? Object.entries(data).filter(([k]) => ["accuracy", "macro avg", "weighted avg"].includes(k))
    : [];

  return (
    <div className="space-y-3">
      <Button size="sm" onClick={load} disabled={loading}>
        {loading ? "Computing..." : "Compute Classification Report"}
      </Button>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {data && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="py-1.5 text-left">Class</th>
                <th className="py-1.5 text-right">Precision</th>
                <th className="py-1.5 text-right">Recall</th>
                <th className="py-1.5 text-right">F1</th>
                <th className="py-1.5 text-right">Support</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([cls, m]) => (
                <tr key={cls} className="border-b last:border-0">
                  <td className="py-1 font-mono">{cls}</td>
                  <td className="py-1 text-right tabular-nums">{m.precision?.toFixed(3)}</td>
                  <td className="py-1 text-right tabular-nums">{m.recall?.toFixed(3)}</td>
                  <td className="py-1 text-right tabular-nums">{m["f1-score"]?.toFixed(3)}</td>
                  <td className="py-1 text-right tabular-nums">{m.support}</td>
                </tr>
              ))}
            </tbody>
            <tfoot className="border-t text-muted-foreground">
              {summary.map(([k, m]) => (
                <tr key={k}>
                  <td className="py-1 italic">{k}</td>
                  <td className="py-1 text-right tabular-nums">
                    {typeof m === "number" ? m.toFixed(3) : m.precision?.toFixed(3)}
                  </td>
                  <td className="py-1 text-right tabular-nums">
                    {typeof m === "number" ? "—" : m.recall?.toFixed(3)}
                  </td>
                  <td className="py-1 text-right tabular-nums">
                    {typeof m === "number" ? "—" : m["f1-score"]?.toFixed(3)}
                  </td>
                  <td className="py-1 text-right tabular-nums">
                    {typeof m === "number" ? "—" : m.support}
                  </td>
                </tr>
              ))}
            </tfoot>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Feature Importance ────────────────────────────────────────────────────────
function FeatureImportance({ modelName }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await fetchJSON(
        `${API}/api/model/${modelName}/interpretability/feature-importance`,
      );
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [modelName]);

  return (
    <div className="space-y-3">
      <Button size="sm" onClick={load} disabled={loading}>
        {loading ? "Computing..." : "Compute Feature Importance"}
      </Button>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {data && (
        <>
          <p className="text-xs text-muted-foreground">
            Baseline loss: <strong>{data.baseline_loss}</strong> — bars show increase in loss when
            feature is permuted
          </p>
          <ResponsiveContainer width="100%" height={Math.max(200, data.importances.length * 28)}>
            <BarChart
              layout="vertical"
              data={data.importances.slice(0, 20)}
              margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={100} />
              <Tooltip formatter={(v) => v.toFixed(5)} />
              <Bar dataKey="importance" isAnimationActive={false}>
                {data.importances.slice(0, 20).map((entry, i) => (
                  <Cell key={i} fill={entry.importance > 0 ? "#3b82f6" : "#e5e7eb"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

// ── Prediction Explorer ───────────────────────────────────────────────────────
function PredictionExplorer({ modelName }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await fetchJSON(
        `${API}/api/model/${modelName}/interpretability/predictions?n_samples=20`,
      );
      setData(d);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [modelName]);

  return (
    <div className="space-y-3">
      <Button size="sm" onClick={load} disabled={loading}>
        {loading ? "Loading..." : "Load Predictions"}
      </Button>
      {error && <p className="text-sm text-destructive">{error}</p>}
      {data && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b">
                <th className="py-1.5 text-left">Index</th>
                <th className="py-1.5 text-right">Actual</th>
                <th className="py-1.5 text-right">Predicted</th>
                {data.is_classification ? (
                  <th className="py-1.5 text-right">Confidence</th>
                ) : (
                  <th className="py-1.5 text-right">|Error|</th>
                )}
                {data.is_classification && <th className="py-1.5 text-center">✓</th>}
              </tr>
            </thead>
            <tbody>
              {data.samples.map((row) => (
                <tr
                  key={row.index}
                  className={`border-b last:border-0 ${
                    data.is_classification && !row.correct ? "bg-red-50" : ""
                  }`}
                >
                  <td className="py-1 font-mono text-muted-foreground">{row.index}</td>
                  <td className="py-1 text-right tabular-nums">{row.actual}</td>
                  <td className="py-1 text-right tabular-nums">{row.predicted}</td>
                  {data.is_classification ? (
                    <td className="py-1 text-right tabular-nums">
                      {(row.confidence * 100).toFixed(1)}%
                    </td>
                  ) : (
                    <td className="py-1 text-right tabular-nums">{row.error}</td>
                  )}
                  {data.is_classification && (
                    <td className="py-1 text-center">{row.correct ? "✅" : "❌"}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────
export default function InterpretabilityPanel({ modelName, configSaved }) {
  if (!modelName || !configSaved) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Model Interpretability</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="confusion">
          <TabsList className="mb-4">
            <TabsTrigger value="confusion">Confusion Matrix</TabsTrigger>
            <TabsTrigger value="report">Class Report</TabsTrigger>
            <TabsTrigger value="importance">Feature Importance</TabsTrigger>
            <TabsTrigger value="predictions">Predictions</TabsTrigger>
          </TabsList>
          <TabsContent value="confusion">
            <ConfusionMatrix modelName={modelName} />
          </TabsContent>
          <TabsContent value="report">
            <ClassificationReport modelName={modelName} />
          </TabsContent>
          <TabsContent value="importance">
            <FeatureImportance modelName={modelName} />
          </TabsContent>
          <TabsContent value="predictions">
            <PredictionExplorer modelName={modelName} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

InterpretabilityPanel.propTypes = {
  modelName: PropTypes.string,
  configSaved: PropTypes.bool.isRequired,
};
