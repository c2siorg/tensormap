/**
 * ROC & Precision-Recall Curves Panel Component
 * Displays threshold-varying diagnostic curves (ROC with AUC, PR with Average Precision)
 * with One-vs-Rest multi-class and binary support.
 * @module
 */

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { RocCurveData, ClassCurveData } from "@/types/analysis";

const CLASS_COLORS = [
  "#2563eb", // blue
  "#16a34a", // green
  "#d97706", // amber
  "#dc2626", // red
  "#9333ea", // purple
  "#0891b2", // cyan
  "#ea580c", // orange
  "#4f46e5", // indigo
];

interface RocPrCurvePanelProps {
  data: RocCurveData | null;
  isLoading?: boolean;
  error?: string | null;
}

export default function RocPrCurvePanel({
  data,
  isLoading = false,
  error = null,
}: RocPrCurvePanelProps) {
  const [curveType, setCurveType] = useState<"roc" | "pr">("roc");
  const [selectedClassIndex, setSelectedClassIndex] = useState<string>("all");

  const classes: ClassCurveData[] = data?.classes || [];
  const isMultiClass = classes.length > 1;

  // Filter classes to display based on selectedClassIndex
  const activeClasses = useMemo(() => {
    if (selectedClassIndex === "all") {
      return classes;
    }
    const idx = parseInt(selectedClassIndex, 10);
    const found = classes.find((c) => c.class_index === idx);
    return found ? [found] : classes;
  }, [classes, selectedClassIndex]);

  // Transform coordinates for Recharts ScatterChart
  const seriesData = useMemo(() => {
    return activeClasses.map((cls, idx) => {
      const colorIndex = cls.class_index % CLASS_COLORS.length;
      const color = CLASS_COLORS[colorIndex] || CLASS_COLORS[idx % CLASS_COLORS.length];

      if (curveType === "roc") {
        const points = cls.roc.fpr.map((fprVal, i) => ({
          x: fprVal,
          y: cls.roc.tpr[i],
          threshold: cls.roc.thresholds[i],
          className: cls.class_name,
        }));
        return {
          name: isMultiClass ? `${cls.class_name} (AUC: ${cls.roc.auc?.toFixed(3) ?? "N/A"})` : "ROC Curve",
          points,
          color,
          auc: cls.roc.auc,
        };
      } else {
        const points = cls.pr.recall.map((recallVal, i) => ({
          x: recallVal,
          y: cls.pr.precision[i],
          threshold: cls.pr.thresholds[i],
          className: cls.class_name,
        }));
        return {
          name: isMultiClass ? `${cls.class_name} (PR-AUC: ${cls.pr.auc?.toFixed(3) ?? "N/A"})` : "PR Curve",
          points,
          color,
          auc: cls.pr.auc,
        };
      }
    });
  }, [activeClasses, curveType, isMultiClass]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Diagnostic Curves (ROC & Precision-Recall)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center items-center py-16">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-2"></div>
            <p className="ml-3 text-sm text-gray-500">Loading curve analysis...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Diagnostic Curves (ROC & Precision-Recall)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-800">
              <strong>Evaluation Error:</strong> {error}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data || classes.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Diagnostic Curves (ROC & Precision-Recall)</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-gray-500 py-8 text-center">
            No curve evaluation data available for this model.
          </p>
        </CardContent>
      </Card>
    );
  }

  // Active single class AUC if filtered
  const singleClass = selectedClassIndex !== "all" && activeClasses.length === 1 ? activeClasses[0] : null;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <CardTitle className="text-base">ROC & Precision-Recall Curves</CardTitle>
            <p className="text-xs text-gray-500 mt-1">
              Threshold-varying diagnostic performance across decision boundaries
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Macro AUC Badges */}
            {curveType === "roc" && data.macro_roc_auc !== null && (
              <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 font-medium">
                Macro ROC-AUC: {data.macro_roc_auc.toFixed(4)}
              </Badge>
            )}

            {curveType === "pr" && data.macro_pr_auc !== null && (
              <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200 font-medium">
                Macro PR-AUC: {data.macro_pr_auc.toFixed(4)}
              </Badge>
            )}

            {/* Single class AUC badge */}
            {singleClass && (
              <Badge variant="outline" className="bg-emerald-50 text-emerald-700 border-emerald-200 font-medium">
                {singleClass.class_name} {curveType === "roc" ? "AUC" : "PR-AUC"}:{" "}
                {(curveType === "roc" ? singleClass.roc.auc : singleClass.pr.auc)?.toFixed(4) ?? "N/A"}
              </Badge>
            )}
          </div>
        </div>

        {/* Controls: Curve Type & Class Selector */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 border-t border-gray-100">
          <Tabs
            value={curveType}
            onValueChange={(val) => setCurveType(val as "roc" | "pr")}
            className="w-auto"
          >
            <TabsList className="grid grid-cols-2 w-64">
              <TabsTrigger value="roc">ROC Curve</TabsTrigger>
              <TabsTrigger value="pr">Precision-Recall</TabsTrigger>
            </TabsList>
          </Tabs>

          {isMultiClass && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 font-medium whitespace-nowrap">Class View:</span>
              <Select
                value={selectedClassIndex}
                onValueChange={(val) => setSelectedClassIndex(val)}
              >
                <SelectTrigger className="w-48 h-8 text-xs">
                  <SelectValue placeholder="Select class" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Classes (OvR)</SelectItem>
                  {classes.map((cls) => (
                    <SelectItem key={cls.class_index} value={cls.class_index.toString()}>
                      {cls.class_name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <ResponsiveContainer width="100%" height={380}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 25, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="x"
              name={curveType === "roc" ? "False Positive Rate" : "Recall"}
              domain={[0, 1]}
              tickCount={6}
              label={{
                value: curveType === "roc" ? "False Positive Rate (1 - Specificity)" : "Recall",
                position: "insideBottom",
                offset: -12,
                fontSize: 12,
                fill: "#4b5563",
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name={curveType === "roc" ? "True Positive Rate" : "Precision"}
              domain={[0, 1]}
              tickCount={6}
              label={{
                value: curveType === "roc" ? "True Positive Rate (Sensitivity)" : "Precision",
                angle: -90,
                position: "insideLeft",
                offset: 5,
                fontSize: 12,
                fill: "#4b5563",
              }}
            />

            {/* Random chance baseline for ROC: diagonal (0,0) to (1,1) */}
            {curveType === "roc" && (
              <ReferenceLine
                segment={[
                  { x: 0, y: 0 },
                  { x: 1, y: 1 },
                ]}
                stroke="#9ca3af"
                strokeDasharray="4 4"
                label={{
                  value: "Random Chance",
                  position: "insideTopRight",
                  fill: "#9ca3af",
                  fontSize: 11,
                }}
              />
            )}

            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const pt = payload[0].payload;
                return (
                  <div className="bg-gray-900 text-white text-xs rounded-md shadow-lg p-2.5 space-y-1">
                    <p className="font-semibold text-blue-300">{pt.className}</p>
                    <p>
                      {curveType === "roc" ? "FPR" : "Recall"}: {Number(pt.x).toFixed(4)}
                    </p>
                    <p>
                      {curveType === "roc" ? "TPR (Sensitivity)" : "Precision"}: {Number(pt.y).toFixed(4)}
                    </p>
                    {pt.threshold !== undefined && Number.isFinite(pt.threshold) && Math.abs(pt.threshold) < 1000 && (
                      <p className="text-gray-400">
                        Threshold: {Number(pt.threshold).toFixed(4)}
                      </p>
                    )}
                  </div>
                );
              }}
            />

            {isMultiClass && <Legend verticalAlign="top" height={36} />}

            {seriesData.map((series) => (
              <Scatter
                key={series.name}
                name={series.name}
                data={series.points}
                line={{ stroke: series.color, strokeWidth: 2 }}
                shape={() => null}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
