/**
 * Neural Network Layer Health & Weight Diagnostics Panel
 * Displays layer-by-layer weight distributions, L2 norms, dead weight ratios, and stability status.
 * @module
 */

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Activity, AlertTriangle, CheckCircle, ShieldAlert, Cpu } from "lucide-react";
import axios from "@/shared/Axios";
import { BACKEND_MODEL_ANALYSIS } from "@/constants/Urls";
import { LayerDiagnosticsData } from "@/types/analysis";
import logger from "@/shared/logger";

interface LayerHealthPanelProps {
  jobId: string;
}

export default function LayerHealthPanel({ jobId }: LayerHealthPanelProps) {
  const [data, setData] = useState<LayerDiagnosticsData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;

    const fetchLayerDiagnostics = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await axios.get(`${BACKEND_MODEL_ANALYSIS}/${jobId}/layer-diagnostics`);
        if (isMounted) {
          setData(response.data as LayerDiagnosticsData);
        }
      } catch (err: any) {
        if (isMounted) {
          logger.error("Failed to fetch layer diagnostics:", err);
          const detail = err.response?.data?.detail || err.message || "Failed to load layer diagnostics";
          setError(detail);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchLayerDiagnostics();

    return () => {
      isMounted = false;
    };
  }, [jobId]);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8">
          <div className="flex items-center justify-center gap-3">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            <p className="text-sm text-gray-600">Profiling layer weights and model health...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-red-200 bg-red-50">
        <CardContent className="py-4">
          <p className="text-sm text-red-800">
            <strong>Layer Diagnostics Error:</strong> {error}
          </p>
        </CardContent>
      </Card>
    );
  }

  if (!data || !data.layers || data.layers.length === 0) {
    return null;
  }

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "HEALTHY":
        return (
          <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200 flex items-center gap-1">
            <CheckCircle className="h-3 w-3" />
            Healthy
          </Badge>
        );
      case "VANISHING_WEIGHTS":
        return (
          <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200 flex items-center gap-1">
            <AlertTriangle className="h-3 w-3" />
            Vanishing
          </Badge>
        );
      case "EXPLODING_WEIGHTS":
        return (
          <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200 flex items-center gap-1">
            <AlertTriangle className="h-3 w-3" />
            Exploding
          </Badge>
        );
      case "HIGH_SPARSITY":
        return (
          <Badge variant="outline" className="bg-orange-50 text-orange-700 border-orange-200 flex items-center gap-1">
            <AlertTriangle className="h-3 w-3" />
            High Sparsity
          </Badge>
        );
      case "NAN_OR_INF":
        return (
          <Badge variant="outline" className="bg-red-100 text-red-800 border-red-300 flex items-center gap-1">
            <ShieldAlert className="h-3 w-3" />
            NaN / Inf
          </Badge>
        );
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getOverallBadge = (status: string) => {
    switch (status) {
      case "HEALTHY":
        return (
          <Badge className="bg-green-600 text-white hover:bg-green-700 flex items-center gap-1.5 px-3 py-1">
            <CheckCircle className="h-4 w-4" />
            Healthy
          </Badge>
        );
      case "WARNING":
        return (
          <Badge className="bg-amber-500 text-white hover:bg-amber-600 flex items-center gap-1.5 px-3 py-1">
            <AlertTriangle className="h-4 w-4" />
            Warning
          </Badge>
        );
      case "CRITICAL":
        return (
          <Badge className="bg-red-600 text-white hover:bg-red-700 flex items-center gap-1.5 px-3 py-1">
            <ShieldAlert className="h-4 w-4" />
            Critical Issues
          </Badge>
        );
      default:
        return <Badge>{status}</Badge>;
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Cpu className="h-5 w-5 text-indigo-600" />
            <div>
              <CardTitle className="text-base">Neural Network Layer Health & Weight Diagnostics</CardTitle>
              <p className="text-xs text-gray-500 mt-0.5">
                Evaluates weight norms, sparsity, vanishing/exploding gradients, and numeric stability
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {getOverallBadge(data.overall_status)}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Quick summary stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
            <p className="text-xs text-gray-500 font-medium">Total Parameters</p>
            <p className="text-lg font-semibold text-gray-900 mt-1">
              {data.total_parameters.toLocaleString()}
            </p>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
            <p className="text-xs text-gray-500 font-medium">Trainable Layers</p>
            <p className="text-lg font-semibold text-gray-900 mt-1">
              {data.total_layers}
            </p>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
            <p className="text-xs text-gray-500 font-medium">Healthy Layers</p>
            <p className="text-lg font-semibold text-green-700 mt-1">
              {data.healthy_layers} / {data.total_layers}
            </p>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg border border-gray-100">
            <p className="text-xs text-gray-500 font-medium">Health Rate</p>
            <p className="text-lg font-semibold text-indigo-700 mt-1">
              {data.total_layers > 0
                ? `${Math.round((data.healthy_layers / data.total_layers) * 100)}%`
                : "100%"}
            </p>
          </div>
        </div>

        {/* Detailed Layers Table */}
        <div className="overflow-x-auto border rounded-md">
          <Table>
            <TableHeader>
              <TableRow className="bg-gray-50/75">
                <TableHead className="font-semibold text-gray-700">Layer Name</TableHead>
                <TableHead className="font-semibold text-gray-700">Type</TableHead>
                <TableHead className="font-semibold text-gray-700">Shape</TableHead>
                <TableHead className="font-semibold text-gray-700 text-right">Params</TableHead>
                <TableHead className="font-semibold text-gray-700 text-right">L2 Norm</TableHead>
                <TableHead className="font-semibold text-gray-700 text-right">Mean ± Std</TableHead>
                <TableHead className="font-semibold text-gray-700 text-right">Dead Weights</TableHead>
                <TableHead className="font-semibold text-gray-700 text-center">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.layers.map((layer) => (
                <TableRow key={layer.layer_name} className="hover:bg-gray-50/50">
                  <TableCell className="font-medium text-gray-900">
                    <div className="flex items-center gap-1.5">
                      <Activity className="h-3.5 w-3.5 text-gray-400" />
                      <span>{layer.layer_name}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-xs text-gray-600">{layer.layer_type}</TableCell>
                  <TableCell className="text-xs font-mono text-gray-500">
                    [{layer.shape.join(", ")}]
                  </TableCell>
                  <TableCell className="text-right text-sm text-gray-700">
                    {layer.param_count.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right text-sm font-mono text-gray-900">
                    {layer.l2_norm.toFixed(4)}
                  </TableCell>
                  <TableCell className="text-right text-xs font-mono text-gray-600">
                    {layer.mean.toFixed(3)} ± {layer.std.toFixed(3)}
                  </TableCell>
                  <TableCell className="text-right text-sm">
                    <span
                      className={
                        layer.dead_weights_ratio > 0.5
                          ? "text-amber-600 font-semibold"
                          : "text-gray-600"
                      }
                    >
                      {(layer.dead_weights_ratio * 100).toFixed(1)}%
                    </span>
                  </TableCell>
                  <TableCell className="text-center">
                    <div className="flex justify-center">{getStatusBadge(layer.status)}</div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
