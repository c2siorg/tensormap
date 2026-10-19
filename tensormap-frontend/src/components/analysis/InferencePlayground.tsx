/**
 * Interactive Model Inference Playground Component.
 * Enables live predictions on custom inputs, slider-based feature perturbation,
 * test sample comparisons, and probability distributions.
 * @module
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Sparkles,
  RefreshCw,
  Zap,
  Dices,
  CheckCircle,
  AlertTriangle,
  HelpCircle,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { getFeatureSchema, runInference } from "@/services/InferenceService";
import type {
  ModelFeatureSchemaResponse,
  InferenceResponse,
  SampleTestRecord,
} from "@/types/inference";
import logger from "@/shared/logger";

interface InferencePlaygroundProps {
  jobId: string;
}

export default function InferencePlayground({ jobId }: InferencePlaygroundProps) {
  const [schema, setSchema] = useState<ModelFeatureSchemaResponse | null>(null);
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<InferenceResponse | null>(null);
  const [activeSample, setActiveSample] = useState<SampleTestRecord | null>(null);
  const [isLoadingSchema, setIsLoadingSchema] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [livePredict, setLivePredict] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Load feature schema on mount or jobId change
  useEffect(() => {
    let isMounted = true;
    const fetchSchema = async () => {
      setIsLoadingSchema(true);
      setError(null);
      try {
        const data = await getFeatureSchema(jobId);
        if (!isMounted) return;
        setSchema(data);

        // Initialize inputs with default (median) values
        const initialInputs: Record<string, number> = {};
        data.features.forEach((feat) => {
          initialInputs[feat.name] = feat.default_value;
        });
        setInputs(initialInputs);

        // Run initial prediction
        try {
          const initialPred = await runInference(jobId, initialInputs);
          if (isMounted) setPrediction(initialPred);
        } catch (predErr: any) {
          logger.warn("Initial inference execution failed:", predErr);
        }
      } catch (err: any) {
        if (!isMounted) return;
        const msg = err.response?.data?.detail || err.message || "Failed to load feature schema";
        logger.error("Error loading feature schema:", err);
        setError(msg);
      } finally {
        if (isMounted) setIsLoadingSchema(false);
      }
    };

    fetchSchema();
    return () => {
      isMounted = false;
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, [jobId]);

  // Execute inference on current inputs
  const executePrediction = useCallback(
    async (currentInputs: Record<string, number>) => {
      setIsPredicting(true);
      try {
        const result = await runInference(jobId, currentInputs);
        setPrediction(result);
      } catch (err: any) {
        logger.error("Inference execution failed:", err);
      } finally {
        setIsPredicting(false);
      }
    },
    [jobId],
  );

  // Handle single feature input change
  const handleFeatureChange = (featureName: string, value: number) => {
    const updatedInputs = { ...inputs, [featureName]: value };
    setInputs(updatedInputs);

    // If an active test sample was loaded, modifying a slider clears the exact sample match
    if (activeSample && activeSample.features[featureName] !== value) {
      setActiveSample(null);
    }

    if (livePredict) {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = setTimeout(() => {
        executePrediction(updatedInputs);
      }, 150);
    }
  };

  // Reset all sliders to median default values
  const handleResetToMedians = () => {
    if (!schema) return;
    const medianInputs: Record<string, number> = {};
    schema.features.forEach((feat) => {
      medianInputs[feat.name] = feat.default_value;
    });
    setInputs(medianInputs);
    setActiveSample(null);
    executePrediction(medianInputs);
  };

  // Load a random sample from holdout test split
  const handleLoadRandomSample = () => {
    if (!schema || !schema.sample_records || schema.sample_records.length === 0) return;
    const randomIndex = Math.floor(Math.random() * schema.sample_records.length);
    const sample = schema.sample_records[randomIndex];

    const sampleInputs: Record<string, number> = { ...sample.features };
    // Ensure all schema features have a value
    schema.features.forEach((feat) => {
      if (!(feat.name in sampleInputs)) {
        sampleInputs[feat.name] = feat.default_value;
      }
    });

    setInputs(sampleInputs);
    setActiveSample(sample);
    executePrediction(sampleInputs);
  };

  if (isLoadingSchema) {
    return (
      <Card className="w-full">
        <CardContent className="flex flex-col items-center justify-center py-16 space-y-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="text-sm text-gray-500 font-medium">
            Loading interactive inference playground...
          </p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full border-red-200 bg-red-50">
        <CardContent className="pt-6">
          <div className="flex items-center gap-2 text-red-800">
            <AlertTriangle className="h-5 w-5 flex-shrink-0" />
            <p className="text-sm font-medium">{error}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!schema) return null;

  const isClassification = schema.problem_type === "classification";
  const confidenceScore = prediction?.confidence ?? 0;
  const confidencePercent = Math.round(confidenceScore * 100);

  // Determine badge color based on confidence
  let confidenceColor = "bg-green-100 text-green-800 border-green-300";
  let progressColorClass = "bg-green-600";
  if (confidencePercent < 50) {
    confidenceColor = "bg-red-100 text-red-800 border-red-300";
    progressColorClass = "bg-red-600";
  } else if (confidencePercent < 80) {
    confidenceColor = "bg-yellow-100 text-yellow-800 border-yellow-300";
    progressColorClass = "bg-yellow-600";
  }

  return (
    <div className="space-y-6">
      {/* Playground Header Card */}
      <Card className="border shadow-sm">
        <CardHeader className="pb-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <CardTitle className="text-lg font-semibold flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-indigo-500" />
                Interactive Inference Playground & Sensitivity Sandbox
              </CardTitle>
              <CardDescription className="text-xs text-gray-500 mt-1">
                Test live model predictions with custom inputs, tweak feature sliders to inspect
                decision boundary sensitivity, or evaluate against holdout test samples.
              </CardDescription>
            </div>

            {/* Quick Action Buttons */}
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleLoadRandomSample}
                disabled={!schema.sample_records || schema.sample_records.length === 0}
                className="text-xs h-8 flex items-center gap-1.5"
                title="Load a random test set sample with ground truth"
              >
                <Dices className="h-3.5 w-3.5 text-indigo-600" />
                Random Test Sample
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleResetToMedians}
                className="text-xs h-8 flex items-center gap-1.5"
                title="Reset all feature sliders to dataset median values"
              >
                <RefreshCw className="h-3.5 w-3.5 text-gray-500" />
                Reset Medians
              </Button>

              <label className="flex items-center gap-1.5 cursor-pointer text-xs font-medium text-gray-700 bg-gray-50 border rounded px-2.5 py-1.5">
                <input
                  type="checkbox"
                  checked={livePredict}
                  onChange={(e) => setLivePredict(e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500 h-3.5 w-3.5"
                />
                <span className="flex items-center gap-1">
                  <Zap className={`h-3 w-3 ${livePredict ? "text-amber-500" : "text-gray-400"}`} />
                  Live Predict
                </span>
              </label>

              {!livePredict && (
                <Button
                  size="sm"
                  onClick={() => executePrediction(inputs)}
                  disabled={isPredicting}
                  className="text-xs h-8"
                >
                  Predict
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Main 2-Column Responsive Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Feature Controls (7 cols) */}
        <Card className="lg:col-span-7 border shadow-sm">
          <CardHeader className="pb-3 border-b bg-gray-50/50">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-semibold text-gray-800">
                Input Features ({schema.features.length})
              </CardTitle>
              <span className="text-xs text-gray-500">Target: {schema.target_field}</span>
            </div>
          </CardHeader>

          <CardContent className="pt-4 space-y-4 max-h-[580px] overflow-y-auto pr-3">
            {schema.features.map((feat) => {
              const currentValue = inputs[feat.name] ?? feat.default_value;
              const minVal = feat.min ?? 0;
              const maxVal = feat.max ?? 100;
              const stepVal = feat.step ?? 0.1;

              return (
                <div
                  key={feat.name}
                  className="space-y-1.5 p-2.5 rounded-lg border bg-white hover:bg-gray-50/60 transition-colors"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span
                      className="text-xs font-semibold text-gray-700 truncate"
                      title={feat.name}
                    >
                      {feat.name}
                    </span>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min={minVal}
                        max={maxVal}
                        step={stepVal}
                        value={currentValue}
                        onChange={(e) => {
                          const parsed = parseFloat(e.target.value);
                          if (!isNaN(parsed)) {
                            handleFeatureChange(feat.name, parsed);
                          }
                        }}
                        className="w-20 px-2 py-0.5 text-xs text-right border rounded bg-gray-50 focus:bg-white focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
                      />
                    </div>
                  </div>

                  {/* Dual Slider Sync */}
                  <div className="pt-1">
                    <Slider
                      min={minVal}
                      max={maxVal}
                      step={stepVal}
                      value={[currentValue]}
                      onValueChange={(val) => handleFeatureChange(feat.name, val[0])}
                    />
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-gray-400">
                    <span>Min: {minVal}</span>
                    <span>Median: {feat.median}</span>
                    <span>Max: {maxVal}</span>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        {/* Right Column: Prediction Results & Probabilities (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card className="border shadow-sm">
            <CardHeader className="pb-3 border-b bg-gray-50/50">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold text-gray-800">
                  Live Prediction Output
                </CardTitle>
                {prediction && (
                  <Badge
                    variant="outline"
                    className="text-[10px] flex items-center gap-1 font-mono"
                  >
                    <Zap className="h-3 w-3 text-amber-500" />
                    {prediction.latency_ms} ms
                  </Badge>
                )}
              </div>
            </CardHeader>

            <CardContent className="pt-4 space-y-4">
              {/* Ground Truth Banner if sample loaded */}
              {activeSample && prediction && (
                <div
                  className={`p-2.5 rounded-md border text-xs flex items-center gap-2 ${
                    isClassification
                      ? prediction.predicted_label === activeSample.actual_class_name
                        ? "bg-green-50 border-green-200 text-green-800"
                        : "bg-amber-50 border-amber-200 text-amber-800"
                      : "bg-blue-50 border-blue-200 text-blue-800"
                  }`}
                >
                  {isClassification ? (
                    prediction.predicted_label === activeSample.actual_class_name ? (
                      <>
                        <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0" />
                        <div>
                          <strong>Sample #{activeSample.index + 1} Match:</strong> Predicted label
                          matches actual ground truth ({activeSample.actual_class_name}).
                        </div>
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="h-4 w-4 text-amber-600 flex-shrink-0" />
                        <div>
                          <strong>Sample #{activeSample.index + 1} Mismatch:</strong> Predicted{" "}
                          <strong>{prediction.predicted_label}</strong>, but actual ground truth was{" "}
                          <strong>{activeSample.actual_class_name}</strong>.
                        </div>
                      </>
                    )
                  ) : (
                    <div>
                      <strong>Sample #{activeSample.index + 1}:</strong> Predicted{" "}
                      {prediction.predicted_value} vs Ground Truth {activeSample.actual_value}
                    </div>
                  )}
                </div>
              )}

              {/* Classification Prediction Display */}
              {isClassification && prediction && (
                <div className="space-y-4">
                  {/* Hero Result Banner */}
                  <div className="p-4 rounded-xl border bg-gradient-to-br from-indigo-50/40 via-white to-blue-50/30 flex flex-col items-center justify-center text-center space-y-2">
                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                      Predicted Class
                    </span>
                    <div className="text-2xl font-bold text-gray-900 tracking-tight">
                      {prediction.predicted_label}
                    </div>

                    <div className="flex items-center gap-2 pt-1">
                      <Badge className={`text-xs px-2.5 py-0.5 font-semibold ${confidenceColor}`}>
                        {confidencePercent}% Confidence
                      </Badge>
                    </div>

                    <div className="w-full max-w-[200px] pt-1">
                      <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${progressColorClass} transition-all duration-300`}
                          style={{ width: `${confidencePercent}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Class Probabilities Distribution */}
                  {prediction.probabilities && prediction.probabilities.length > 0 && (
                    <div className="space-y-2.5 pt-2">
                      <span className="text-xs font-semibold text-gray-700 block">
                        Class Probability Distribution
                      </span>

                      <div className="space-y-2">
                        {prediction.probabilities.map((prob) => {
                          const isTop = prob.class_index === prediction.predicted_class;
                          const pct = Math.round(prob.probability * 100);

                          return (
                            <div key={prob.class_index} className="space-y-1">
                              <div className="flex items-center justify-between text-xs">
                                <span
                                  className={`truncate ${isTop ? "font-bold text-gray-900" : "text-gray-600"}`}
                                >
                                  {prob.class_name}
                                </span>
                                <span
                                  className={`font-mono text-xs ${isTop ? "font-bold text-blue-600" : "text-gray-500"}`}
                                >
                                  {(prob.probability * 100).toFixed(1)}%
                                </span>
                              </div>

                              <div className="h-2 w-full bg-gray-100 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${isTop ? "bg-blue-600" : "bg-gray-400"} transition-all duration-200`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Regression Prediction Display */}
              {!isClassification && prediction && (
                <div className="p-6 rounded-xl border bg-gradient-to-br from-blue-50/40 via-white to-indigo-50/30 flex flex-col items-center justify-center text-center space-y-2">
                  <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Predicted Value
                  </span>
                  <div className="text-3xl font-bold text-blue-600 font-mono">
                    {prediction.predicted_value}
                  </div>
                  <span className="text-xs text-gray-400">Target Field: {schema.target_field}</span>
                </div>
              )}

              {/* What-If Tip */}
              <div className="p-3 bg-gray-50 rounded-lg border text-[11px] text-gray-500 flex items-start gap-2">
                <HelpCircle className="h-4 w-4 text-gray-400 flex-shrink-0 mt-0.5" />
                <span>
                  <strong>What-If Analysis:</strong> Tweak feature sliders on the left to observe
                  how changes in feature values influence prediction certainty and decision boundary
                  boundaries.
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
