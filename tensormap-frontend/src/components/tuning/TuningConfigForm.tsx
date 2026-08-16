/**
 * Tuning Configuration Form Component
 * Allows users to configure hyperparameter search strategy and parameters.
 * @module
 */

import { useState, useMemo, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import { TuningSearchSpace } from "@/services/tuningService";
import { AlertTriangle, Loader2 } from "lucide-react";

const OPTIMIZERS = ["adam", "sgd", "rmsprop"];
const BATCH_SIZES = [16, 32, 64, 128];
const EPOCHS_OPTIONS = [20, 50, 100];
const LEARNING_RATE_MIN = 1e-5;
const LEARNING_RATE_MAX = 1e-2;

const METRICS = [
  { value: "val_accuracy", direction: "maximize" },
  { value: "val_loss", direction: "minimize" },
  { value: "accuracy", direction: "maximize" },
  { value: "loss", direction: "minimize" },
];

interface TuningConfigFormProps {
  modelName: string;
  onStart: (config: {
    strategy: "grid" | "random";
    search_space: TuningSearchSpace;
    max_trials: number;
    metric: string;
    direction: "maximize" | "minimize";
    early_stop_threshold: number | null;
  }) => void;
  estimateTime?: (space: TuningSearchSpace, strategy: string, maxTrials: number) => number;
}

export default function TuningConfigForm({
  modelName,
  onStart,
  estimateTime,
}: TuningConfigFormProps) {
  const [strategy, setStrategy] = useState<"grid" | "random">("random");
  const [maxTrials, setMaxTrials] = useState(10);

  // Search space state
  const [selectedOptimizers, setSelectedOptimizers] = useState(["adam", "sgd"]);
  const [lrLogMin, setLrLogMin] = useState(-4); // log10(1e-4) = -4
  const [lrLogMax, setLrLogMax] = useState(-3); // log10(1e-3) = -3
  const [selectedBatchSizes, setSelectedBatchSizes] = useState([32, 64]);
  const [selectedEpochs, setSelectedEpochs] = useState([50]);

  const [metric, setMetric] = useState("val_accuracy");
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [enableEarlyStop, setEnableEarlyStop] = useState(false);
  const [earlyStopThreshold, setEarlyStopThreshold] = useState(0.95);

  // Auto-set direction based on metric
  useEffect(() => {
    const metricInfo = METRICS.find((m) => m.value === metric);
    if (metricInfo) {
      setDirection(metricInfo.direction);
    }
  }, [metric]);

  // Build search space
  const searchSpace: TuningSearchSpace = useMemo(() => {
    return {
      optimizer: selectedOptimizers,
      learning_rate: {
        type: "log_uniform",
        min: Math.pow(10, lrLogMin),
        max: Math.pow(10, lrLogMax),
      },
      batch_size: selectedBatchSizes,
      epochs: selectedEpochs,
    };
  }, [selectedOptimizers, lrLogMin, lrLogMax, selectedBatchSizes, selectedEpochs]);

  // Calculate grid combinations count
  const gridCombinations = useMemo(() => {
    if (strategy !== "grid") return 0;
    return (
      selectedOptimizers.length *
      selectedBatchSizes.length *
      selectedEpochs.length *
      2 // Learning rate range (simplified estimate)
    );
  }, [strategy, selectedOptimizers, selectedBatchSizes, selectedEpochs]);

  // Estimate time
  const estimatedSeconds = useMemo(() => {
    if (estimateTime) {
      const trials = strategy === "grid" ? gridCombinations : maxTrials;
      return estimateTime(searchSpace, strategy, trials);
    }
    // Fallback: 2 minutes per trial
    const trials = strategy === "grid" ? gridCombinations : maxTrials;
    return trials * 120;
  }, [estimateTime, searchSpace, strategy, maxTrials, gridCombinations]);

  const estimatedMinutes = Math.ceil(estimatedSeconds / 60);
  const showWarning = estimatedMinutes > 30;
  const gridError = strategy === "grid" && gridCombinations > 50;

  const handleStart = () => {
    if (gridError) return;

    onStart({
      strategy,
      search_space: searchSpace,
      max_trials: maxTrials,
      metric,
      direction,
      early_stop_threshold: enableEarlyStop ? earlyStopThreshold : null,
    });
  };

  const toggleOptimizer = (opt: string) => {
    setSelectedOptimizers((prev) =>
      prev.includes(opt) ? prev.filter((o) => o !== opt) : [...prev, opt],
    );
  };

  const toggleBatchSize = (size: number) => {
    setSelectedBatchSizes((prev) =>
      prev.includes(size) ? prev.filter((s) => s !== size) : [...prev, size],
    );
  };

  const toggleEpochs = (epoch: number) => {
    setSelectedEpochs((prev) =>
      prev.includes(epoch) ? prev.filter((e) => e !== epoch) : [...prev, epoch],
    );
  };

  const isValid =
    selectedOptimizers.length > 0 &&
    selectedBatchSizes.length > 0 &&
    selectedEpochs.length > 0 &&
    !gridError;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Hyperparameter Tuning</CardTitle>
        <CardDescription>
          Configure hyperparameter search for model: <strong>{modelName}</strong>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Strategy Selection */}
        <div className="space-y-2">
          <Label>Search Strategy</Label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="strategy"
                value="random"
                checked={strategy === "random"}
                onChange={(e) => setStrategy(e.target.value as "random")}
                className="cursor-pointer"
              />
              <span>Random Search</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="strategy"
                value="grid"
                checked={strategy === "grid"}
                onChange={(e) => setStrategy(e.target.value as "grid")}
                className="cursor-pointer"
              />
              <span>Grid Search</span>
            </label>
          </div>
        </div>

        {/* Max Trials (Random Search Only) */}
        {strategy === "random" && (
          <div className="space-y-2">
            <Label htmlFor="maxTrials">Max Trials</Label>
            <Input
              id="maxTrials"
              type="number"
              min={1}
              max={50}
              value={maxTrials}
              onChange={(e) => setMaxTrials(parseInt(e.target.value) || 1)}
              className="max-w-xs"
            />
            <p className="text-sm text-gray-500">Number of random combinations to try (1-50)</p>
          </div>
        )}

        {/* Search Space Section */}
        <div className="space-y-4 border-t pt-4">
          <h3 className="font-semibold">Search Space</h3>

          {/* Optimizer */}
          <div className="space-y-2">
            <Label>Optimizer</Label>
            <div className="flex gap-4">
              {OPTIMIZERS.map((opt) => (
                <label key={opt} className="flex items-center gap-2 cursor-pointer">
                  <Checkbox
                    checked={selectedOptimizers.includes(opt)}
                    onCheckedChange={() => toggleOptimizer(opt)}
                  />
                  <span className="capitalize">{opt}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Learning Rate */}
          <div className="space-y-2">
            <Label>Learning Rate (Log Scale)</Label>
            <div className="space-y-4">
              <div className="px-4">
                <Slider
                  min={-5}
                  max={-2}
                  step={0.5}
                  value={[lrLogMin, lrLogMax]}
                  onValueChange={([min, max]) => {
                    setLrLogMin(min);
                    setLrLogMax(max);
                  }}
                  className="w-full"
                />
              </div>
              <div className="flex justify-between text-sm text-gray-600">
                <span>
                  Min: {Math.pow(10, lrLogMin).toExponential(1)}
                </span>
                <span>
                  Max: {Math.pow(10, lrLogMax).toExponential(1)}
                </span>
              </div>
              <p className="text-sm text-gray-500">log_uniform distribution</p>
            </div>
          </div>

          {/* Batch Size */}
          <div className="space-y-2">
            <Label>Batch Size</Label>
            <div className="flex gap-4">
              {BATCH_SIZES.map((size) => (
                <label key={size} className="flex items-center gap-2 cursor-pointer">
                  <Checkbox
                    checked={selectedBatchSizes.includes(size)}
                    onCheckedChange={() => toggleBatchSize(size)}
                  />
                  <span>{size}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Epochs */}
          <div className="space-y-2">
            <Label>Epochs</Label>
            <div className="flex gap-4">
              {EPOCHS_OPTIONS.map((epoch) => (
                <label key={epoch} className="flex items-center gap-2 cursor-pointer">
                  <Checkbox
                    checked={selectedEpochs.includes(epoch)}
                    onCheckedChange={() => toggleEpochs(epoch)}
                  />
                  <span>{epoch}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Metric Selection */}
        <div className="space-y-2 border-t pt-4">
          <Label htmlFor="metric">Optimization Metric</Label>
          <div className="flex gap-4 items-center">
            <Select value={metric} onValueChange={setMetric}>
              <SelectTrigger id="metric" className="max-w-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {METRICS.map((m) => (
                  <SelectItem key={m.value} value={m.value}>
                    {m.value}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex gap-2">
              <span className="text-sm text-gray-600">Direction:</span>
              <span className="text-sm font-medium capitalize">{direction}</span>
            </div>
          </div>
        </div>

        {/* Early Stop */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Checkbox
              id="enableEarlyStop"
              checked={enableEarlyStop}
              onCheckedChange={(checked) => setEnableEarlyStop(checked as boolean)}
            />
            <Label htmlFor="enableEarlyStop" className="cursor-pointer">
              Enable Early Stop
            </Label>
          </div>
          {enableEarlyStop && (
            <div className="ml-6 space-y-2">
              <Label htmlFor="earlyStopThreshold">Threshold</Label>
              <Input
                id="earlyStopThreshold"
                type="number"
                step={0.01}
                min={0}
                max={1}
                value={earlyStopThreshold}
                onChange={(e) => setEarlyStopThreshold(parseFloat(e.target.value) || 0)}
                className="max-w-xs"
              />
              <p className="text-sm text-gray-500">
                Stop early if {direction === "maximize" ? "≥" : "≤"} this value
              </p>
            </div>
          )}
        </div>

        {/* ETA Estimation */}
        {isValid && (
          <div className="space-y-2 border-t pt-4">
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              <span className="text-sm font-medium">Estimated time:</span>
              <span className="text-sm">
                ~{estimatedMinutes} minutes (
                {strategy === "grid" ? gridCombinations : maxTrials} trials × ~
                {Math.ceil(estimatedSeconds / (strategy === "grid" ? gridCombinations : maxTrials))}{" "}
                sec/trial)
              </span>
            </div>
          </div>
        )}

        {/* Warnings */}
        {showWarning && !gridError && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              ⚠️ Estimated time exceeds 30 minutes. Consider reducing max_trials or narrowing the
              search space.
            </AlertDescription>
          </Alert>
        )}

        {gridError && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Grid search would generate {gridCombinations} combinations (max: 50). Please reduce
              the search space.
            </AlertDescription>
          </Alert>
        )}

        {/* Start Button */}
        <Button onClick={handleStart} disabled={!isValid} className="w-full">
          Start Hyperparameter Tuning
        </Button>
      </CardContent>
    </Card>
  );
}
