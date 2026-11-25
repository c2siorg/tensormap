import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { AlertCircle, ArrowRight, Layers, RefreshCw, SlidersHorizontal } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { TuningContainer } from "@/components/tuning";
import { getTrainingHistory } from "@/services/ModelServices";
import logger from "@/shared/logger";

/**
 * Workspace page for automated hyperparameter tuning (grid & random search).
 *
 * Lets users pick any saved model in the project, verify its baseline
 * training configuration is ready, and launch or monitor a TuningContainer
 * session without leaving the workspace shell.
 */
export default function Tuning() {
  const { projectId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const searchParamsRef = useRef(searchParams);
  searchParamsRef.current = searchParams;
  const initialLoadRef = useRef(true);

  const loadModels = useCallback(async () => {
    if (initialLoadRef.current) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError(null);
    try {
      const list = await getTrainingHistory(projectId);
      const modelItems = Array.isArray(list) ? list : [];
      setModels(modelItems);

      const paramModel = searchParamsRef.current.get("model");
      if (paramModel && modelItems.some((m) => m.model_name === paramModel)) {
        setSelectedModel(paramModel);
      } else if (modelItems.length > 0) {
        setSelectedModel((prev) =>
          prev && modelItems.some((m) => m.model_name === prev) ? prev : modelItems[0].model_name,
        );
      } else {
        setSelectedModel("");
      }
    } catch (err) {
      logger.error("Failed to load models for tuning:", err);
      setError("Failed to load saved models for this project.");
      setModels([]);
    } finally {
      initialLoadRef.current = false;
      setLoading(false);
      setRefreshing(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const activeModelInfo = useMemo(
    () => models.find((m) => m.model_name === selectedModel) || null,
    [models, selectedModel],
  );

  const hasBaselineConfig = Boolean(
    activeModelInfo &&
    activeModelInfo.epochs != null &&
    activeModelInfo.training_split != null &&
    activeModelInfo.loss &&
    activeModelInfo.optimizer,
  );

  const handleModelChange = (modelName) => {
    setSelectedModel(modelName);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("model", modelName);
    setSearchParams(nextParams, { replace: true });
  };

  if (loading) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        Loading models...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-4 space-y-0">
          <div>
            <CardTitle className="flex items-center gap-2">
              <SlidersHorizontal className="h-5 w-5" />
              Hyperparameter Tuning
            </CardTitle>
            <CardDescription>
              Run grid search or random search sweeps across optimizers, learning rates, batch
              sizes, and epochs.
            </CardDescription>
          </div>
          <Button variant="outline" size="sm" onClick={loadModels} disabled={refreshing}>
            <RefreshCw className={`mr-2 h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
            Refresh Models
          </Button>
        </CardHeader>
        <CardContent>
          {models.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed py-10 text-center">
              <Layers className="h-8 w-8 text-muted-foreground" />
              <div className="space-y-1">
                <p className="text-sm font-medium">No saved models yet</p>
                <p className="text-xs text-muted-foreground">
                  Design and save a model architecture first, then configure its dataset and target
                  before tuning.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate(`/workspace/${projectId}/models`)}
              >
                Go to Models Canvas
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium">Model:</span>
                <Select value={selectedModel} onValueChange={handleModelChange}>
                  <SelectTrigger className="w-64" aria-label="Select model to tune">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((m) => (
                      <SelectItem key={m.id} value={m.model_name}>
                        {m.model_name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {activeModelInfo && (
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <Badge variant="secondary" className="capitalize">
                    Optimizer: {activeModelInfo.optimizer || "unset"}
                  </Badge>
                  <Badge variant="secondary">
                    Batch size: {activeModelInfo.batch_size ?? "unset"}
                  </Badge>
                  <Badge variant="secondary">Epochs: {activeModelInfo.epochs ?? "unset"}</Badge>
                  {activeModelInfo.loss && <Badge variant="outline">{activeModelInfo.loss}</Badge>}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {selectedModel && !hasBaselineConfig && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Training configuration required</AlertTitle>
          <AlertDescription className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <span>
              Set baseline training parameters (optimizer, loss function, epochs, train/validation
              split, and linked dataset) for <strong>{selectedModel}</strong> on the Training page
              before starting hyperparameter tuning.
            </span>
            <Button
              size="sm"
              variant="outline"
              onClick={() => navigate(`/workspace/${projectId}/training`)}
            >
              Configure Training
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {selectedModel && hasBaselineConfig && (
        <TuningContainer key={selectedModel} modelName={selectedModel} />
      )}
    </div>
  );
}
