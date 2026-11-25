import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowRight,
  BarChart2,
  GitCompare,
  Layers,
  PackageCheck,
  RefreshCw,
} from "lucide-react";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import AnalysisDashboard from "@/components/analysis/AnalysisDashboard";
import { ComparisonDashboard } from "@/components/comparison";
import ExportPanel from "@/components/export/ExportPanel";
import ReproducibilityDetails from "@/components/training/ReproducibilityDetails";
import { getTrainingJobs } from "@/services/comparisonService";
import { getTrainingHistory } from "@/services/ModelServices";
import logger from "@/shared/logger";

const STATUS_BADGE_VARIANTS = {
  completed: "default",
  running: "secondary",
  pending: "outline",
  failed: "destructive",
  cancelled: "destructive",
};

/**
 * Post-training workspace hub for model diagnostics, artifact exports,
 * reproducibility inspection, and multi-run comparison.
 */
export default function Analysis() {
  const { projectId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [activeTab, setActiveTab] = useState("diagnostics");

  const [loadingModels, setLoadingModels] = useState(true);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [error, setError] = useState(null);

  const searchParamsRef = useRef(searchParams);
  searchParamsRef.current = searchParams;
  const initialModelsLoadedRef = useRef(false);

  const loadModels = useCallback(async () => {
    if (!initialModelsLoadedRef.current) {
      setLoadingModels(true);
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
      logger.error("Failed to load models for analysis:", err);
      setError("Failed to load project models.");
      setModels([]);
    } finally {
      initialModelsLoadedRef.current = true;
      setLoadingModels(false);
    }
  }, [projectId]);

  const loadJobs = useCallback(async (modelName) => {
    if (!modelName) {
      setJobs([]);
      setSelectedJobId("");
      return;
    }
    setLoadingJobs(true);
    setError(null);
    try {
      const resp = await getTrainingJobs(modelName);
      const jobList = resp.data?.data || [];
      setJobs(jobList);

      const paramJobId = searchParamsRef.current.get("jobId");
      if (paramJobId && jobList.some((j) => j.job_id === paramJobId)) {
        setSelectedJobId(paramJobId);
      } else {
        const completedJob = jobList.find((j) => j.status === "completed");
        setSelectedJobId(completedJob?.job_id || jobList[0]?.job_id || "");
      }
    } catch (err) {
      logger.error("Failed to load training runs:", err);
      setError(
        err.response?.data?.detail ||
          err.response?.data?.message ||
          "Failed to load training runs.",
      );
      setJobs([]);
      setSelectedJobId("");
    } finally {
      setLoadingJobs(false);
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  useEffect(() => {
    if (selectedModel) {
      loadJobs(selectedModel);
    } else {
      setJobs([]);
      setSelectedJobId("");
    }
  }, [selectedModel, loadJobs]);

  const selectedJob = useMemo(
    () => jobs.find((j) => j.job_id === selectedJobId) || null,
    [jobs, selectedJobId],
  );

  const handleModelChange = (modelName) => {
    setSelectedModel(modelName);
    setSelectedJobId("");
    const nextParams = new URLSearchParams(searchParams);
    nextParams.set("model", modelName);
    nextParams.delete("jobId");
    setSearchParams(nextParams, { replace: true });
  };

  const handleJobChange = (jobId) => {
    setSelectedJobId(jobId);
    const nextParams = new URLSearchParams(searchParams);
    if (selectedModel) nextParams.set("model", selectedModel);
    nextParams.set("jobId", jobId);
    setSearchParams(nextParams, { replace: true });
  };

  if (loadingModels) {
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
              <BarChart2 className="h-5 w-5" />
              Post-Training Analysis &amp; Run Comparison
            </CardTitle>
            <CardDescription>
              Inspect confusion matrices, diagnostic curves, feature importance, artifacts,
              reproducibility snapshots, and cross-run comparisons.
            </CardDescription>
          </div>
          {selectedModel && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadJobs(selectedModel)}
              disabled={loadingJobs}
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${loadingJobs ? "animate-spin" : ""}`} />
              Refresh Runs
            </Button>
          )}
        </CardHeader>
        <CardContent>
          {models.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed py-10 text-center">
              <Layers className="h-8 w-8 text-muted-foreground" />
              <div className="space-y-1">
                <p className="text-sm font-medium">No models found in this project</p>
                <p className="text-xs text-muted-foreground">
                  Create and train a model first to explore post-training diagnostics and
                  comparisons.
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
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Model:</span>
                <Select value={selectedModel} onValueChange={handleModelChange}>
                  <SelectTrigger className="w-56" aria-label="Select model for analysis">
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

              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Training Run:</span>
                <Select
                  value={selectedJobId}
                  onValueChange={handleJobChange}
                  disabled={jobs.length === 0 || loadingJobs}
                >
                  <SelectTrigger className="w-80" aria-label="Select training run">
                    <SelectValue
                      placeholder={
                        loadingJobs
                          ? "Loading runs..."
                          : jobs.length === 0
                            ? "No runs recorded"
                            : "Select a run"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {jobs.map((job) => (
                      <SelectItem key={job.job_id} value={job.job_id}>
                        Run {job.job_id.slice(0, 8)} ({job.status})
                        {job.started_at
                          ? ` — ${new Date(job.started_at).toLocaleDateString()}`
                          : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {selectedJob && (
                <Badge
                  variant={STATUS_BADGE_VARIANTS[selectedJob.status] || "outline"}
                  className="capitalize"
                >
                  {selectedJob.status}
                </Badge>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {selectedModel && (
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="diagnostics" className="gap-1.5">
              <BarChart2 className="h-4 w-4" />
              Run Diagnostics
            </TabsTrigger>
            <TabsTrigger value="export" className="gap-1.5">
              <PackageCheck className="h-4 w-4" />
              Export &amp; Reproducibility
            </TabsTrigger>
            <TabsTrigger value="compare" className="gap-1.5">
              <GitCompare className="h-4 w-4" />
              Compare Runs
            </TabsTrigger>
          </TabsList>

          <TabsContent value="diagnostics" className="mt-4">
            {!selectedJob ? (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center gap-3 py-10 text-center">
                  <p className="text-sm font-medium">
                    No training runs found for <strong>{selectedModel}</strong>
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Launch a training run or hyperparameter sweep to unlock confusion matrices,
                    diagnostic curves, and predictions.
                  </p>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate(`/workspace/${projectId}/training`)}
                    >
                      Go to Training
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        navigate(
                          `/workspace/${projectId}/tuning?model=${encodeURIComponent(selectedModel)}`,
                        )
                      }
                    >
                      Go to Tuning
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : selectedJob.status !== "completed" ? (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Run not completed</AlertTitle>
                <AlertDescription>
                  Selected run <code>{selectedJob.job_id.slice(0, 8)}</code> has status{" "}
                  <strong>{selectedJob.status}</strong>. Diagnostics are available once a training
                  run completes.
                </AlertDescription>
              </Alert>
            ) : (
              <AnalysisDashboard key={selectedJob.job_id} jobId={selectedJob.job_id} />
            )}
          </TabsContent>

          <TabsContent value="export" className="mt-4 space-y-6">
            {!selectedJob ? (
              <Card className="border-dashed">
                <CardContent className="py-10 text-center text-sm text-muted-foreground">
                  Select a training run to export artifacts or inspect reproducibility metadata.
                </CardContent>
              </Card>
            ) : (
              <>
                {selectedJob.status === "completed" ? (
                  <ExportPanel
                    key={`export-${selectedJob.job_id}`}
                    jobId={selectedJob.job_id}
                    modelName={selectedModel}
                  />
                ) : (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Model export unavailable</AlertTitle>
                    <AlertDescription>
                      Exports require a completed training job (current status:{" "}
                      <strong>{selectedJob.status}</strong>).
                    </AlertDescription>
                  </Alert>
                )}
                <ReproducibilityDetails
                  key={`repro-${selectedJob.job_id}`}
                  jobId={selectedJob.job_id}
                />
              </>
            )}
          </TabsContent>

          <TabsContent value="compare" className="mt-4">
            <ComparisonDashboard key={selectedModel} modelName={selectedModel} />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
