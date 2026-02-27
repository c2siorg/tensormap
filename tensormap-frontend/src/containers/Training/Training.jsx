import { useState, useEffect, useRef, useCallback } from "react";
import { useParams } from "react-router-dom";
import io from "socket.io-client";
import { useRecoilState } from "recoil";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import * as urls from "../../constants/Urls";
import * as strings from "../../constants/Strings";
import logger from "../../shared/logger";
import Result from "../../components/ResultPanel/Result/Result";
import {
  download_code,
  runModel,
  getAllModels,
  updateTrainingConfig,
  deleteModel,
} from "../../services/ModelServices";
import { getAllFiles } from "../../services/FileServices";
import { models as modelListAtom } from "../../shared/atoms";
import { Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

const optimizerOptions = [
  { key: "opt_1", label: "Adam", value: "adam" },
  { key: "opt_2", label: "SGD", value: "sgd" },
];

const metricOptions = [
  { key: "acc_1", label: "Accuracy", value: "accuracy" },
  { key: "acc_2", label: "MSE", value: "mse" },
];

const problemTypeOptions = [
  { key: "prob_type_1", label: "Multi class classification", value: "1" },
  { key: "prob_type_2", label: "Linear Regression", value: "2" },
];

export default function Training() {
  const { projectId } = useParams();
  const [modelList, setModelList] = useRecoilState(modelListAtom);

  const [selectedModel, setSelectedModel] = useState(null);
  const [resultValues, setResultValues] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const socketRef = useRef(null);
  const timeoutRef = useRef(null);

  // Training config state
  const [fileList, setFileList] = useState([]);
  const [fileDetails, setFileDetails] = useState([]);
  const [fieldsList, setFieldsList] = useState([]);
  const [configSaved, setConfigSaved] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    file_id: "",
    target_field: "",
    problem_type_id: "",
    optimizer: "",
    metric: "",
    epochs: "",
    batch_size: "",
    training_split: "",
  });

  useEffect(() => {
    const socket = io(urls.WS_DL_RESULTS, {
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 10000,
    });
    socketRef.current = socket;

    const dlResultListener = (resp) => {
      clearTimeout(timeoutRef.current);
      if (resp.message && resp.message.includes("Starting")) {
        setResultValues([]);
        setIsLoading(true);
      } else if (resp.message && resp.message.includes("Finish")) {
        setIsLoading(false);
      } else {
        setResultValues((prev) => {
          let newValues = [...prev];
          newValues[parseInt(resp.test)] = resp.message;
          return newValues;
        });
      }
    };

    socket.on(strings.DL_RESULT_LISTENER, dlResultListener);

    socket.on("connect_error", (err) => {
      logger.warn("Socket connection error:", err);
      setConnectionError("Lost connection to server");
    });

    socket.on("connect", () => {
      setConnectionError(null);
    });

    socket.on("disconnect", (reason) => {
      if (reason === "io server disconnect") {
        socket.connect();
      }
    });

    getAllModels(projectId)
      .then((response) => {
        const models = response.map((file, index) => ({
          label: file.name + strings.MODEL_EXTENSION,
          value: file.name,
          id: file.id,
          key: index,
        }));
        setModelList(models);
      })
      .catch((error) => {
        logger.error("Error loading models:", error);
        setModelList([]);
      });

    getAllFiles(projectId)
      .then((response) => {
        const files = response.map((file, index) => ({
          label: `${file.file_name}.${file.file_type}`,
          value: String(file.file_id),
          key: index,
        }));
        setFileList(files);
        setFileDetails(response);
      })
      .catch((error) => {
        logger.error("Error loading files:", error);
      });

    return () => {
      clearTimeout(timeoutRef.current);
      socket.off(strings.DL_RESULT_LISTENER, dlResultListener);
      socket.disconnect();
    };
  }, [projectId, setModelList]);

  const handleFileSelect = useCallback(
    (value) => {
      const selected = fileDetails.find((item) => item.file_id === value);
      if (selected) {
        const fields = selected.fields.map((item, index) => ({
          label: item,
          value: item,
          key: index,
        }));
        setFieldsList(fields);
      }
      setTrainingConfig((prev) => ({ ...prev, file_id: value, target_field: "" }));
    },
    [fileDetails],
  );

  const handleModelSelect = (value) => {
    setSelectedModel(value);
    setConfigSaved(false);
  };

  const handleSaveConfig = () => {
    if (!selectedModel) return;

    const data = {
      model_name: selectedModel,
      file_id: trainingConfig.file_id,
      target_field: trainingConfig.target_field || null,
      training_split: Number(trainingConfig.training_split) * 100,
      problem_type_id: Number(trainingConfig.problem_type_id),
      optimizer: trainingConfig.optimizer,
      metric: trainingConfig.metric,
      epochs: Number(trainingConfig.epochs),
      project_id: projectId || null,
    };

    updateTrainingConfig(data)
      .then((resp) => {
        if (resp.success) {
          setConfigSaved(true);
        } else {
          logger.error("Failed to save training config:", resp.message);
        }
      })
      .catch((error) => {
        logger.error("Error saving training config:", error);
      });
  };

  const isConfigValid =
    trainingConfig.file_id &&
    trainingConfig.problem_type_id &&
    trainingConfig.optimizer &&
    trainingConfig.metric &&
    trainingConfig.epochs &&
    trainingConfig.training_split;

  const handleDownload = () => {
    if (selectedModel) {
      download_code(selectedModel, projectId).catch((error) => logger.error(error));
    }
  };

  const handleRun = () => {
    if (!selectedModel) return;
    if (!socketRef.current?.connected) {
      setResultValues([
        "Cannot start training: not connected to server. Please wait and try again.",
      ]);
      return;
    }
    setResultValues([]);
    setIsLoading(true);
    timeoutRef.current = setTimeout(() => {
      setIsLoading(false);
      setResultValues(["Training timed out. The model may still be running on the server."]);
    }, 300000);
    runModel(selectedModel, projectId)
      .then(() => { })
      .catch((error) => {
        clearTimeout(timeoutRef.current);
        logger.error(error.response?.data);
        setResultValues([error.response?.data?.message ?? "An error occurred"]);
        setIsLoading(false);
      });
  };

  const handleClear = () => {
    setResultValues([]);
    setIsLoading(false);
  };

  const handleDeleteModel = () => {
    const modelToDelete = modelList.find((m) => m.value === selectedModel);
    if (!modelToDelete) return;

    setIsDeleting(true);
    deleteModel(modelToDelete.id, projectId)
      .then((resp) => {
        setIsDeleting(false);
        setIsDeleteDialogOpen(false);
        if (resp.success) {
          setModelList((prev) => prev.filter((m) => m.id !== modelToDelete.id));
          setSelectedModel(null);
          setConfigSaved(false);
        } else {
          logger.error("Error deleting model:", resp.message);
        }
      })
      .catch((error) => {
        setIsDeleting(false);
        setIsDeleteDialogOpen(false);
        logger.error("Error deleting model:", error);
      });
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Training</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            <Select onValueChange={handleModelSelect}>
              <SelectTrigger className="w-64">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {modelList.map((model) => (
                  <SelectItem key={model.key} value={model.value}>
                    {model.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              variant="destructive"
              size="icon"
              disabled={!selectedModel}
              onClick={() => setIsDeleteDialogOpen(true)}
              title="Delete Model"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
            <Button
              onClick={handleDownload}
              disabled={!selectedModel || !configSaved}
              variant="outline"
            >
              Download Code
            </Button>
            <Button onClick={handleRun} disabled={!selectedModel || !configSaved || isLoading}>
              {isLoading ? "Running..." : "Run Model"}
            </Button>
            <Button
              onClick={handleClear}
              disabled={resultValues.length === 0 && !isLoading}
              variant="secondary"
            >
              Clear
            </Button>
          </div>
        </CardContent>
      </Card>

      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Model</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the model &quot;{selectedModel}&quot;? This action cannot be undone and will permanently remove the model and its configurations.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)} disabled={isDeleting}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDeleteModel} disabled={isDeleting}>
              {isDeleting ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {selectedModel && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Training Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="space-y-1">
                <Label>Dataset File</Label>
                <Select onValueChange={handleFileSelect}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a file" />
                  </SelectTrigger>
                  <SelectContent>
                    {fileList.map((f) => (
                      <SelectItem key={f.key} value={f.value}>
                        {f.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label>Problem Type</Label>
                <Select
                  onValueChange={(v) =>
                    setTrainingConfig((prev) => ({ ...prev, problem_type_id: v }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select problem type" />
                  </SelectTrigger>
                  <SelectContent>
                    {problemTypeOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label>Target Field</Label>
                <Select
                  onValueChange={(v) => setTrainingConfig((prev) => ({ ...prev, target_field: v }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select target field" />
                  </SelectTrigger>
                  <SelectContent>
                    {fieldsList.map((f) => (
                      <SelectItem key={f.key} value={f.value}>
                        {f.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label>Optimizer</Label>
                <Select
                  onValueChange={(v) => setTrainingConfig((prev) => ({ ...prev, optimizer: v }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select optimizer" />
                  </SelectTrigger>
                  <SelectContent>
                    {optimizerOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label>Result Metrics</Label>
                <Select
                  onValueChange={(v) => setTrainingConfig((prev) => ({ ...prev, metric: v }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select metric" />
                  </SelectTrigger>
                  <SelectContent>
                    {metricOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1">
                <Label>Epochs</Label>
                <Input
                  type="number"
                  min="1"
                  placeholder="Number of epochs"
                  value={trainingConfig.epochs}
                  onChange={(e) =>
                    setTrainingConfig((prev) => ({ ...prev, epochs: e.target.value }))
                  }
                />
              </div>

              <div className="space-y-1">
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  min="1"
                  placeholder="Batch size"
                  value={trainingConfig.batch_size}
                  onChange={(e) =>
                    setTrainingConfig((prev) => ({ ...prev, batch_size: e.target.value }))
                  }
                />
              </div>

              <div className="space-y-1">
                <Label>Train:Test Ratio</Label>
                <Input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  placeholder="e.g. 0.8"
                  value={trainingConfig.training_split}
                  onChange={(e) =>
                    setTrainingConfig((prev) => ({ ...prev, training_split: e.target.value }))
                  }
                />
              </div>
            </div>

            <Button className="mt-4" onClick={handleSaveConfig} disabled={!isConfigValid}>
              {configSaved ? "Configuration Saved" : "Save Configuration"}
            </Button>
          </CardContent>
        </Card>
      )}

      {connectionError && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {connectionError}
        </div>
      )}

      {isLoading && resultValues.length === 0 && (
        <Card>
          <CardContent className="flex items-center justify-center py-16 text-muted-foreground">
            Model is running, please wait...
          </CardContent>
        </Card>
      )}

      {resultValues.length > 0 && (
        <div className="space-y-2">
          {resultValues.map((item, index) => (
            <Result result={item} key={index} />
          ))}
        </div>
      )}
    </div>
  );
}
