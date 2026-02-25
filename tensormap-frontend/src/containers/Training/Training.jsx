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
} from "../../services/ModelServices";
import { getAllFiles } from "../../services/FileServices";
import { models as modelListAtom } from "../../shared/atoms";

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
  // Validation state
  const [validationErrors, setValidationErrors] = useState({
    model: "",
    file_id: "",
    problem_type_id: "",
    optimizer: "",
    metric: "",
    target_field: "",
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
          label: file + strings.MODEL_EXTENSION,
          value: file,
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

  // Validation functions
  const validateEpochs = (value) => {
    if (!value || value.trim() === "") {
      return "Epochs is required";
    }
    const trimmed = value.trim();
    if (!/^\d+$/.test(trimmed)) {
      return "Epochs must be a positive integer";
    }
    const num = Number(trimmed);
    if (num <= 0) {
      return "Epochs must be a positive integer";
    }
    return "";
  };

  const validateBatchSize = (value) => {
    if (!value || value.trim() === "") {
      return "Batch size is required";
    }
    const trimmed = value.trim();
    if (!/^\d+$/.test(trimmed)) {
      return "Batch size must be a positive integer";
    }
    const num = Number(trimmed);
    if (num <= 0) {
      return "Batch size must be a positive integer";
    }
    return "";
  };

  const validateTrainingSplit = (value) => {
    const num = parseFloat(value);
    if (!value || value.trim() === "") {
      return "Training split is required";
    }
    if (isNaN(num) || num <= 0 || num >= 1) {
      return "Training split must be between 0 and 1 (exclusive)";
    }
    return "";
  };

  const validateModel = (value) => {
    if (!value) {
      return "A model must be selected";
    }
    return "";
  };

  const validateFile = (value) => {
    if (!value) {
      return "A dataset file must be selected";
    }
    return "";
  };

  const validateProblemType = (value) => {
    if (!value) {
      return "Problem type must be selected";
    }
    return "";
  };

  const validateOptimizer = (value) => {
    if (!value) {
      return "Optimizer must be selected";
    }
    return "";
  };

  const validateMetric = (value) => {
    if (!value) {
      return "Result metric must be selected";
    }
    return "";
  };

  const validateTargetField = (value) => {
    if (!value || value.trim() === "") {
      return "Target field must be specified";
    }
    return "";
  };

  // Real-time validation handler
  const updateValidationErrors = useCallback((field, value) => {
    let error = "";
    switch (field) {
      case "epochs":
        error = validateEpochs(value);
        break;
      case "batch_size":
        error = validateBatchSize(value);
        break;
      case "training_split":
        error = validateTrainingSplit(value);
        break;
      case "model":
        error = validateModel(value);
        break;
      case "file_id":
        error = validateFile(value);
        break;
      case "problem_type_id":
        error = validateProblemType(value);
        break;
      case "optimizer":
        error = validateOptimizer(value);
        break;
      case "metric":
        error = validateMetric(value);
        break;
      case "target_field":
        error = validateTargetField(value);
        break;
      default:
        break;
    }
    setValidationErrors((prev) => ({ ...prev, [field]: error }));
  }, []); // Empty deps since validation functions are stable within component

  // Check if form has any validation errors
  const hasValidationErrors = () => {
    return Object.values(validationErrors).some((error) => error !== "");
  };

  // Validate all fields
  const validateAllFields = () => {
    const errors = {
      model: validateModel(selectedModel),
      file_id: validateFile(trainingConfig.file_id),
      problem_type_id: validateProblemType(trainingConfig.problem_type_id),
      optimizer: validateOptimizer(trainingConfig.optimizer),
      metric: validateMetric(trainingConfig.metric),
      target_field: validateTargetField(trainingConfig.target_field),
      epochs: validateEpochs(trainingConfig.epochs),
      batch_size: validateBatchSize(trainingConfig.batch_size),
      training_split: validateTrainingSplit(trainingConfig.training_split),
    };
    setValidationErrors(errors);
    return !Object.values(errors).some((error) => error !== "");
  };

  const handleFileSelect = useCallback(
    (value) => {
      const normalizedValue = String(value);
      const selected = fileDetails.find((item) => String(item.file_id) === normalizedValue);
      if (selected && selected.fields && selected.fields.length > 0) {
        const fields = selected.fields.map((item, index) => ({
          label: item,
          value: item,
          key: index,
        }));
        setFieldsList(fields);
      } else {
        setFieldsList([]);
      }
      setTrainingConfig((prev) => ({ ...prev, file_id: value, target_field: "" }));
      updateValidationErrors("file_id", value);
      updateValidationErrors("target_field", "");
    },
    [fileDetails, updateValidationErrors],
  );

  const handleModelSelect = (value) => {
    setSelectedModel(value);
    setConfigSaved(false);
    updateValidationErrors("model", value);
  };

  const handleSaveConfig = () => {
    if (!selectedModel) return;

    // Final validation check before submitting
    if (!validateAllFields()) {
      return;
    }

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
    selectedModel &&
    trainingConfig.file_id &&
    trainingConfig.problem_type_id &&
    trainingConfig.optimizer &&
    trainingConfig.metric &&
    trainingConfig.epochs &&
    trainingConfig.batch_size &&
    trainingConfig.training_split &&
    trainingConfig.target_field &&
    !hasValidationErrors();

  const handleDownload = () => {
    if (selectedModel) {
      download_code(selectedModel, projectId).catch((error) => logger.error(error));
    }
  };

  const handleRun = () => {
    if (!selectedModel) return;

    // Final validation check before training
    if (!validateAllFields()) {
      return;
    }

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
      .then(() => {})
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

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Training</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            <div className="space-y-1">
              <Select onValueChange={handleModelSelect}>
                <SelectTrigger className={`w-64 ${validationErrors.model ? "border-red-500" : ""}`}>
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
              {validationErrors.model && (
                <p className="text-sm text-red-500">{validationErrors.model}</p>
              )}
            </div>
            <Button
              onClick={handleDownload}
              disabled={!selectedModel || !configSaved}
              variant="outline"
            >
              Download Code
            </Button>
            <Button
              onClick={handleRun}
              disabled={!selectedModel || !configSaved || isLoading || hasValidationErrors()}
            >
              {isLoading ? "Training..." : "Train"}
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
                  <SelectTrigger className={validationErrors.file_id ? "border-red-500" : ""}>
                    <SelectValue placeholder="Select a file" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {fileList.map((f) => (
                      <SelectItem key={f.key} value={f.value}>
                        {f.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.file_id && (
                  <p className="text-sm text-red-500">{validationErrors.file_id}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Problem Type</Label>
                <Select
                  onValueChange={(v) => {
                    setTrainingConfig((prev) => ({ ...prev, problem_type_id: v }));
                    updateValidationErrors("problem_type_id", v);
                    setConfigSaved(false);
                  }}
                >
                  <SelectTrigger
                    className={validationErrors.problem_type_id ? "border-red-500" : ""}
                  >
                    <SelectValue placeholder="Select problem type" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {problemTypeOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.problem_type_id && (
                  <p className="text-sm text-red-500">{validationErrors.problem_type_id}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Target Field</Label>
                <div className="relative">
                  <Input
                    type="text"
                    placeholder={
                      fieldsList.length === 0 && trainingConfig.file_id
                        ? "Enter target field name (e.g. species, class, target)"
                        : fieldsList.length === 0
                          ? "Select a dataset file first or enter field name"
                          : "Select from list or enter field name"
                    }
                    value={trainingConfig.target_field}
                    className={validationErrors.target_field ? "border-red-500" : ""}
                    onChange={(e) => {
                      const value = e.target.value;
                      setTrainingConfig((prev) => ({ ...prev, target_field: value }));
                      updateValidationErrors("target_field", value);
                      setConfigSaved(false);
                    }}
                    list={fieldsList.length > 0 ? "target-fields" : undefined}
                  />
                  {fieldsList.length > 0 && (
                    <datalist id="target-fields">
                      {fieldsList.map((f) => (
                        <option key={f.key} value={f.value}>
                          {f.label}
                        </option>
                      ))}
                    </datalist>
                  )}
                </div>
                {fieldsList.length > 0 && (
                  <p className="text-xs text-gray-500">
                    Available fields: {fieldsList.map((f) => f.label).join(", ")}
                  </p>
                )}
                {validationErrors.target_field && (
                  <p className="text-sm text-red-500">{validationErrors.target_field}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Optimizer</Label>
                <Select
                  onValueChange={(v) => {
                    setTrainingConfig((prev) => ({ ...prev, optimizer: v }));
                    updateValidationErrors("optimizer", v);
                    setConfigSaved(false);
                  }}
                >
                  <SelectTrigger className={validationErrors.optimizer ? "border-red-500" : ""}>
                    <SelectValue placeholder="Select optimizer" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {optimizerOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.optimizer && (
                  <p className="text-sm text-red-500">{validationErrors.optimizer}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Result Metrics</Label>
                <Select
                  onValueChange={(v) => {
                    setTrainingConfig((prev) => ({ ...prev, metric: v }));
                    updateValidationErrors("metric", v);
                    setConfigSaved(false);
                  }}
                >
                  <SelectTrigger className={validationErrors.metric ? "border-red-500" : ""}>
                    <SelectValue placeholder="Select metric" />
                  </SelectTrigger>
                  <SelectContent className="z-[9999] bg-white shadow-lg border backdrop-blur-sm">
                    {metricOptions.map((o) => (
                      <SelectItem key={o.key} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {validationErrors.metric && (
                  <p className="text-sm text-red-500">{validationErrors.metric}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Epochs</Label>
                <Input
                  type="number"
                  min="1"
                  placeholder="Number of epochs"
                  value={trainingConfig.epochs}
                  className={validationErrors.epochs ? "border-red-500" : ""}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({ ...prev, epochs: value }));
                    updateValidationErrors("epochs", value);
                    setConfigSaved(false);
                  }}
                />
                {validationErrors.epochs && (
                  <p className="text-sm text-red-500">{validationErrors.epochs}</p>
                )}
              </div>

              <div className="space-y-1">
                <Label>Batch Size</Label>
                <Input
                  type="number"
                  min="1"
                  placeholder="Batch size"
                  value={trainingConfig.batch_size}
                  className={validationErrors.batch_size ? "border-red-500" : ""}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({ ...prev, batch_size: value }));
                    updateValidationErrors("batch_size", value);
                    setConfigSaved(false);
                  }}
                />
                {validationErrors.batch_size && (
                  <p className="text-sm text-red-500">{validationErrors.batch_size}</p>
                )}
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
                  className={validationErrors.training_split ? "border-red-500" : ""}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({ ...prev, training_split: value }));
                    updateValidationErrors("training_split", value);
                    setConfigSaved(false);
                  }}
                />
                {validationErrors.training_split && (
                  <p className="text-sm text-red-500">{validationErrors.training_split}</p>
                )}
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
