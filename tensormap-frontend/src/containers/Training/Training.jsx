import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import io from "socket.io-client";
import { useRecoilState } from "recoil";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import * as urls from "../../constants/Urls";
import * as strings from "../../constants/Strings";
import Result from "../../components/ResultPanel/Result/Result";
import { download_code, runModel, getAllModels } from "../../services/ModelServices";
import { models as modelListAtom } from "../../shared/atoms";

export default function Training() {
  const { projectId } = useParams();
  const [modelList, setModelList] = useRecoilState(modelListAtom);

  const [selectedModel, setSelectedModel] = useState(null);
  const [resultValues, setResultValues] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const socket = io(urls.WS_DL_RESULTS);

    const dlResultListener = (resp) => {
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

    getAllModels(projectId).then((response) => {
      const models = response.map((file, index) => ({
        label: file + strings.MODEL_EXTENSION,
        value: file,
        key: index,
      }));
      setModelList(models);
    });

    return () => {
      socket.off(strings.DL_RESULT_LISTENER, dlResultListener);
      socket.disconnect();
    };
  }, [projectId, setModelList]);

  const handleDownload = () => {
    if (selectedModel) {
      download_code(selectedModel).catch((error) => console.error(error));
    }
  };

  const handleRun = () => {
    if (!selectedModel) return;
    setResultValues([]);
    setIsLoading(true);
    runModel(selectedModel)
      .then(() => {})
      .catch((error) => {
        console.error(error.response?.data);
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
            <Select onValueChange={(value) => setSelectedModel(value)}>
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
            <Button onClick={handleDownload} disabled={!selectedModel} variant="outline">
              Download Code
            </Button>
            <Button onClick={handleRun} disabled={!selectedModel || isLoading}>
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
