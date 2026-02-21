import { useState, useEffect } from "react";
import io from "socket.io-client";
import { useRecoilState } from "recoil";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import * as urls from "../../constants/Urls";
import * as strings from "../../constants/Strings";
import Result from "./Result/Result";
import { download_code, runModel, getAllModels } from "../../services/ModelServices";
import { models as modelListAtom } from "../../shared/atoms";

function ResultPanel() {
  const [modelList, setModelList] = useRecoilState(modelListAtom);

  const [selectedModel, setSelectedModel] = useState(null);
  const [disableDownloadButton, setDisableDownloadButton] = useState(true);
  const [disableRunButton, setDisableRunButton] = useState(true);
  const [disableClearButton, setDisableClearButton] = useState(true);
  const [resultValues, setResultValues] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    try {
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

      getAllModels().then((response) => {
        const models = response.map((file, index) => ({
          label: file + strings.MODEL_EXTENSION,
          value: file,
          key: index,
        }));
        setModelList(models);
      });

      return () => {
        socket.off(strings.DL_RESULT_LISTENER, dlResultListener);
      };
    } catch (error) {
      console.error("Error connecting to the socket:", error);
    }
  }, []);

  const modelSelectHandler = (value) => {
    setSelectedModel(value);
    setDisableDownloadButton(false);
    setDisableRunButton(false);
  };

  const modelDownloadHandler = () => {
    if (selectedModel != null) {
      download_code(selectedModel).catch((error) => console.error(error));
    }
  };

  const runButtonHandler = () => {
    setResultValues([]);
    setIsLoading(true);
    if (selectedModel != null) {
      runModel(selectedModel)
        .then(() => {
          setDisableRunButton(false);
          setDisableClearButton(false);
        })
        .catch((error) => {
          console.error(error.response?.data);
          setResultValues([error.response?.data?.message ?? "An error occurred"]);
          setIsLoading(false);
          setDisableClearButton(false);
        });

      setDisableRunButton(true);
    }
  };

  const clearButtonHandler = () => {
    setResultValues([]);
    setIsLoading(false);
    setDisableClearButton(true);
  };

  let results = null;

  if (isLoading && resultValues.length === 0) {
    results = (
      <Card className="mt-2">
        <CardContent className="flex h-48 items-center justify-center text-muted-foreground">
          Model is running please wait ...
        </CardContent>
      </Card>
    );
  }

  if (resultValues.length > 0) {
    results = resultValues.map((item, index) => <Result result={item} key={index} />);
  }

  return (
    <div>
      <div className="flex items-center gap-4 rounded-md border p-3">
        <span className="text-sm font-medium">Model Execution</span>
        <Select onValueChange={modelSelectHandler}>
          <SelectTrigger className="w-48">
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
        <div className="ml-auto flex gap-2">
          <Button variant="outline" onClick={modelDownloadHandler} disabled={disableDownloadButton}>
            Download Code
          </Button>
          <Button onClick={runButtonHandler} disabled={disableRunButton}>
            Model Run
          </Button>
          <Button variant="secondary" onClick={clearButtonHandler} disabled={disableClearButton}>
            Clear Panel
          </Button>
        </div>
      </div>
      <div>{results}</div>
    </div>
  );
}

export default ResultPanel;
