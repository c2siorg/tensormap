import { useEffect, useCallback } from "react";
import { useParams } from "react-router-dom";
import PropTypes from "prop-types";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getAllFiles } from "../../services/FileServices";

const optimizerOptions = [
  { key: "opt_1", label: "Adam", value: "adam" },
  { key: "opt_2", label: "SGD", value: "sgd" },
];

const metricOptions = [
  { key: "acc_1", label: "Accuracy", value: "accuracy" },
  { key: "acc_2", label: "MSE", value: "mse" },
];

const problemTypeOptions = [
  { key: "prob_type_1", label: "Multi class classification [All values float]", value: "1" },
  { key: "prob_type_2", label: "Linear Regression [All values float]", value: "2" },
  { key: "prob_type_3", label: "Simple CNN Classifications", value: "3" },
];

function PropertiesBar({ formState, setFormState }) {
  const { projectId } = useParams();

  useEffect(() => {
    getAllFiles(projectId)
      .then((response) => {
        const fileList = response.map((file, index) => ({
          label: `${file.file_name}.${file.file_type}`,
          value: String(file.file_id),
          key: index,
        }));
        setFormState((prevState) => ({
          ...prevState,
          fileList,
          totalDetails: response,
        }));
      })
      .catch((err) => console.error(err));
  }, [setFormState, projectId]);

  const fileSelectHandler = useCallback(
    (value) => {
      const fileId = value;
      const selectedFileDetails = formState.totalDetails.filter((item) => item.file_id === fileId);
      const newFieldsList = selectedFileDetails[0].fields.map((item, index) => ({
        label: item,
        value: item,
        key: index,
      }));
      setFormState((prevState) => ({
        ...prevState,
        selectedFile: fileId,
        fieldsList: newFieldsList,
      }));
    },
    [formState.totalDetails, setFormState],
  );

  return (
    <div className="max-h-[60vh] space-y-3 overflow-y-auto rounded-md border bg-card p-4">
      <p className="text-center text-sm font-medium text-muted-foreground">Code Related</p>

      <div className="space-y-1">
        <Label>Model Name</Label>
        <Input
          placeholder="Model Name"
          onChange={(e) => setFormState((prev) => ({ ...prev, modalName: e.target.value }))}
        />
      </div>

      <div className="space-y-1">
        <Label>Problem Type</Label>
        <Select
          onValueChange={(v) => setFormState((prev) => ({ ...prev, problemType: Number(v) }))}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select Problem Type" />
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
        <Label>Optimizer</Label>
        <Select onValueChange={(v) => setFormState((prev) => ({ ...prev, optimizer: v }))}>
          <SelectTrigger>
            <SelectValue placeholder="Optimizer" />
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
        <Select onValueChange={(v) => setFormState((prev) => ({ ...prev, metric: v }))}>
          <SelectTrigger>
            <SelectValue placeholder="Result Metrics" />
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
          placeholder="No of Epochs"
          onChange={(e) => setFormState((prev) => ({ ...prev, epochCount: e.target.value }))}
        />
      </div>

      <div className="space-y-1">
        <Label>Batch Size</Label>
        <Input
          type="number"
          placeholder="Batch Size"
          onChange={(e) => setFormState((prev) => ({ ...prev, batchSize: e.target.value }))}
        />
      </div>

      <div className="space-y-1">
        <Label>File</Label>
        <Select onValueChange={fileSelectHandler} disabled={!formState.fileList || formState.fileList.length === 0}>
          <SelectTrigger>
            <SelectValue placeholder={!formState.fileList ? "Loading..." : formState.fileList.length === 0 ? "No files" : "Select a File"} />
          </SelectTrigger>
          <SelectContent>
            {formState.fileList?.map((f) => (
              <SelectItem key={f.key} value={f.value}>
                {f.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1">
        <Label>Target Field</Label>
        <Select onValueChange={(v) => setFormState((prev) => ({ ...prev, targetField: v }))} disabled={!formState.fieldsList || formState.fieldsList.length === 0}>
          <SelectTrigger>
            <SelectValue placeholder={!formState.fieldsList ? "Select a file first" : "Target field"} />
          </SelectTrigger>
          <SelectContent>
            {formState.fieldsList?.map((f) => (
              <SelectItem key={f.key} value={f.value}>
                {f.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1">
        <Label>Train:Test Ratio</Label>
        <Input
          type="number"
          placeholder="Train:Test ratio"
          onChange={(e) => setFormState((prev) => ({ ...prev, trainTestRatio: e.target.value }))}
        />
      </div>
    </div>
  );
}

PropertiesBar.propTypes = {
  formState: PropTypes.shape({
    fileList: PropTypes.array,
    totalDetails: PropTypes.array,
    fieldsList: PropTypes.array,
  }).isRequired,
  setFormState: PropTypes.func.isRequired,
};

export default PropertiesBar;
