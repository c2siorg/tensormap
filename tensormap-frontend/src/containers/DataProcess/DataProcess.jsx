import { useState, useEffect } from "react";
import { useRecoilState } from "recoil";
import { useParams } from "react-router-dom";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import ColumnStatsPanel from "../../components/Process/ColumnStatsPanel";
import DisplayDataset from "../../components/Process/DisplayDataset";
import SelectFileModal from "../../components/Process/SelectFileModal";
import PreprocessData from "../../components/Process/PreprocessData";
import CorrelationHeatmap from "../../components/Process/CorrelationHeatmap";
import { getAllFiles } from "../../services/FileServices";
import { projectFiles } from "../../shared/atoms";
import logger from "../../shared/logger";

/**
 * Data preprocessing page. Displays dataset preview, statistics, and
 * preprocessing tools. Supports multiple datasets per project.
 */
function DataProcess() {
  const { projectId } = useParams();
  const [files, setFiles] = useRecoilState(projectFiles);
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (files.length > 0 && !selectedFileId) {
      const firstCsv = files.find((f) => f.file_type === "csv");
      setSelectedFileId(String(firstCsv ? firstCsv.file_id : files[0].file_id));
    }
    if (files.length === 0) {
      setSelectedFileId(null);
    }
    setIsLoading(false);
  }, [files, selectedFileId]);

  const refreshFiles = () => {
    getAllFiles(projectId)
      .then((response) => {
        setFiles(response);
        if (response.length > 0 && !selectedFileId) {
          const firstCsv = response.find((f) => f.file_type === "csv");
          setSelectedFileId(String(firstCsv ? firstCsv.file_id : response[0].file_id));
        }
      })
      .catch((error) => {
        logger.error("Error loading files:", error);
        setFiles([]);
        setSelectedFileId(null);
      });
  };

  const selectedFile = files.find((f) => String(f.file_id) === String(selectedFileId));
  const isNonCsv = selectedFile && selectedFile.file_type && selectedFile.file_type !== "csv";

  return (
    <div className="space-y-6">
      {isLoading ? (
        <div className="flex h-48 items-center justify-center">
          <span className="text-muted-foreground">Loading...</span>
        </div>
      ) : files.length === 0 ? (
        <SelectFileModal />
      ) : (
        <div className="space-y-6">
          {/* Dataset selector */}
          <div className="flex items-center gap-4">
            <label htmlFor="file-select" className="font-medium text-gray-700">
              Select Dataset:
            </label>
            <select
              id="file-select"
              value={selectedFileId || ""}
              onChange={(e) => setSelectedFileId(e.target.value)}
              className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="" disabled>
                Choose a dataset
              </option>
              {files.map((file) => (
                <option key={file.file_id} value={String(file.file_id)}>
                  {file.file_name}
                  {file.file_type && file.file_type !== "csv"
                    ? ` (${file.file_type.toUpperCase()} - Image Dataset)`
                    : ""}
                </option>
              ))}
            </select>
          </div>

          {isNonCsv ? (
            <Card className="border-amber-200 bg-amber-50/50">
              <CardHeader>
                <CardTitle className="text-base font-semibold text-amber-800">
                  Image Dataset Selected
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-amber-700">
                <p>
                  The selected file <strong>{selectedFile.file_name}</strong> is an image dataset
                  archive ({selectedFile.file_type?.toUpperCase() || "ZIP"}). Tabular
                  operations—such as CSV data preview, column statistics, correlation heatmaps, and
                  preprocessing transformations—only apply to tabular CSV datasets.
                </p>
                <p>
                  To configure and train image classification models using this dataset, visit the{" "}
                  <strong>Deep Learning</strong> canvas and select this image dataset in model
                  configurations.
                </p>
              </CardContent>
            </Card>
          ) : (
            selectedFileId && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <Card>
                    <CardHeader>
                      <CardTitle>Dataset Preview</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <DisplayDataset fileId={selectedFileId} />
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>Preprocessing</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <PreprocessData
                        fileId={selectedFileId}
                        fileName={selectedFile?.file_name}
                        updateFileList={refreshFiles}
                      />
                    </CardContent>
                  </Card>
                </div>
                <Card>
                  <CardContent className="pt-4">
                    <ColumnStatsPanel fileId={selectedFileId} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Correlation Heatmap</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CorrelationHeatmap fileId={selectedFileId} />
                  </CardContent>
                </Card>
              </div>
            )
          )}
        </div>
      )}
    </div>
  );
}

export default DataProcess;
