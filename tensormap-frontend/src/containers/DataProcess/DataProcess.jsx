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
      setSelectedFileId(String(files[0].file_id));
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
          setSelectedFileId(String(response[0].file_id));
        }
      })
      .catch((error) => {
        logger.error("Error loading files:", error);
        setFiles([]);
        setSelectedFileId(null);
      });
  };

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
                </option>
              ))}
            </select>
          </div>

          {selectedFileId && (
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
                    <PreprocessData fileId={selectedFileId} updateFileList={refreshFiles} />
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
          )}
        </div>
      )}
    </div>
  );
}

export default DataProcess;
