import { useCallback, useEffect, useRef, useState } from "react";
import { useSetRecoilState } from "recoil";
import { useParams } from "react-router-dom";
import { Upload, FileSpreadsheet } from "lucide-react";
import { projectFiles } from "../../shared/atoms";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import * as strings from "../../constants/Strings";
import { getAllFiles, uploadFile, deleteFile } from "../../services/FileServices";
import FeedbackDialog from "../../components/shared/FeedbackDialog";
import logger from "../../shared/logger";

/**
 * Single-dataset upload page for a project workspace.
 *
 * Displays either an empty state with a file picker or the currently
 * uploaded CSV with Replace/Remove actions. Only one dataset per project.
 */
function DataUpload() {
  const { projectId } = useParams();
  const fileInputRef = useRef(null);
  const setProjectFiles = useSetRecoilState(projectFiles);
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackSuccess, setFeedbackSuccess] = useState(true);
  const [feedbackMessage, setFeedbackMessage] = useState("");

  const fetchDataset = useCallback(() => {
    setLoading(true);
    getAllFiles(projectId)
      .then((response) => {
        setProjectFiles(response);
        if (response.length > 0) {
          const file = response[0];
          setDataset({
            id: file.file_id,
            name: `${file.file_name}.${file.file_type}`,
            columns: file.fields?.length ?? 0,
            rows: file.row_count ?? 0,
          });
        } else {
          setDataset(null);
        }
      })
      .catch((error) => {
        logger.error("Error retrieving files:", error);
        setDataset(null);
      })
      .finally(() => setLoading(false));
  }, [projectId, setProjectFiles]);

  useEffect(() => {
    fetchDataset();
  }, [fetchDataset]);

  const handleUpload = async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      await uploadFile(file, projectId);
      setFeedbackSuccess(true);
      setFeedbackMessage(strings.UPLOAD_SUCCESS_MODEL_MESSAGE);
      fetchDataset();
    } catch (error) {
      logger.error("Upload error:", error);
      setFeedbackSuccess(false);
      setFeedbackMessage(error.message || "Failed to upload file");
    } finally {
      setUploading(false);
      setFeedbackOpen(true);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleReplace = async (file) => {
    if (!file || !dataset) return;
    setUploading(true);
    try {
      await deleteFile(dataset.id);
      await uploadFile(file, projectId);
      setFeedbackSuccess(true);
      setFeedbackMessage("Dataset replaced successfully");
      fetchDataset();
    } catch (error) {
      logger.error("Replace error:", error);
      setFeedbackSuccess(false);
      setFeedbackMessage(error.message || "Failed to replace dataset");
      fetchDataset();
    } finally {
      setUploading(false);
      setFeedbackOpen(true);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleRemove = async () => {
    if (!dataset) return;
    setUploading(true);
    try {
      await deleteFile(dataset.id);
      setDataset(null);
      setProjectFiles([]);
      setFeedbackSuccess(true);
      setFeedbackMessage("Dataset removed successfully");
    } catch (error) {
      logger.error("Remove error:", error);
      setFeedbackSuccess(false);
      setFeedbackMessage(error.message || "Failed to remove dataset");
    } finally {
      setUploading(false);
      setFeedbackOpen(true);
    }
  };

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (dataset) {
      handleReplace(file);
    } else {
      handleUpload(file);
    }
  };

  return (
    <>
      <FeedbackDialog
        open={feedbackOpen}
        onClose={() => setFeedbackOpen(false)}
        success={feedbackSuccess}
        message={feedbackMessage}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={onFileChange}
      />
      <div className="mx-auto max-w-lg">
        <Card>
          <CardHeader>
            <CardTitle>{strings.UPLOAD_NEW_FILE_TITLE}</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex h-48 items-center justify-center text-muted-foreground">
                Loading...
              </div>
            ) : dataset ? (
              <div className="flex flex-col items-center gap-4 py-6">
                <FileSpreadsheet className="h-12 w-12 text-primary" />
                <p className="text-lg font-medium">{dataset.name}</p>
                <div className="flex gap-4 text-sm text-muted-foreground">
                  <span>Columns: {dataset.columns}</span>
                  <span>Rows: {dataset.rows.toLocaleString()}</span>
                </div>
                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    disabled={uploading}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    {uploading ? "Uploading..." : "Replace"}
                  </Button>
                  <Button variant="destructive" disabled={uploading} onClick={handleRemove}>
                    Remove
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-4 py-6">
                <Upload className="h-12 w-12 text-muted-foreground" />
                <p className="text-lg font-medium">{strings.DATASET_EMPTY_TITLE}</p>
                <p className="text-sm text-muted-foreground">{strings.DATASET_EMPTY_DESC}</p>
                <Button disabled={uploading} onClick={() => fileInputRef.current?.click()}>
                  {uploading ? "Uploading..." : "Choose CSV File"}
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}

export default DataUpload;
