import { useState, useEffect } from "react";
import PropTypes from "prop-types";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { downloadDataset, getCovMatrix, transformData } from "../../services/FileServices";
import TransformationCreator from "./TransformationCreator";
import TransformationList from "./TransformationList";
import FeedbackDialog from "../shared/FeedbackDialog";
import logger from "../../shared/logger";

/**
 * CSV data preprocessing panel.
 *
 * Loads column names and lets users build a list of per-column
 * transformations, then submits them to the backend.
 *
 * @param {{ fileId: string, updateFileList: () => void }} props
 */
function PreprocessData({ fileId, fileName, updateFileList }) {
  const [data, setData] = useState(null);
  const [transformations, setTransformations] = useState([]);
  const [disableButton, setDisableButton] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalOpenDuplicateFeature, setModalOpenDuplicateFeature] = useState(false);
  const [datasetCreatedSuccesfully, setDatasetCreatedSuccesfully] = useState(true);
  const [message, setMessage] = useState("");
  const [fetchError, setFetchError] = useState(null);
  const [downloadError, setDownloadError] = useState(null);
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setFetchError(null);
        const response = await getCovMatrix(fileId);
        setData(Object.keys(response.data_types));
      } catch (e) {
        logger.error(e);
        setFetchError(e.message);
        setData(null);
      }
    };
    fetchData();
  }, [fileId]);

  useEffect(() => {
    setDisableButton(transformations.length === 0);
  }, [transformations]);

  const handleAddTransformation = (transformation) => {
    if (transformations.some((t) => t.feature === transformation.feature)) {
      setModalOpenDuplicateFeature(true);
    } else setTransformations((prev) => [...prev, transformation]);
  };

  const handleDeleteTransformation = (index) => {
    setTransformations((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async () => {
    try {
      const dataTransformed = await transformData(fileId, transformations);
      if (dataTransformed) {
        setDatasetCreatedSuccesfully(true);
        setMessage(dataTransformed.message);
        setModalOpen(true);
        updateFileList();
      } else {
        setDatasetCreatedSuccesfully(false);
        setMessage("Transformation failed");
        setModalOpen(true);
      }
    } catch (e) {
      logger.error("Error transforming data:", e);
      setDatasetCreatedSuccesfully(false);
      setMessage(e.message || "An error occurred during transformation");
      setModalOpen(true);
    }
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    setDownloadError(null);
    try {
      const response = await downloadDataset(fileId);
      const contentDisposition = response.headers?.["content-disposition"] || "";
      const encodedFilename = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
      const regularFilename = contentDisposition.match(/filename="?([^";]+)"?/i)?.[1];
      let filename = fileName || "dataset";
      if (encodedFilename) {
        try {
          filename = decodeURIComponent(encodedFilename);
        } catch (error) {
          logger.warn("Could not decode dataset download filename:", error);
        }
      } else if (regularFilename) {
        filename = regularFilename;
      }
      if (!filename.toLowerCase().endsWith(".csv")) {
        filename += ".csv";
      }

      const blobUrl = window.URL.createObjectURL(response.data);
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => window.URL.revokeObjectURL(blobUrl), 200);
    } catch (error) {
      logger.error("Error downloading dataset:", error);
      let errorMessage = error.message || "Failed to download dataset";
      const responseData = error.response?.data;
      if (responseData instanceof Blob) {
        try {
          const errorBody = JSON.parse(await responseData.text());
          errorMessage = errorBody.detail || errorBody.message || errorMessage;
        } catch {
          // Keep the request error message when the response is not a JSON error.
        }
      } else if (responseData?.detail || responseData?.message) {
        errorMessage = responseData.detail || responseData.message;
      }
      setDownloadError(errorMessage);
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div>
      <FeedbackDialog
        open={modalOpenDuplicateFeature}
        onClose={() => setModalOpenDuplicateFeature(false)}
        success={false}
        message="Multiple Transformations for the same feature found"
      />
      <FeedbackDialog
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        success={datasetCreatedSuccesfully}
        message={message}
      />
      {fetchError && <div className="mb-4 text-sm text-destructive">{fetchError}</div>}
      {downloadError && (
        <div className="mb-4 text-sm text-destructive" role="alert">
          {downloadError}
        </div>
      )}
      <TransformationCreator features={data ? data : []} onAdd={handleAddTransformation} />
      {transformations.length > 0 && (
        <TransformationList
          transformations={transformations}
          onDelete={handleDeleteTransformation}
        />
      )}
      <Button onClick={handleSubmit} disabled={disableButton} className="mt-4">
        Apply Transformations
      </Button>
      <Button
        onClick={handleDownload}
        disabled={isDownloading}
        variant="outline"
        className="mt-4 ml-2"
      >
        <Download className="mr-2 h-4 w-4" aria-hidden="true" />
        {isDownloading ? "Downloading..." : "Download CSV"}
      </Button>
    </div>
  );
}

PreprocessData.propTypes = {
  fileId: PropTypes.string.isRequired,
  fileName: PropTypes.string,
  updateFileList: PropTypes.func.isRequired,
};

export default PreprocessData;
