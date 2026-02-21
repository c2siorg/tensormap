import { useState, useEffect } from "react";
import PropTypes from "prop-types";
import { Button } from "@/components/ui/button";
import { getCovMatrix, transformData } from "../../services/FileServices";
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
function PreprocessData({ fileId, updateFileList }) {
  const [data, setData] = useState(null);
  const [transformations, setTransformations] = useState([]);
  const [disableButton, setDisableButton] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [modalOpenDuplicateFeature, setModalOpenDuplicateFeature] = useState(false);
  const [datasetCreatedSuccesfully, setDatasetCreatedSuccesfully] = useState(true);
  const [message, setMessage] = useState("");
  const [fetchError, setFetchError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setFetchError(null);
        const response = await getCovMatrix(fileId);
        setData(Object.keys(response.data_types));
      } catch (e) {
        logger.error(e);
        setFetchError("Failed to load column data");
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
    </div>
  );
}

PreprocessData.propTypes = {
  fileId: PropTypes.string.isRequired,
  updateFileList: PropTypes.func.isRequired,
};

export default PreprocessData;
